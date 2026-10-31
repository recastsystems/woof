"""The engine range, the drift check, and the two places that must name it.

THE DEFECT ALL OF THIS ANSWERS, measured on a real install on 2026-08-27
(``evidence/userwalk-20260827/``): ``pip install woof hex`` resolved woof
2.5.7, whose bytes this port's sixteen-file seam manifest refuses; the
forecast lane stopped with two SHA-256 digests and no version number,
``woof hex doctor`` reported the estate healthy and exited 0, and no
document in the distribution named a version that works.

Three surfaces, one fact.  The declaration bounds what pip may resolve, the
report names the drift at install time, and the door names it at launch.
Every test below is written so that it fails when the surface stops doing
that -- the instrument is validated in both directions, with real bytes and
real SHA-256, and never against a stub of itself.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from woof.hex import doctor
from woof.hex import engine_pin
from woof.hex import forecast_door as door

#: (WOOF) This tree is the engine and the hex model in one distribution, so the requirement names it
#: (engine_pin.DISTRIBUTION); the measurement records keep the engine's own spellings (gpuwm>=, gpuwm/),
#: which name the same files here as woof/.
DIST = engine_pin.DISTRIBUTION


def _folded(value):
    if isinstance(value, dict):
        return {_folded(k): _folded(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_folded(v) for v in value]
    if isinstance(value, str):
        if value.startswith("gpuwm/"):
            return "woof/" + value[len("gpuwm/"):]
        if value.startswith("gpuwm>="):
            return DIST + value[len("gpuwm"):]
    return value


# ---------------------------------------------------------------------------
# the derivation
# ---------------------------------------------------------------------------
def _row(version: str, *, on_pypi=True, moved=(), buildable=True):
    return engine_pin.PublishedEngine(
        version=version,
        on_pypi=on_pypi,
        moved=tuple(moved),
        offline_build_road=buildable,
    )


def test_the_floor_is_the_lowest_engine_that_is_all_three_things(monkeypatch):
    """Installable, byte-compatible, buildable from.  Any one missing is out.

    Each clause has a measured victim.  ``2.5.5`` satisfies the manifest and
    is a git tag with NO PyPI release, so a floor there named a version pip
    cannot resolve; and its published tree is missing a file of the vendored
    ``cc`` crate, so the manual's own printed ``cargo build --offline`` dies
    there while the same command finishes at 2.5.6.
    """

    monkeypatch.setattr(engine_pin, "PUBLISHED_ENGINES", (
        _row("2.5.4", moved=("woof/config.py",)),
        _row("2.5.5", on_pypi=False, buildable=False),
        _row("2.5.6"),
        _row("2.5.7", moved=("woof/core/microphysics.py",)),
    ))
    assert engine_pin.measured_gpuwm_floor() == "2.5.6"

    # Teeth, one clause at a time: drop each disqualifier and the floor moves
    # down to 2.5.5, which proves the clause was doing the work.
    monkeypatch.setattr(engine_pin, "PUBLISHED_ENGINES", (
        _row("2.5.5"),
        _row("2.5.6"),
    ))
    assert engine_pin.measured_gpuwm_floor() == "2.5.5"


def test_the_range_is_the_floors_whole_minor_so_pip_never_downgrades(monkeypatch):
    """Plan call 14 (open defect B2): an exact-patch range downgraded the engine.

    0.3.3's ``<2.8.1`` made ``pip install woof hex`` replace a user's 2.8.7
    with 2.8.0.  A measured row inside the minor that moved the seam no
    longer lowers the ceiling; the helper binding refuses that engine by
    name at the door instead.
    """

    monkeypatch.setattr(engine_pin, "GPUWM_INSTALL_MINIMUM", "2.5.0")
    monkeypatch.setattr(engine_pin, "PUBLISHED_ENGINES", (
        _row("2.5.6"),
        _row("2.5.7", moved=("woof/io/restart.py",)),
    ))
    assert engine_pin.gpuwm_ceiling() == "2.6"
    assert engine_pin.gpuwm_requirement() == f"{DIST}>=2.5.6,<2.6"


def test_the_2_9_0_pin_derives_the_call_14_range(monkeypatch):
    monkeypatch.setattr(engine_pin, "GPUWM_INSTALL_MINIMUM", "2.5.0")
    monkeypatch.setattr(engine_pin, "PUBLISHED_ENGINES", (
        _row("2.8.7", moved=("woof/core/physics.py",)),
        _row("2.8.9", moved=("woof/core/physics.py",)),
        _row("2.9.0"),
        _row("2.9.1", moved=("woof/core/rrtmg_legacy.py",)),
    ))
    assert engine_pin.gpuwm_requirement() == f"{DIST}>=2.9.0,<2.10"


def test_an_unmeasured_minor_is_never_admitted(monkeypatch):
    """The property that stops the NEXT engine minor re-opening this.

    The ceiling is the next minor after the floor -- never "unbounded".  An
    engine minor nobody has compared against the manifest is exactly the
    engine the original defect resolved onto.
    """

    monkeypatch.setattr(engine_pin, "GPUWM_INSTALL_MINIMUM", "2.5.0")
    monkeypatch.setattr(engine_pin, "PUBLISHED_ENGINES", (
        _row("2.5.6"),
        _row("2.5.7"),
    ))
    assert engine_pin.gpuwm_ceiling() == "2.6"
    assert engine_pin.gpuwm_requirement() == f"{DIST}>=2.5.6,<2.6"


def test_no_usable_engine_is_a_refusal_rather_than_a_silent_range(monkeypatch):
    monkeypatch.setattr(engine_pin, "PUBLISHED_ENGINES", (
        _row("2.5.7", moved=("woof/io/restart.py",)),
    ))
    with pytest.raises(engine_pin.EnginePinError) as raised:
        engine_pin.gpuwm_requirement()
    assert "measure_engine_verdicts.py" in str(raised.value)


@pytest.mark.parametrize("measured,minimum,expected,ceiling", [
    ("2.9.0", "2.9.2", "2.9.2", "2.10"),
    ("3.0.1", "2.9.2", "3.0.1", "3.1"),
])
def test_installation_policy_does_not_relabel_measured_bytes(
    monkeypatch, measured, minimum, expected, ceiling,
):
    rows = (_row(measured),)
    monkeypatch.setattr(engine_pin, "PUBLISHED_ENGINES", rows)
    monkeypatch.setattr(engine_pin, "GPUWM_INSTALL_MINIMUM", minimum)
    assert engine_pin.measured_gpuwm_floor() == measured
    assert engine_pin.wanted_version() == measured
    assert engine_pin.pinned_engine().version == measured
    assert engine_pin.gpuwm_floor() == expected
    assert engine_pin.gpuwm_requirement() == f"{DIST}>={expected},<{ceiling}"
    assert engine_pin.PUBLISHED_ENGINES == rows
    if expected != measured:
        assert engine_pin.engine(expected) is None


def test_shipped_installation_minimum_keeps_the_release_range():
    assert engine_pin.GPUWM_INSTALL_MINIMUM == "2.9.11"
    assert engine_pin.gpuwm_requirement() == f"{DIST}>=2.9.11,<2.10"


def _repin_documents() -> dict[str, dict]:
    """Every repin measurement of record, keyed by the engine version it pinned."""

    import json

    found = {}
    for path in sorted((ROOT / "verification" / "engine-verdicts").glob("repin-*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        found[document["pin"]["version"]] = dict(_folded(document), _path=path)
    return found


def _version_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


#: The measurement the shipped table is rendered from: the repin JSON whose
#: pin is the measured seam floor. Under ``verification/`` rather than
#: ``evidence/`` because it is the table's SOURCE, not a record of a run: the
#: sdist grafts ``verification/`` and holds ``evidence/`` out, so this gate
#: runs on every published surface instead of skipping there.  Derived rather
#: than named, so an engine repin is the instrument's output plus its JSON,
#: with no version in this file to transcribe -- the transcriptions are what
#: the engine_pin docstring counts going wrong.
VERDICTS_OF_RECORD = _repin_documents()[engine_pin.measured_gpuwm_floor()]["_path"]


def test_the_shipped_table_says_what_the_measurement_said():
    """Only the measured seam may resolve after a repin."""
    documents = _repin_documents()
    pinned = engine_pin.measured_gpuwm_floor()
    verdicts = documents[pinned]
    # The record's requirement was derived under the policy of its day (a
    # one-patch range until plan call 14); the floor it measured is what
    # still binds, and the range is that floor's whole minor.
    assert verdicts["requirement"].startswith(f"{DIST}>={pinned},")
    assert engine_pin.gpuwm_requirement() == f"{DIST}>={engine_pin.gpuwm_floor()},<{engine_pin.gpuwm_ceiling()}"
    assert [row.version for row in engine_pin.PUBLISHED_ENGINES if row.usable] == [pinned]
    current = engine_pin.engine(pinned)
    assert current.on_pypi and current.offline_build_road and current.moved == ()
    # The previous pin cannot silently remain installable.
    previous = [v for v in documents if _version_key(v) < _version_key(pinned)]
    assert previous, "no earlier repin of record to hold the previous pin to"
    assert engine_pin.engine(max(previous, key=_version_key)).moved
    for row in engine_pin.PUBLISHED_ENGINES:
        if row.version != pinned:
            assert row.moved, f"unqualified engine {row.version} claims the new seam"
    # Tag-only rows and the incomplete old build road retain their refusals.
    assert not engine_pin.engine("2.5.5").on_pypi
    assert not engine_pin.engine("2.5.5").offline_build_road
    assert not engine_pin.engine("2.6.2").on_pypi
    assert not engine_pin.engine("2.8.2").on_pypi


def test_the_table_is_the_instruments_output_not_a_transcription():
    """The shipped rows must equal the shipped JSON, field for field, and
    the generated block must be the instrument's own rendering of it.

    THE BREAKAGE THIS PREVENTS is the one this module's docstring counts:
    three separate hand-transcriptions of these figures, three different
    wrong answers.  The defence is that the table is rendered from the
    instrument's JSON, so this test re-renders it -- with the instrument's
    own renderer, imported from ``tools/`` -- and demands equality, which
    fails on a hand-patched row even when the patch looks plausible.

    Until 2026-09-13 this read a receipt under ``evidence/`` and took the
    ``receipts`` fixture, so on the assembled public tree and in the sdist
    it SKIPPED: the one gate that keeps the table honest was disarmed on
    exactly the surfaces a user installs from.  The JSON now ships beside
    the contracts and the gate runs everywhere.
    """

    import importlib.util
    import json

    assert VERDICTS_OF_RECORD.is_file(), (
        f"{VERDICTS_OF_RECORD.relative_to(ROOT).as_posix()} is missing; the "
        "table has no measurement to be rendered from"
    )
    verdicts = _folded(json.loads(VERDICTS_OF_RECORD.read_text(encoding="utf-8")))
    assert verdicts["pin"]["version"] == engine_pin.measured_gpuwm_floor()
    assert verdicts["instrument"] == "tools/repin_source_tables.py"
    measured = {
        row["tag"].lstrip("v"): (
            bool(row["on_pypi"]),
            tuple(row["moved"]),
            bool(row["offline_build_road"]),
        )
        for row in verdicts["tags"]
    }
    shipped = {
        row.version: (row.on_pypi, row.moved, row.offline_build_road)
        for row in engine_pin.PUBLISHED_ENGINES
    }
    assert shipped == measured, (
        "PUBLISHED_ENGINES disagrees with the JSON it is supposed to be "
        "rendered from.  Re-splice it: python tools/measure_engine_verdicts.py "
        "--offline --engine-git <arwen clone> --wheel-cache <cache> "
        f"--pin-to {engine_pin.measured_gpuwm_floor()} --splice"
    )

    # The manifest the JSON was measured against is the one the package
    # ships: a JSON re-measured against some other bytes would render the
    # same shape of table for the wrong pin.
    assert verdicts["manifest"] == engine_pin.seam_manifest()

    spec = importlib.util.spec_from_file_location(
        "_measure_engine_verdicts", ROOT / "tools" / "measure_engine_verdicts.py"
    )
    assert spec is not None and spec.loader is not None
    instrument = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(instrument)
    source = (ROOT / ".." / ".." / "woof" / "hex" / "engine_pin.py").read_text(encoding="utf-8")
    start = source.index(instrument.BEGIN_MARK)
    end = source.index(instrument.END_MARK) + len(instrument.END_MARK)
    assert source[start:end] == instrument.render_rows(verdicts["tags"]), (
        "the generated block in engine_pin.py is not the instrument's own "
        "rendering of the JSON of record; it was edited by hand"
    )


# ---------------------------------------------------------------------------
# the byte check, on real bytes
# ---------------------------------------------------------------------------
def _seam_tree(tmp_path: Path, version: str) -> tuple[Path, dict[str, str]]:
    """A fake woof checkout and the manifest that pins its exact bytes.

    Real files, real SHA-256, real reads.  The manifest is computed FROM the
    bytes written, so the clean case is genuinely clean and every drift below
    is a byte that actually moved rather than a mocked verdict.
    """

    root = tmp_path / f"gpuwm-{version}"
    payloads = {
        "woof/core/mpas_column_batch.py": b"def run_mpas_column_batch(): ...\n",
        "woof/core/microphysics.py": b"def apply(): ...\n",
        "woof/io/restart.py": b"RESTART = 1\n",
        "docs/mpas-seam.md": b"# the seam\n",
    }
    for relative, payload in payloads.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "woof"\nversion = "{version}"\n', encoding="utf-8"
    )
    manifest = {
        relative: sha256(payload).hexdigest()
        for relative, payload in payloads.items()
    }
    return root, manifest


def test_a_matching_tree_reports_no_drift(tmp_path, monkeypatch):
    root, manifest = _seam_tree(tmp_path, "2.5.6")
    monkeypatch.setattr(engine_pin, "seam_manifest", lambda: manifest)

    inspection = engine_pin.inspect_seam(root)
    assert not inspection.drifted
    assert inspection.absent == ()
    assert inspection.checked == len(manifest)
    assert door.seam_pin_problem(root) is None


def test_a_moved_byte_is_found_and_a_missing_file_is_a_different_finding(
    tmp_path, monkeypatch
):
    root, manifest = _seam_tree(tmp_path, "2.5.7")
    monkeypatch.setattr(engine_pin, "seam_manifest", lambda: manifest)

    (root / "woof/core/microphysics.py").write_bytes(b"def apply(): pass\n")
    (root / "docs/mpas-seam.md").unlink()

    inspection = engine_pin.inspect_seam(root)
    assert inspection.moved == ("woof/core/microphysics.py",)
    assert inspection.absent == ("docs/mpas-seam.md",)
    assert inspection.drifted


def test_the_checkout_version_is_read_from_the_tree_not_from_the_install(
    tmp_path, monkeypatch
):
    """A refusal that names a version must name the version ON DISK.

    ``woof/__init__.py`` reads its own version back out of INSTALLED
    metadata, so importing it from a checkout answers with whatever pip has
    rather than what the user pointed at -- the exact confusion this refusal
    exists to end.
    """

    root, _ = _seam_tree(tmp_path, "2.5.7")
    assert engine_pin.checkout_version(root) == "2.5.7"
    assert engine_pin.checkout_version(tmp_path / "not-a-checkout") is None


# ---------------------------------------------------------------------------
# the door's refusal names a version
# ---------------------------------------------------------------------------
def test_a_moved_engine_runs_and_the_door_records_what_moved(
    tmp_path, monkeypatch
):
    """RETIRED 2026-10-10 as a gate: an engine whose seam bytes are not the
    pinned version's is not refused.  The door used to answer "the seam is
    pinned to woof X ... running against moved bytes would be running
    physics nobody proved", which names missing evidence and no breakage,
    and it closed the forecast on every engine newer than the pin (8-card RTX PRO 6000 host:
    woof 2.8.9 against the 2.8.7 pin).  The match is recorded instead.
    """

    root, manifest = _seam_tree(tmp_path, "2.5.7")
    monkeypatch.setattr(engine_pin, "seam_manifest", lambda: manifest)
    (root / "woof/core/microphysics.py").write_bytes(b"moved\n")

    assert door.seam_pin_problem(root) is None
    record = door.seam_pin_record(root)
    assert record["matches_measured_row"] is False
    assert record["declared_engine_version"] == "2.5.7"
    assert record["pinned_engine_version"] == engine_pin.wanted_version()
    assert record["moved_from_pin"] == ["woof/core/microphysics.py"]
    assert record["absent_from_pin"] == []

    identity = engine_pin.identity_of(engine_pin.inspect_seam(root))
    assert not identity.matches_measured_row and not identity.admitted
    assert identity.version == "2.5.7"
    assert identity.build_commit == engine_pin.UNMEASURED_BUILD_COMMIT
    assert identity.manifest["woof/core/microphysics.py"] != manifest[
        "woof/core/microphysics.py"
    ], "an unmatched tree is named by the bytes it was FOUND with"


def test_a_matching_engine_is_recorded_as_the_measured_row(tmp_path, monkeypatch):
    root, manifest = _seam_tree(tmp_path, "2.5.7")
    monkeypatch.setattr(engine_pin, "seam_manifest", lambda: manifest)

    assert door.seam_pin_problem(root) is None
    record = door.seam_pin_record(root)
    assert record["matches_measured_row"] is True
    assert record["measured_row"] == engine_pin.wanted_version()
    assert record["moved_from_pin"] == [] and record["absent_from_pin"] == []


def test_the_door_refuses_a_tree_with_no_seam_module_by_its_breakage(
    tmp_path, monkeypatch
):
    """What still refuses is what cannot run: the tree does not carry the
    module the seam constructs.  The refusal is the door's own type, so
    ``run_forecast`` relays it verbatim."""

    root, manifest = _seam_tree(tmp_path, "2.5.7")
    monkeypatch.setattr(engine_pin, "seam_manifest", lambda: manifest)
    (root / "woof/core/mpas_column_batch.py").unlink()

    problem = door.seam_pin_problem(root)
    assert problem is not None
    assert "woof/core/mpas_column_batch.py" in problem
    assert "no physics for the run to call" in problem
    assert engine_pin.gpuwm_requirement() in problem, "the remedy is not printed"
    assert "proved" not in problem and "pinned to" not in problem
    assert isinstance(door._refuse(problem), door.ForecastDoorRefusal)


# ---------------------------------------------------------------------------
# doctor sees it at install time
# ---------------------------------------------------------------------------
def _doctor_seam_finding(findings):
    return next(f for f in findings if f.subject.startswith("woof seam bytes"))


def test_doctor_passes_when_the_installed_engine_matches(tmp_path, monkeypatch):
    root, manifest = _seam_tree(tmp_path, "2.5.6")
    monkeypatch.setattr(engine_pin, "seam_manifest", lambda: manifest)
    monkeypatch.setattr(engine_pin, "installed_root", lambda: root)
    monkeypatch.setattr(
        doctor, "_distribution_version", lambda name: "2.5.6"
    )

    findings = doctor.check_physics_seam()
    seam = _doctor_seam_finding(findings)
    assert seam.status == doctor.VERIFIED
    assert doctor.blocking_gaps(findings) == []


def test_doctor_names_a_moved_engine_and_does_not_block_on_it(
    tmp_path, monkeypatch
):
    """The drift is still SEEN at install time, and it no longer blocks.

    This finding was blocking while the forecast refused a moved engine: a
    report that said "every check passed" over a dead forecast lane was the
    measured defect (2026-08-27 walk).  The forecast stopped refusing on
    2026-10-10 (``engine_pin.identity_of``), so a blocking finding would now
    be the false line: it would say the forecast cannot run when it does.
    """

    root, manifest = _seam_tree(tmp_path, "2.5.7")
    monkeypatch.setattr(engine_pin, "seam_manifest", lambda: manifest)
    monkeypatch.setattr(engine_pin, "installed_root", lambda: root)
    monkeypatch.setattr(
        doctor, "_distribution_version", lambda name: "2.5.7"
    )
    (root / "woof/core/microphysics.py").write_bytes(b"moved\n")

    findings = doctor.check_physics_seam()
    seam = _doctor_seam_finding(findings)
    assert seam.status == doctor.INFO and not seam.required
    assert "2.5.7" in seam.detail and engine_pin.wanted_version() in seam.detail
    assert "woof/core/microphysics.py" in seam.detail
    assert "refuses" not in seam.detail
    assert doctor.blocking_gaps(findings) == []


def test_doctor_says_so_rather_than_crashing_when_gpuwm_cannot_be_located(
    monkeypatch,
):
    monkeypatch.setattr(engine_pin, "installed_root", lambda: None)
    finding = doctor._check_seam_bytes("2.5.7")
    assert finding.status == doctor.INFO
    assert doctor.blocking_gaps([finding]) == []


# ---------------------------------------------------------------------------
# the checkout finding must not contradict the line above it
# ---------------------------------------------------------------------------
def _doctor_checkout_finding(findings):
    return next(f for f in findings if f.subject.startswith("woof seam source"))


def test_the_checkout_finding_stops_claiming_the_pin_is_unsatisfiable(
    tmp_path, monkeypatch
):
    """THE DEFECT THIS PREVENTS, measured 2026-08-28 on a real 2.5.8 install.

    ``woof hex doctor`` printed these two lines one under the other::

        OK      woof seam bytes: 16 of 16 pinned files are in this install
                and all 16 match
        INFO    woof source checkout: ... An installed woof satisfies pip
                and does not satisfy the pin, so that lane needs a source
                checkout at the pinned commit.

    The second was a CONSTANT string, written when the manifest pinned a
    document no wheel carried, and it outlived that.  It is now derived from
    the same inspection the first line prints, so an install that carries
    every pinned path can never be told it does not.
    """

    root, manifest = _seam_tree(tmp_path, "2.5.8")
    monkeypatch.setattr(engine_pin, "seam_manifest", lambda: manifest)
    monkeypatch.setattr(engine_pin, "installed_root", lambda: root)
    monkeypatch.setattr(doctor, "_distribution_version", lambda name: "2.5.8")

    findings = doctor.check_physics_seam()
    seam = _doctor_seam_finding(findings)
    checkout = _doctor_checkout_finding(findings)

    assert seam.status == doctor.VERIFIED
    assert checkout.status == doctor.INFO
    assert checkout.evidence["absent"] == []
    assert "does not satisfy the pin" not in checkout.detail
    assert "no wheel places" not in checkout.detail
    assert f"all {len(manifest)} pinned paths resolve" in checkout.detail, (
        "when every pinned path is in the install, the finding under the OK "
        "line has to say so; the two lines are read together"
    )
    assert "no --gpuwm-checkout" in checkout.detail, (
        "an install that carries every pinned path is the forecast's seam "
        "source; the line under the OK line must not send the user to clone"
    )
    assert "RECORD digest" in checkout.detail


def test_the_checkout_finding_still_names_a_path_an_older_engine_lacks(
    tmp_path, monkeypatch
):
    """The other direction, so the instrument is not a constant either way.

    Engines through 2.5.7 ship no ``docs/mpas-seam.md`` in the wheel.  There
    the finding must go back to naming the absent path, because there an
    install genuinely cannot satisfy the pin on its own.
    """

    root, manifest = _seam_tree(tmp_path, "2.5.7")
    monkeypatch.setattr(engine_pin, "seam_manifest", lambda: manifest)
    monkeypatch.setattr(engine_pin, "installed_root", lambda: root)
    monkeypatch.setattr(doctor, "_distribution_version", lambda name: "2.5.7")
    (root / "docs/mpas-seam.md").unlink()

    checkout = _doctor_checkout_finding(doctor.check_physics_seam())
    assert checkout.evidence["absent"] == ["docs/mpas-seam.md"]
    assert "docs/mpas-seam.md" in checkout.detail
    assert "cannot satisfy the seam pin on its own" in checkout.detail


def test_the_remedy_stops_giving_a_reason_that_is_false_at_the_pinned_engine():
    """``engine_pin.remedy()`` is printed by doctor and by the door.

    It told every reader that one of the sixteen pinned paths is a document
    no wheel places in site-packages.  At the only engine this port admits
    that is false, and a remedy carrying a dead reason is a refusal that
    does not name its breakage.
    """

    text = engine_pin.remedy()
    assert "no wheel" not in text
    assert text.splitlines()[0].strip().startswith("pip install")
    assert "optional" in text and "--gpuwm-checkout" in text


# ---------------------------------------------------------------------------
# admitted engines: a measured row for a build that is not on PyPI
# ---------------------------------------------------------------------------
#: The document the admitted table is rendered from; one file for every
#: admitted engine, keyed by version, written by the instrument's --admit.
ADMITTED_OF_RECORD = ROOT / "verification" / "engine-verdicts" / "admitted-engines.json"


def _instrument():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_measure_engine_verdicts", ROOT / "tools" / "measure_engine_verdicts.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_admitted_table_is_the_instruments_output_not_a_transcription():
    """The second generated block is held to the same standard as the first.

    A row here admits an engine the published table cannot see, so a
    hand-typed digest would admit whatever bytes the typist believed in.
    The block must be the instrument's own rendering of the document it
    wrote, field for field, and every shipped row must equal its JSON row.
    """

    import json

    assert ADMITTED_OF_RECORD.is_file()
    document = json.loads(ADMITTED_OF_RECORD.read_text(encoding="utf-8"))
    instrument = _instrument()
    assert document["schema"] == instrument.ADMITTED_SCHEMA
    assert document["instrument"] == "tools/measure_engine_verdicts.py"

    source = (ROOT / ".." / ".." / "woof" / "hex" / "engine_pin.py").read_text(encoding="utf-8")
    start = source.index(instrument.ADMITTED_BEGIN_MARK)
    end = source.index(instrument.ADMITTED_END_MARK) + len(instrument.ADMITTED_END_MARK)
    assert source[start:end] == instrument.render_admitted_rows(document["engines"]), (
        "the ADMITTED_ENGINES block is not the instrument's own rendering of "
        "admitted-engines.json; re-splice it with --splice-admitted"
    )

    shipped = {row.version: row for row in engine_pin.ADMITTED_ENGINES}
    assert sorted(shipped) == sorted(row["version"] for row in document["engines"])
    for measured in document["engines"]:
        row = shipped[measured["version"]]
        assert row.base == measured["base"]
        assert row.build_commit == measured["build_commit"]
        assert row.manifest == measured["digests"]
        assert row.glacier_composed_tu_sha256 == measured["glacier_composed_tu_sha256"]
        assert list(row.moved_from_base) == measured["moved_from_base"]
        assert [list(pair) for pair in row.artifacts] == [
            [artifact["name"], artifact["sha256"]] for artifact in measured["artifacts"]
        ]
        # The row was measured against the manifest this package ships:
        # what it says moved is what differs from the shipped digests now.
        manifest = engine_pin.seam_manifest()
        assert set(row.manifest) == set(manifest)
        assert row.moved_from_base == tuple(
            sorted(path for path in manifest if row.manifest[path] != manifest[path])
        )


def _row_rules_hold(row) -> None:
    """What a row may be.  A local version on a base inside the derived
    range, because the specifier is what pip enforces and the row only
    decides bytes; distinct from the pin (a row that moved nothing would be
    the pin under another name); and absent from the published table,
    which measures PyPI and has no business admitting it."""

    floor, ceiling = engine_pin.gpuwm_floor(), engine_pin.gpuwm_ceiling()
    assert "+" in row.version and row.version.split("+", 1)[0] == row.base
    assert engine_pin._key(floor) <= engine_pin._key(row.base) < engine_pin._key(ceiling)
    assert row.moved_from_base
    assert engine_pin.engine(row.version) is None
    assert engine_pin.admitted_engine(row.version) is row
    identity = engine_pin.identity_for(row.version)
    assert identity.admitted and identity.build_commit == row.build_commit
    assert identity.manifest == row.manifest


def test_the_published_distribution_ships_no_admitted_row():
    """A row names a build that is not public, so the public table is empty
    and the public pin is the only engine a published install runs on.
    Whoever builds a private engine writes its row into their own tree with
    ``--admit``; every rule a row must satisfy is exercised below on a
    synthetic one."""

    assert engine_pin.ADMITTED_ENGINES == ()
    assert engine_pin.pinned_engine().version == engine_pin.wanted_version()
    assert not engine_pin.pinned_engine().admitted
    for row in engine_pin.ADMITTED_ENGINES:  # a private tree's rows, if any
        _row_rules_hold(row)


def test_an_admitted_row_rides_the_specifier_and_is_not_a_published_engine(
    tmp_path, monkeypatch
):
    _, _, row = _admitted_tree(tmp_path, monkeypatch)
    _row_rules_hold(row)
    assert engine_pin.admitted_engine(row.version + ".2") is None


def _admitted_tree(tmp_path: Path, monkeypatch, version: str | None = None):
    """A fake public checkout, a fake private build of it whose bytes differ
    on one path, and the admitted row measured FROM that build's bytes.

    Real files and real digests again: the row's manifest is computed from
    what was written, so acceptance below is a byte match and every refusal
    is a byte or a version string that actually differs.
    """

    base = engine_pin.gpuwm_floor()
    version = version or f"{base}+local.1"
    root, manifest = _seam_tree(tmp_path, engine_pin.wanted_version())
    private = tmp_path / "gpuwm-private"
    for relative in manifest:
        target = private / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / relative).read_bytes())
    (private / "woof/core/microphysics.py").write_bytes(b"def apply(): ...  # private\n")
    (private / "pyproject.toml").write_text(
        f'[project]\nname = "woof"\nversion = "{version}"\n', encoding="utf-8"
    )
    digests = {
        relative: sha256((private / relative).read_bytes()).hexdigest()
        for relative in manifest
    }
    row = engine_pin.AdmittedEngine(
        version=version,
        base=base,
        build_commit="9" * 40,
        digests=tuple(sorted(digests.items())),
        glacier_composed_tu_sha256="7" * 64,
        moved_from_base=("woof/core/microphysics.py",),
        artifacts=(("gpuwm-fake.whl", "0" * 64),),
    )
    monkeypatch.setattr(engine_pin, "seam_manifest", lambda: manifest)
    monkeypatch.setattr(engine_pin, "ADMITTED_ENGINES", (row,))
    return root, private, row


def test_an_admitted_engines_bytes_under_its_own_version_are_accepted(
    tmp_path, monkeypatch
):
    root, private, row = _admitted_tree(tmp_path, monkeypatch)

    inspection = engine_pin.inspect_seam(private)
    assert inspection.engine == row.version
    assert inspection.declared == row.version
    assert inspection.admitted and inspection.accepted
    assert not inspection.drifted and inspection.moved == () and inspection.absent == ()
    assert inspection.checked == len(row.manifest)
    assert door.seam_pin_problem(private) is None

    # The public tree beside it is untouched by the row's existence.
    public = engine_pin.inspect_seam(root)
    assert public.engine == engine_pin.wanted_version()
    assert not public.admitted and public.accepted
    assert door.seam_pin_problem(root) is None


def test_the_same_bytes_under_another_version_string_are_not_named_as_the_row(
    tmp_path, monkeypatch
):
    """A re-versioned copy of a measured build is neither that build nor the
    pin.  The refusal names the version the bytes belong to and the version
    the tree claims, and the remedy is not a pip line."""

    _, private, row = _admitted_tree(tmp_path, monkeypatch)
    (private / "pyproject.toml").write_text(
        '[project]\nname = "woof"\nversion = "2.8.0"\n', encoding="utf-8"
    )

    inspection = engine_pin.inspect_seam(private)
    assert inspection.engine == row.version and inspection.declared == "2.8.0"
    assert inspection.misdeclared and not inspection.accepted
    assert inspection.moved == (), "the bytes ARE the row's; only the name is wrong"

    # Retired 2026-10-10 as a refusal.  The receipt names the tree by what
    # it declares and by its bytes, and does NOT name it as the row: a row
    # is keyed by its version string and its bytes together.
    assert door.seam_pin_problem(private) is None
    record = door.seam_pin_record(private)
    assert record["matches_measured_row"] is False and record["measured_row"] is None
    assert record["declared_engine_version"] == "2.8.0"
    identity = engine_pin.identity_of(inspection)
    assert identity.version == "2.8.0" and not identity.admitted


def test_an_admitted_version_over_other_bytes_is_recorded_against_the_pin(
    tmp_path, monkeypatch
):
    """The other half of the key.  A tree that calls itself the admitted
    version and carries bytes the row did not measure is an unmeasured
    engine, and is refused the way every unmeasured engine is: against the
    public pin, naming what moved."""

    _, private, row = _admitted_tree(tmp_path, monkeypatch)
    (private / "woof/io/restart.py").write_bytes(b"RESTART = 3  # nobody measured this\n")

    inspection = engine_pin.inspect_seam(private)
    assert inspection.engine is None and not inspection.admitted
    assert inspection.declared == row.version
    assert inspection.moved == ("woof/core/microphysics.py", "woof/io/restart.py"), (
        "measured against the PUBLIC pin, so the row's own private byte "
        "counts as moved too: the row admits one engine, not a family"
    )
    # Retired 2026-10-10 as a refusal: the tree runs and the record names
    # what moved.
    assert door.seam_pin_problem(private) is None
    record = door.seam_pin_record(private)
    assert record["matches_measured_row"] is False
    assert record["declared_engine_version"] == row.version
    assert "woof/io/restart.py" in record["moved_from_pin"]


def test_doctor_names_the_admitted_engine_it_verified(tmp_path, monkeypatch):
    _, private, row = _admitted_tree(tmp_path, monkeypatch)
    monkeypatch.setattr(engine_pin, "installed_root", lambda: private)
    monkeypatch.setattr(doctor, "_distribution_version", lambda name: row.version)

    findings = doctor.check_physics_seam()
    seam = _doctor_seam_finding(findings)
    assert seam.status == doctor.VERIFIED
    assert row.version in seam.detail and "admitted engine" in seam.detail
    assert seam.evidence["engine"] == row.version and seam.evidence["admitted"]
    assert doctor.blocking_gaps(findings) == []


def test_doctor_names_an_install_whose_metadata_disowns_its_bytes(
    tmp_path, monkeypatch
):
    """An INSTALL root carries no pyproject.toml; what it declares is pip's
    metadata.  Bytes of an admitted row under metadata that says the public
    version is the misdeclared case seen from site-packages."""

    _, private, row = _admitted_tree(tmp_path, monkeypatch)
    (private / "pyproject.toml").unlink()
    monkeypatch.setattr(engine_pin, "installed_root", lambda: private)
    monkeypatch.setattr(engine_pin, "installed_version", lambda: "2.8.0")
    monkeypatch.setattr(doctor, "_distribution_version", lambda name: "2.8.0")

    assert engine_pin.declared_version(private) == "2.8.0"
    findings = doctor.check_physics_seam()
    seam = _doctor_seam_finding(findings)
    assert seam.status == doctor.INFO and not seam.required
    assert row.version in seam.detail and "2.8.0" in seam.detail
    assert doctor.blocking_gaps(findings) == []


def test_the_adapter_admits_the_row_and_provenances_the_run_by_it(
    tmp_path, monkeypatch
):
    """The wall itself.  The adapter's checkout guard must accept the
    admitted tree, hand back THAT row's identity for the receipts, and
    refuse the two impostors with the spelling every earlier refusal used."""

    from woof.hex import cuda_arwen_physics_v841 as adapter

    root, private, row = _admitted_tree(tmp_path, monkeypatch)
    surface = sha256()
    for relative in ("woof/core/mpas_column_batch.py", "docs/mpas-seam.md"):
        surface.update((private / relative).read_bytes())
    monkeypatch.setattr(adapter, "MPAS_SEAM_CONTRACT_SURFACE_SHA256", surface.hexdigest())

    identity = adapter._verify_checkout_root(private)
    assert identity.admitted and identity.version == row.version
    assert identity.build_commit == row.build_commit
    assert identity.manifest == row.manifest
    assert identity.glacier_composed_tu_sha256 == row.glacier_composed_tu_sha256

    (private / "pyproject.toml").write_text(
        '[project]\nname = "woof"\nversion = "2.8.0"\n', encoding="utf-8"
    )
    # Retired 2026-10-10 as refusals: a re-versioned copy and a moved byte
    # are both recorded as the tree that was found, not as the row.
    misdeclared = adapter._verify_checkout_root(private)
    assert not misdeclared.matches_measured_row and not misdeclared.admitted
    assert misdeclared.version == "2.8.0"
    assert misdeclared.build_commit == engine_pin.UNMEASURED_BUILD_COMMIT

    (private / "pyproject.toml").write_text(
        f'[project]\nname = "woof"\nversion = "{row.version}"\n', encoding="utf-8"
    )
    (private / "woof/io/restart.py").write_bytes(b"RESTART = 3\n")
    moved = adapter._verify_checkout_root(private)
    assert not moved.matches_measured_row and moved.version == row.version
    assert "woof/io/restart.py" in moved.moved_from_pin
    assert moved.manifest["woof/io/restart.py"] == sha256(b"RESTART = 3\n").hexdigest()

    # What cannot run is still refused, by its breakage.
    kept = (private / "woof/core/mpas_column_batch.py").read_bytes()
    (private / "woof/core/mpas_column_batch.py").unlink()
    with pytest.raises(FileNotFoundError, match="no physics to run in this tree"):
        adapter._verify_checkout_root(private)
    (private / "woof/core/mpas_column_batch.py").write_bytes(kept)

    # And the public tree still resolves to the public pin's identity, with
    # the surface digest the public tree actually has.
    surface = sha256()
    for relative in ("woof/core/mpas_column_batch.py", "docs/mpas-seam.md"):
        surface.update((root / relative).read_bytes())
    monkeypatch.setattr(adapter, "MPAS_SEAM_CONTRACT_SURFACE_SHA256", surface.hexdigest())
    public = adapter._verify_checkout_root(root)
    assert not public.admitted and public.version == engine_pin.wanted_version()
    assert public.build_commit == adapter.ARWEN_BUILD_COMMIT
