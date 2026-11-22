"""Refusals that used to be tracebacks, false statements, or absent.

Four rows of the 2.3.2 reachability audit, each a different way for a
shipped feature to be unreachable, and each guarded here against the
exact defect that was measured.

**A refusal must never state something false.**  The DA nowcast one-command
path downloaded two NEXRAD Level-II volumes and then died with
``woof domain: --polygon local GeoJSON file does not exist:
danow_full\\case\\domain-box.geojson`` -- for a file that was on disk, 243
bytes, written by the survey stage one step earlier.  The cause was a
relative ``--out``: every stage runs as a subprocess with
``cwd=repo_root``, so the parent wrote the polygon under the CALLER's
working directory and ``woof domain`` looked for the same relative path
under ``site-packages``.  Two things are guarded: the path is absolute
before the survey spends a byte, and the message names the path the
process actually looked at plus the directory it resolved against, so a
`cd`-shaped bug identifies itself instead of sending the reader hunting.

**A refusal must arrive before the expensive work.**  The
``--domain-polygon`` existence check ran after the download, so "you named
a missing file" cost the user two radar volumes.

**A case that needs data nobody ships must say so.**  ``woof verify
real74_d01`` raised a twenty-line ``FileNotFoundError`` at exit 1, and
raised it INSTEAD of the clean CuPy refusal every other verify case
gives, because the missing namelist is reached before the first ``import
cupy``.

**A remedy must be reachable and must be true.**  The LES tornado case
modules ran a config audit before argparse, so ``--help`` itself failed;
the config they wanted is under ``configs/``, which is not a package and
is in no wheel; and the remedy they printed -- "pass the experiment .toml
that `woof domain` wrote" -- named a file the wizard does not emit and a
flag the module did not have.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_module(module: str, *args: str, cwd: Path,
                env: dict | None = None) -> subprocess.CompletedProcess:
    """Run ``python -m module`` and return the completed process.

    The return code is read off the process object directly.  No pipe,
    no ``grep -c``, no ``$?`` after a pipe -- each of which has produced
    a false green in this repository before.
    """

    environment = {**os.environ, "PYTHONPATH": str(REPO_ROOT),
                   "PYTHONSAFEPATH": "1"}
    environment.update(env or {})
    return subprocess.run([sys.executable, "-m", module, *args],
                          cwd=cwd, capture_output=True, text=True,
                          errors="replace", env=environment)


# ==========================================================================
# The polygon refusal states something true.
# ==========================================================================

def test_the_polygon_refusal_names_an_absolute_path(tmp_path):
    """The message must be checkable, and checking it must confirm it."""

    from woof.domain_wizard import load_polygon_footprint

    missing = tmp_path / "case" / "domain-box.geojson"
    with pytest.raises(ValueError) as caught:
        load_polygon_footprint(missing)
    message = str(caught.value)
    assert str(missing.resolve()) in message, message
    assert "does not exist" in message


def test_the_polygon_refusal_resolves_a_relative_path_and_says_against_what(
        tmp_path, monkeypatch):
    """The exact defect: a relative path read from a different cwd.

    The old message echoed the argument as typed, so a reader was told
    ``case\\domain-box.geojson`` does not exist while a file of that name
    sat in the directory they were standing in.
    """

    from woof.domain_wizard import load_polygon_footprint

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    with pytest.raises(ValueError) as caught:
        load_polygon_footprint("case/domain-box.geojson")
    message = str(caught.value)
    # Absolute, and rooted in the directory the check really used.
    assert str(elsewhere.resolve()) in message, message
    assert "relative to the working directory" in message


def test_the_polygon_refusal_does_not_call_a_present_file_missing(tmp_path,
                                                                  monkeypatch):
    """The negative control: a file that IS there must not be refused."""

    from woof.domain_wizard import load_polygon_footprint

    case = tmp_path / "case"
    case.mkdir()
    polygon = case / "domain-box.geojson"
    polygon.write_text(
        '{"type":"Polygon","coordinates":'
        '[[[-98.0,35.0],[-97.0,35.0],[-97.0,36.0],[-98.0,36.0],'
        '[-98.0,35.0]]]}', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    footprint = load_polygon_footprint("case/domain-box.geojson")
    assert footprint is not None


def test_a_directory_is_refused_as_a_directory_not_as_absent(tmp_path):
    """`is_file()` is false for a directory too; the message must not
    claim the path does not exist when it plainly does."""

    from woof.domain_wizard import load_polygon_footprint

    directory = tmp_path / "domain-box.geojson"
    directory.mkdir()
    with pytest.raises(ValueError) as caught:
        load_polygon_footprint(directory)
    assert "is not a regular file" in str(caught.value)


# ==========================================================================
# real74_d01 refuses by name instead of raising FileNotFoundError.
# ==========================================================================

def test_real74_refuses_a_missing_bundle_by_name(tmp_path, monkeypatch):
    from woof.verify.cases import real74_d01

    monkeypatch.setattr(real74_d01, "BUNDLE", tmp_path / "not-here")
    with pytest.raises(ValueError) as caught:
        real74_d01.require_bundle()
    message = str(caught.value)
    assert "WOOF_REAL74_REFERENCE_BUNDLE" in message
    assert "reference bundle" in message
    # A ValueError is what puts it on the CLI's refusal boundary; a
    # FileNotFoundError is what used to escape as a traceback.
    assert not isinstance(caught.value, FileNotFoundError)
    # The offered fallback is the real default location, not the
    # override's basename pasted under ~/Downloads.
    assert str(real74_d01.DEFAULT_BUNDLE) in message
    assert "Downloads\\not-here" not in message
    assert "Downloads/not-here" not in message


def test_the_real74_entry_points_actually_call_the_guard(tmp_path,
                                                         monkeypatch):
    """The guard must be CALLED, not merely defined.

    This is the essential test of the row.  The first version of the
    fix added ``require_bundle()`` and wired it to nothing: every unit
    test of the function passed, and `woof verify real74_d01` on the
    installed wheel still died with the same twenty-line
    ``FileNotFoundError``.  A validator with no caller is the shape this
    program has been burned by repeatedly, so both doors are pinned
    here -- the verify door and the config-driven door.
    """

    from woof.verify.cases import real74_d01

    monkeypatch.setattr(real74_d01, "BUNDLE", tmp_path / "not-here")

    with pytest.raises(ValueError) as caught:
        real74_d01.run(outdir=str(tmp_path / "out"))
    assert "WOOF_REAL74_REFERENCE_BUNDLE" in str(caught.value)

    with pytest.raises(ValueError) as caught:
        real74_d01._case_grid(object())
    assert "WOOF_REAL74_REFERENCE_BUNDLE" in str(caught.value)


def test_real74_accepts_a_bundle_that_is_there(tmp_path, monkeypatch):
    """The negative control."""

    from woof.verify.cases import real74_d01

    bundle = tmp_path / "bundle"
    (bundle / "namelists").mkdir(parents=True)
    (bundle / "namelists" / "namelist.wps").write_text("&share\n/\n",
                                                       encoding="utf-8")
    monkeypatch.setattr(real74_d01, "BUNDLE", bundle)
    assert real74_d01.require_bundle() == bundle


def test_real74_distinguishes_a_present_directory_from_an_absent_one(
        tmp_path, monkeypatch):
    from woof.verify.cases import real74_d01

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    monkeypatch.setattr(real74_d01, "BUNDLE", bundle)
    with pytest.raises(ValueError) as caught:
        real74_d01.require_bundle()
    assert "exists but has no" in str(caught.value)


def test_the_bundle_default_and_env_var_stay_in_the_case_module():
    """The standing rule, for the two names this row added.

    Whether the token leaks into a generic POSITION is already gated by
    ``tools/check_case_token_leakage.py`` and pinned by
    ``tests/test_case_token_leakage.py``; this does not duplicate that
    scan.  What it pins is narrower and is this row's own contract: the
    bundle default and the environment variable that overrides it are
    defined in the case module and referenced from nowhere else, so the
    refusal added here cannot become the reason a case name reaches a
    default somewhere generic.
    """

    import subprocess

    from _woof_snapshot import left_out

    out = subprocess.run(
        ["git", "grep", "-l", "WOOF_REAL74_REFERENCE_BUNDLE", "--",
         "woof/", "tools/"],
        cwd=REPO_ROOT, capture_output=True, text=True)
    hits = sorted(line for line in out.stdout.splitlines() if line.strip())
    case = "woof/verify/cases/real74_d01.py"
    # a tree that leaves the case module out carries the names nowhere at all
    assert hits == ([] if left_out(case) else [case]), hits
    # `tilestream/realdata.py` reads the same variable and predates this
    # row; it is the streamed transport's own resolver, not a default in
    # the model package, and it is deliberately NOT swept in above so
    # that this pin stays about what this row added.


# ==========================================================================
# The LES tornado modules answer --help and name a reachable remedy.
# ==========================================================================

LES_MODULES = (
    "woof.verify.cases.les_tornado_mayfield_20211210",
    "woof.verify.cases.les_tornado_dodgecity_20160524",
)


@pytest.mark.parametrize("module", LES_MODULES)
def test_les_help_works_without_the_config(module, tmp_path):
    """``--help`` must answer on a machine that has no ``configs/``.

    The audit measured this failing on an installed wheel, where the
    config the module audited resolved to ``site-packages/configs/...``
    -- a path that has never existed anywhere.
    """

    proc = _run_module(module, "--help", cwd=tmp_path,
                       env={"WOOF_CONFIGS_ROOT": str(tmp_path / "empty")})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "--config" in proc.stdout


@pytest.mark.parametrize("name", (
    "les_tornado_100m_mayfield_20211210.toml",
    "les_tornado_100m_dodgecity_20160524.toml",
))
def test_les_refusal_is_true_and_names_a_flag_the_module_has(name):
    """The old remedy named a file the wizard does not emit, through an
    argument the module did not accept.

    Built in-process rather than by running the module: from a source
    checkout the config IS on disk, so a subprocess would never reach
    the refusal and the test would skip -- and a test that skips on the
    machine that runs it is not evidence of anything.
    """

    from woof.verify.cases import _repo_config

    message = _repo_config.missing_config_message(name)
    assert name in message
    assert "--config" in message
    assert "WOOF_CONFIGS_ROOT" in message
    assert "Searched:" in message
    # The retired remedy, which was wrong twice: the wizard does not
    # emit these configs, and the module took no path argument.
    assert "pass the experiment .toml that" not in message
    assert "does NOT emit it" in message


@pytest.mark.parametrize("module", LES_MODULES)
def test_les_refuses_a_named_config_that_is_not_there(module, tmp_path):
    """``--config`` pointing at nothing is a sentence and exit 2."""

    proc = _run_module(module, "--config", str(tmp_path / "nope.toml"),
                       cwd=tmp_path)
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 2, combined
    assert "--config is not a readable file" in combined
    assert "nope.toml" in combined
    assert "Traceback" not in combined, combined


@pytest.mark.parametrize("module", LES_MODULES)
def test_les_accepts_an_explicit_config(module, tmp_path):
    """The negative control: the flag the remedy names must work."""

    from woof.verify.cases import _repo_config

    name = {"woof.verify.cases.les_tornado_mayfield_20211210":
            "les_tornado_100m_mayfield_20211210.toml",
            "woof.verify.cases.les_tornado_dodgecity_20160524":
            "les_tornado_100m_dodgecity_20160524.toml"}[module]
    config = _repo_config.locate(name)
    if config is None:
        pytest.skip("this tree has no checkout configs/ to point at")
    proc = _run_module(module, "--config", str(config), cwd=tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert str(config) in proc.stdout


def test_the_configs_directory_really_is_in_the_wheel():
    """The premise the LES refusal asserts, checked rather than assumed.

    This pin used to run the other way, and it did its job: it asserted
    ``configs/`` was in no distribution, so that whoever packaged it would
    be sent here to bring the refusal's explanation with them. 2.5.0
    packages it -- ``configs`` is a declared top-level package landing at
    ``<site-packages>/configs``, which is exactly the path
    ``_repo_config`` resolves -- so the pin is inverted rather than
    deleted. Dropping it would let the directory silently fall back out of
    the wheel while the refusal kept claiming an install has it, and
    ``docs/public/VERIFICATION.md`` and ``docs/public/LES.md`` both send
    readers to files under it.

    The wheel-contents measurement lives in
    ``tests/test_configs_are_packaged.py``; what is checked here is only
    the premise this file's refusal text depends on.
    """

    import tomllib

    with (REPO_ROOT / "pyproject.toml").open("rb") as stream:
        pyproject = tomllib.load(stream)
    setuptools = pyproject.get("tool", {}).get("setuptools", {})
    packages = setuptools.get("packages", {})
    include = packages.get("find", {}).get("include", [])
    package_data = setuptools.get("package-data", {})
    assert any(entry.startswith("configs") for entry in include), (
        "configs/ is in no distribution again; the LES refusal now tells a "
        "reader their install is incomplete when it is the packaging that "
        "regressed")
    assert "configs" in package_data, (
        "configs/ is a declared package with no package-data pattern, so "
        "the wheel carries the directory and none of its files")

    # ...and the refusal must not claim a pip install has never had the
    # file, which is what it said while the pin ran the other way.
    from woof.verify.cases import _repo_config

    message = _repo_config.missing_config_message(
        "les_tornado_100m_mayfield_20211210.toml")
    assert "has never had this file" not in message


# ==========================================================================
# `woof cases` explains the difference it advertises.
# ==========================================================================

def test_every_case_row_names_the_command_that_runs_it():
    from woof.cli import case_door
    from woof.verify import cases

    records = cases.manifest()
    assert len(records) > 15, records
    verify_doors = 0
    for record in records:
        door = case_door(record)
        assert record["name"] in door
        if "verify" in record["capabilities"]:
            assert door == f"woof verify {record['name']}"
            verify_doors += 1
        else:
            assert door.startswith("python -m woof.verify.cases.")
    assert verify_doors, "no case carries the verify capability"
    assert verify_doors < len(records), (
        "every case is verify-capable, so this row's premise -- that the "
        "listing advertises more cases than `woof verify` accepts -- no "
        "longer holds and the test needs rewriting")


def test_the_case_door_matches_what_verify_actually_accepts():
    """The two halves, held against each other.

    `woof cases` and `woof verify`'s choice list are built from the
    same registry; this asserts they cannot drift, which is what made
    "21 advertised, 10 accepted" invisible.
    """

    from woof.cli import build_parser, case_door
    from woof.verify import cases

    parser = build_parser()
    accepted: set[str] = set()
    for action in parser._actions:
        if getattr(action, "dest", None) == "command":
            verify = action.choices["verify"]
            for sub in verify._actions:
                if sub.dest == "case" and sub.choices:
                    accepted = set(sub.choices)
    assert accepted, "could not read `woof verify`'s choice list"
    for record in cases.manifest():
        door = case_door(record)
        if door.startswith("woof verify "):
            assert record["name"] in accepted, (
                f"`woof cases` offers `{door}` but `woof verify` "
                f"refuses {record['name']}")
        else:
            assert record["name"] not in accepted, (
                f"`woof verify` accepts {record['name']} but the "
                f"listing sends the reader to `{door}`")
