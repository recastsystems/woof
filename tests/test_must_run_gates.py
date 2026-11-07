"""A gate whose fixtures ship must RUN, and one that cannot must say so.

THE BREAKAGE THIS PREVENTS
--------------------------
``tools/battery/no_silent_deselection.py`` and ``tools/battery/
list_census.json`` count COLLECTED items -- the plugin says so at
``pytest_collection_finish``, and the census records nothing but a per-file
collected count.  A ``skipif``-marked item is collected.  So a deck whose
every test skips contributes its full count, clears its floor, and the leg is
green: the instrument written to detect "coverage that left" is blind to the
way coverage most often leaves.

The worked example is in this tree.  ``tools/rrtmg_wrf461_oracle/lw_gate.py
:23-26`` resolves the RRTMG longwave fixture root to

    os.path.expanduser("~/.woof/oracle/rrtmg_lw/fixtures")

-- an oracle scratch directory that no clone carries -- and
``tests/test_rrtmg_lw_numpy.py:38`` is a whole-module ``skipif`` on that
directory existing.  It exists in no clone, so 100% of the longwave gates
skip everywhere, inside a module whose own line 92 reads *"this gate must
never silently skip again"*.  Nothing in the repository went red, because
nothing in the repository was asking whether a gate RAN.

WHAT THIS FILE IS
-----------------
The static half of ``tools/battery/must_run_gates.txt``:

* every must-run entry is a real file, needs no GPU, and (if it declares an
  unbounded skip count) says why;
* every declared-skipping entry is a real file and carries a reason;
* **every module-level skip gate in the tree is in one half or the other** --
  so a new deck cannot start skipping silently, it has to be declared.

and the negative controls for the runtime half, ``tools/battery/
no_silent_skip.py``, run as real pytest sessions against synthetic gates: a
gate that skips everything must make the session RED, and the same gate
passing must leave it green.  A guard whose failure path is never exercised
is the shape this whole file exists to refuse.
"""

from __future__ import annotations

import ast
import pathlib
import subprocess
import sys

import pytest

from tools.battery.no_silent_skip import (
    declared_test_exceptions, parse_declared_skipping, parse_manifest,
)

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = REPOSITORY_ROOT / "tools" / "battery" / "must_run_gates.txt"
GUARD = REPOSITORY_ROOT / "tools" / "battery" / "no_silent_skip.py"

MANIFEST_TEXT = MANIFEST.read_text(encoding="utf-8") if MANIFEST.is_file() else ""
MUST_RUN = parse_manifest(MANIFEST_TEXT)
DECLARED = parse_declared_skipping(MANIFEST_TEXT)


def _test_modules() -> list[pathlib.Path]:
    return sorted(list((REPOSITORY_ROOT / "tests").glob("test_*.py"))
                  + list((REPOSITORY_ROOT / "tilestream").glob("test_*.py")))


def module_level_skip_gate(path: pathlib.Path) -> str | None:
    """The module-scope construct that can retire this whole file, or None.

    Two shapes retire a module without any test of its own saying so: a
    ``pytestmark`` carrying a skip, and ``pytest.skip(...,
    allow_module_level=True)`` at module scope.  Both are found by reading the
    module's own source, never by importing it -- importing a test module to
    ask whether it skips is how a detector ends up running the code it is
    judging.  Function and class bodies are not descended into: a skip inside
    a test retires that test, which is a different question and is the one
    the manifest's ceilings answer.
    """

    source = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(source)
    except SyntaxError:                              # pragma: no cover
        return None
    pending: list[ast.AST] = list(tree.body)
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            continue
        for field in ("body", "orelse", "finalbody"):
            pending.extend(getattr(node, field, None) or [])
        for handler in getattr(node, "handlers", None) or []:
            pending.extend(handler.body)
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "pytestmark"
                for target in node.targets):
            segment = ast.get_source_segment(source, node.value) or ""
            if "skip" in segment:
                return " ".join(segment.split())
        for inner in ast.walk(node):
            if isinstance(inner, ast.Call) and any(
                    keyword.arg == "allow_module_level"
                    for keyword in inner.keywords):
                return " ".join((ast.get_source_segment(source, inner)
                                 or "").split())
    return None


SKIP_GATES = sorted(
    path.relative_to(REPOSITORY_ROOT).as_posix()
    for path in _test_modules() if module_level_skip_gate(path))


def test_the_manifest_is_present_and_names_something() -> None:
    """An empty manifest passes every other test in this file."""

    assert MANIFEST.is_file(), (
        f"{MANIFEST} is missing.  It is the only record of which decks read "
        "fixtures that ship, and without it tools/battery/no_silent_skip.py "
        "loads no gates and every skip is silent again.")
    assert MUST_RUN, (
        "tools/battery/must_run_gates.txt parsed to zero must-run entries; "
        "the guard would then approve any session, which is the "
        "green-on-nothing shape it was written against.")
    assert b"\r" not in MANIFEST.read_bytes(), (
        "tools/battery/must_run_gates.txt contains CR bytes; the repository "
        "commits LF and .gitattributes does no conversion")


@pytest.mark.parametrize("entry", sorted(MUST_RUN), ids=lambda value: value)
def test_every_must_run_entry_is_a_file(entry: str) -> None:
    assert (REPOSITORY_ROOT / entry).is_file(), (
        f"{entry} is on tools/battery/must_run_gates.txt and is not in this "
        "tree.  A manifest naming a path that does not exist makes the leg "
        "that reads it exit before collecting anything -- which is exactly "
        "what tools/battery/stage1_files.txt:1222 does with the "
        "release-excluded tests/test_n5s_toolchain.py.")


@pytest.mark.parametrize("entry", sorted(MUST_RUN), ids=lambda value: value)
def test_no_must_run_entry_needs_a_gpu(entry: str) -> None:
    """The property that earns a place here, checked rather than asserted.

    A GPU-bound file on this list would skip on every hosted runner, so the
    guard would fail the CPU lane for an absence nobody can fix there.
    ``tests/conftest.py`` is the authority on what "GPU-bound" means in this
    tree, so this asks conftest rather than pattern-matching the filename --
    the same way tests/test_gpu_shard_manifest.py does, in the other
    direction.
    """

    from conftest import _cupy_scope

    whole, functions = _cupy_scope(str(REPOSITORY_ROOT / entry))
    assert not whole and not functions, (
        f"{entry} opens a CUDA device by conftest's own detector, so it "
        "cannot be required to run on a CPU lane.  Move it to the "
        "DECLARED-SKIPPING half with 'needs a CUDA device' as the reason.")


@pytest.mark.parametrize(
    "entry", sorted(name for name, (ceiling, _) in MUST_RUN.items()
                    if ceiling is None), ids=lambda value: value)
def test_every_unbounded_entry_says_why(entry: str) -> None:
    """``skips<=*`` switches off the strongest arm, so it costs a sentence."""

    _ceiling, comment = MUST_RUN[entry]
    assert comment, (
        f"{entry} declares skips<=* -- an unbounded skip count -- and gives "
        "no reason.  Unbounded means only the entirely-skipped arm still "
        "applies to it, so the entry has to say what makes the count a "
        "property of the fixture data rather than a constant.")


@pytest.mark.parametrize("entry", sorted(DECLARED), ids=lambda value: value)
def test_every_declared_skipping_entry_is_a_file_with_a_reason(
        entry: str) -> None:
    path, _, test = entry.partition("::")
    assert (REPOSITORY_ROOT / path).is_file(), (
        f"{entry} is declared as skipping and is not in this tree; delete "
        "the entry in the commit that deleted the file.")
    assert len(DECLARED[entry].split()) >= 4, (
        f"{entry} is declared as skipping with the reason "
        f"{DECLARED[entry]!r}.  The reason is the entire value of the "
        "declaration -- it is what turns a silent skip into a stated one -- "
        "so it has to name what is absent and what would restore it.")
    if test:
        name = test.split("[", 1)[0]
        tree = ast.parse((REPOSITORY_ROOT / path).read_text(encoding="utf-8"))
        defined = {node.name for node in tree.body
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        assert name in defined, (
            f"{entry} declares an exception for a test {path} does not define "
            "at module level; delete or correct the entry in the commit that "
            "renamed, split or fixed the test.")


def test_no_entry_is_in_both_halves() -> None:
    both = sorted(set(MUST_RUN) & set(DECLARED))
    assert not both, (
        f"{both} are both required to run and declared as unable to; the "
        "guard reads the first half and a reader reads the second, so the "
        "two would disagree about the same file.")


@pytest.mark.parametrize("entry", SKIP_GATES, ids=lambda value: value)
def test_every_module_level_skip_gate_is_declared(entry: str) -> None:
    """THE NOT-SILENCE RULE, and the reason this file exists.

    One line at the top of a module retires every test in it.  That line is
    legitimate -- a Rust artifact that is not built, a device that is not
    present -- but it must be a statement somebody wrote down, because the
    census cannot see it and the leg stays green either way.
    """

    assert entry in MUST_RUN or entry in DECLARED, (
        f"{entry} carries a module-level skip -- "
        f"{module_level_skip_gate(REPOSITORY_ROOT / entry)} -- so every test "
        "in it can vanish from a leg that still reports green, and "
        "tools/battery/list_census.json counts the skipped items toward its "
        "floor.  Put it on tools/battery/must_run_gates.txt if what it needs "
        "ships in this repository, or in that file's DECLARED-SKIPPING half "
        "with the reason it cannot run in a fresh clone.")


def test_the_detector_finds_a_module_level_skip_it_has_never_seen(
        tmp_path: pathlib.Path) -> None:
    """Non-vacuity, both directions, against synthetic modules.

    Without this, ``SKIP_GATES`` going empty -- a detector that stopped
    detecting -- would make the rule above pass on every file in the tree.
    """

    skipping = tmp_path / "test_skipping.py"
    skipping.write_text(
        "import os\n"
        "import pytest\n"
        "pytestmark = pytest.mark.skipif(not os.path.isdir('/nope'),\n"
        "                                reason='fixtures absent')\n"
        "def test_x():\n"
        "    assert True\n", encoding="utf-8")
    module_level = tmp_path / "test_module_level.py"
    module_level.write_text(
        "import pytest\n"
        "pytest.skip('no device', allow_module_level=True)\n"
        "def test_x():\n"
        "    assert True\n", encoding="utf-8")
    inner = tmp_path / "test_inner.py"
    inner.write_text(
        "import pytest\n"
        "def test_x():\n"
        "    pytest.skip('one test, not the module')\n", encoding="utf-8")

    assert module_level_skip_gate(skipping) is not None
    assert module_level_skip_gate(module_level) is not None
    assert module_level_skip_gate(inner) is None, (
        "a skip inside one test is not a module-level gate; the detector "
        "must not widen to it, or every file with a data-dependent skip "
        "would need a manifest entry")


def _synthetic_leg(workspace: pathlib.Path, manifest: str, body: str,
                   files: dict[str, str] | None = None,
                   args: list[str] | None = None) -> \
        subprocess.CompletedProcess:
    """One real pytest session, against the shipped guard, on a fake gate.

    The guard is loaded from ``tools/battery/no_silent_skip.py`` in THIS
    repository -- not copied -- so these are tests of the file that ships.
    ``pytest.ini`` pins the workspace as rootdir, because the guard resolves
    manifest entries against ``config.rootpath``.
    """

    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    battery = workspace / "tools" / "battery"
    battery.mkdir(parents=True, exist_ok=True)
    (battery / "must_run_gates.txt").write_text(manifest, encoding="utf-8")
    (workspace / "conftest.py").write_text(
        "import importlib.util\n"
        "import sys\n"
        "\n"
        "\n"
        "def pytest_configure(config):\n"
        f"    path = {str(GUARD)!r}\n"
        "    spec = importlib.util.spec_from_file_location('_guard', path)\n"
        "    module = importlib.util.module_from_spec(spec)\n"
        "    sys.modules['_guard'] = module\n"
        "    spec.loader.exec_module(module)\n"
        "    module.pytest_configure(config)\n", encoding="utf-8")
    tests = workspace / "tests"
    tests.mkdir(exist_ok=True)
    (tests / "test_gate.py").write_text(body, encoding="utf-8")
    for name, text in (files or {}).items():
        (workspace / name).write_text(text, encoding="utf-8")
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
        + (args if args is not None else ["tests/test_gate.py"]),
        cwd=str(workspace), capture_output=True, text=True)


@pytest.mark.parametrize("workers", [[], ["-n", "2"]],
                         ids=["in-process", "xdist"])
def test_the_guard_leaves_a_gate_a_marker_expression_emptied(
        tmp_path: pathlib.Path, workers: list[str]) -> None:
    """``-m`` is the operator's selection: a listed gate whose every test
    the expression excludes is not a silence.  Measured on the release
    node's card leg, ``-m "gpu and not slow and not network"``: six
    CPU-only gates the expression emptied turned a leg with zero
    failures into exit 1 (proof/node-reds-276).  Under xdist the
    deselection happens on the workers, so that route is held too."""
    done = _synthetic_leg(
        tmp_path, "tests/test_gate.py\n",
        "import pytest\n\n\n@pytest.mark.gpu\ndef test_one():\n"
        "    assert True\n",
        files={"tests/test_other.py": "def test_ok():\n    assert True\n"},
        args=["-m", "not gpu", "tests/test_gate.py", "tests/test_other.py"]
        + workers)
    assert "COLLECTED NOTHING" not in done.stdout, done.stdout
    assert done.returncode == 0, done.stdout + done.stderr


def test_the_guard_fails_a_gate_that_skipped_every_test(
        tmp_path: pathlib.Path) -> None:
    """THE NEGATIVE CONTROL.  This is the RRTMG longwave shape exactly.

    Without the guard this session is ``2 skipped``, exit 0, green -- and the
    census floor of 2 is met by the two skipped items.
    """

    done = _synthetic_leg(
        tmp_path, "tests/test_gate.py\n",
        "import pytest\n"
        "pytestmark = pytest.mark.skipif(True, reason='fixtures absent')\n"
        "def test_one():\n"
        "    assert True\n"
        "def test_two():\n"
        "    assert True\n")
    assert done.returncode != 0, (
        "a must-run gate skipped every one of its tests and the session "
        f"still exited 0:\n{done.stdout}\n{done.stderr}")
    assert "MUST-RUN GATE ENTIRELY SKIPPED" in done.stdout, done.stdout
    assert "tests/test_gate.py" in done.stdout, done.stdout


def test_the_guard_leaves_a_gate_that_ran_alone(
        tmp_path: pathlib.Path) -> None:
    """The other direction: the guard must not fail a leg that is fine."""

    done = _synthetic_leg(
        tmp_path, "tests/test_gate.py\n",
        "def test_one():\n"
        "    assert True\n"
        "def test_two():\n"
        "    assert True\n")
    assert done.returncode == 0, f"{done.stdout}\n{done.stderr}"
    assert "MUST-RUN GATE" not in done.stdout, done.stdout


def test_the_guard_fails_a_gate_that_collected_nothing(
        tmp_path: pathlib.Path) -> None:
    """``allow_module_level`` produces no test reports at all, only a skip.

    This is the arm that catches a gate emptied rather than marked out.
    """

    done = _synthetic_leg(
        tmp_path, "tests/test_gate.py\n",
        "import pytest\n"
        "pytest.skip('no fixtures', allow_module_level=True)\n"
        "def test_one():\n"
        "    assert True\n")
    assert done.returncode != 0, f"{done.stdout}\n{done.stderr}"
    assert "MUST-RUN GATE COLLECTED NOTHING" in done.stdout, done.stdout


def test_the_guard_fails_a_gate_that_skipped_above_its_ceiling(
        tmp_path: pathlib.Path) -> None:
    """A declared ceiling is a floor on coverage, not permission to shrink."""

    body = ("import pytest\n"
            "def test_one():\n"
            "    assert True\n"
            "def test_two():\n"
            "    pytest.skip('data dependent')\n"
            "def test_three():\n"
            "    pytest.skip('data dependent')\n")
    over = _synthetic_leg(tmp_path / "over",
                          "tests/test_gate.py  skips<=1\n", body)
    assert over.returncode != 0, f"{over.stdout}\n{over.stderr}"
    assert "ABOVE ITS CEILING" in over.stdout, over.stdout

    under = _synthetic_leg(tmp_path / "under",
                           "tests/test_gate.py  skips<=2\n", body)
    assert under.returncode == 0, f"{under.stdout}\n{under.stderr}"
    assert "MUST-RUN GATE" not in under.stdout, under.stdout


def test_an_unbounded_entry_still_fails_when_the_gate_skips_everything(
        tmp_path: pathlib.Path) -> None:
    """``skips<=*`` relaxes the count, never the "it ran at all" question."""

    done = _synthetic_leg(
        tmp_path, "tests/test_gate.py  skips<=*  # data dependent\n",
        "import pytest\n"
        "def test_one():\n"
        "    pytest.skip('every column was night')\n")
    assert done.returncode != 0, f"{done.stdout}\n{done.stderr}"
    assert "MUST-RUN GATE ENTIRELY SKIPPED" in done.stdout, done.stdout


@pytest.mark.parametrize("manifest", [
    "", "tests/test_gate.py skips<=bad\n", "tests/test_gate.py skips<=-1\n",
    "tests/test_gate.py skipz<=2\n", "tests/test_gate.py skips<=*\n",
    "tests/test_gate.py\ntests/test_gate.py\n", "../test_gate.py\n",
    "./tests/test_gate.py\n", "C:/tests/test_gate.py\n", r"tests\test_gate.py",
])
def test_unusable_manifest_cannot_disable_the_guard(tmp_path, manifest):
    done = _synthetic_leg(tmp_path, manifest, "def test_one():\n    assert True\n")
    assert done.returncode != 0
    assert "required must-run manifest" in done.stderr + done.stdout


def test_missing_manifest_is_a_configuration_failure(tmp_path):
    from tools.battery.no_silent_skip import _gates
    with pytest.raises(pytest.UsageError, match="required must-run manifest"):
        _gates(tmp_path)


# ---------------------------------------------------------------------------
# DECLARED TEST EXCEPTIONS: one test id, not a file.
# ---------------------------------------------------------------------------

_EXCEPTION_BODY = ("import pytest\n"
                   "def test_one():\n"
                   "    assert True\n"
                   "def test_broken():\n"
                   "    assert False, 'the defect the ruling names'\n"
                   "@pytest.mark.parametrize('x', [1, 2])\n"
                   "def test_param(x):\n"
                   "    assert x == 1\n")


def _declared(entry: str) -> str:
    return ("tests/test_other.py\n"
            "# --- DECLARED-SKIPPING-BEGIN ---\n"
            f"{entry}  a reason of at least four words\n")


@pytest.mark.parametrize("workers", [[], ["-n", "2"]],
                         ids=["in-process", "xdist"])
def test_a_declared_test_id_skips_exactly_that_test(
        tmp_path: pathlib.Path, workers: list[str]) -> None:
    """The named test skips with the declared reason; its neighbours run.

    Without the entry this leg is red on test_broken.  With it, exactly one
    test skips and the other two still pass, so the file keeps gating.
    """
    files = {"tests/test_other.py": "def test_ok():\n    assert True\n"}
    args = ["-rs", "tests/test_gate.py", "tests/test_other.py"] + workers
    red = _synthetic_leg(tmp_path / "red", _declared("tests/test_absent.py::test_x"),
                         _EXCEPTION_BODY, files=files, args=args)
    assert red.returncode != 0, red.stdout

    done = _synthetic_leg(tmp_path / "declared",
                          _declared("tests/test_gate.py::test_broken"),
                          _EXCEPTION_BODY.replace("assert x == 1", "assert x"),
                          files=files, args=args)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "1 skipped" in done.stdout and "4 passed" in done.stdout, done.stdout
    assert "declared exception, tools/battery/must_run_gates.txt: " \
        "a reason of at least four words" in done.stdout, done.stdout


def test_a_declared_id_without_brackets_covers_every_parametrization(
        tmp_path: pathlib.Path) -> None:
    done = _synthetic_leg(tmp_path, _declared("tests/test_gate.py::test_param"),
                          _EXCEPTION_BODY.replace("assert False", "assert True"),
                          args=["tests/test_gate.py"])
    assert done.returncode == 0, done.stdout + done.stderr
    assert "2 skipped" in done.stdout and "2 passed" in done.stdout, done.stdout

    one = _synthetic_leg(tmp_path / "one",
                         _declared("tests/test_gate.py::test_param[2]"),
                         _EXCEPTION_BODY.replace("assert False", "assert True"),
                         args=["tests/test_gate.py"])
    assert one.returncode == 0, one.stdout + one.stderr
    assert "1 skipped" in one.stdout and "3 passed" in one.stdout, one.stdout


def test_a_declared_id_that_matches_nothing_stops_the_session(
        tmp_path: pathlib.Path) -> None:
    """An exception that outlived its test must not read as declared."""
    done = _synthetic_leg(tmp_path, _declared("tests/test_gate.py::test_renamed"),
                          _EXCEPTION_BODY, args=["tests/test_gate.py"])
    assert done.returncode != 0
    assert "match no collected test" in done.stdout + done.stderr


def test_a_declared_exception_does_not_count_against_a_must_run_ceiling(
        tmp_path: pathlib.Path) -> None:
    manifest = ("tests/test_gate.py\n"
                "# --- DECLARED-SKIPPING-BEGIN ---\n"
                "tests/test_gate.py::test_broken  a reason of at least four words\n")
    done = _synthetic_leg(tmp_path, manifest,
                          _EXCEPTION_BODY.replace("assert x == 1", "assert x"),
                          args=["tests/test_gate.py"])
    assert done.returncode == 0, done.stdout + done.stderr
    assert "MUST-RUN GATE" not in done.stdout, done.stdout


def test_the_tree_declares_no_test_exception() -> None:
    """The one exception the 2.8.6 ruling made (the mp=28 g3 gate,
    6e26fd29c) died with its defect: the accumulator rework on
    development branch puts all 22 oracle fixtures inside the flat 2e-6
    bound, so g3 gates for real.  Any new exception is a new ruling."""
    assert sorted(declared_test_exceptions(DECLARED)) == []
