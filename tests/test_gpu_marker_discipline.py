"""``-m "not gpu"`` must open no CUDA device.  This proves it, per module.

The exclusion used to depend on authors writing ``@pytest.mark.gpu`` by hand.
Five Noah-MP CUDA modules did not, gating only on
``pytest.importorskip("cupy")``, and an unmarked test is not excluded by
``not gpu`` -- so a run believed to be CPU-only compiled and ran 34 CUDA gates
on a machine whose owner had asked that no GPU work happen there.

``conftest.pytest_collection_modifyitems`` now applies the marker
automatically.  These tests exist so that automation cannot be quietly removed
or narrowed: they assert the property (*no cupy-importing test survives
``-m "not gpu"``*) rather than the mechanism, so they keep working if the
mechanism is rewritten and fail if it is deleted.

The *detector* checks are pure source inspection.  The *selection* checks are
not, and the docstring used to claim otherwise.  They shell out to ``pytest
--collect-only`` once per module, deliberately with ``GPUWM_NO_LOCAL_GPU``
stripped, and several CUDA modules probe the device at import time -- one
``pytest.importorskip("cupy")`` plus ``getDeviceCount()`` at module scope, and
``pytest.skip(..., allow_module_level=True)`` when it throws.  So these tests
do open a device, at one remove, and their answers depend on this host having
one that answers.

That is why a module which skips itself at import is now reported as
*unanswerable* rather than as a marking failure: with the card saturated,
``getDeviceCount()`` does not return cleanly, the RRTMG modules skip at
collection, and the gate used to read zero-selected as "this coverage runs on
no machine" and go red -- on a busy card, at a release cut, pointing at a merge
it had nothing to do with.
"""

from __future__ import annotations

import functools
import pathlib
import re
import subprocess
import sys
import warnings

import pytest

from conftest import (CUPY_INSTALLED, _cupy_import_chain,
                      _cupy_install_scope, _cupy_scope, _imports_cupy,
                      _import_time_edges)

_TESTS = pathlib.Path(__file__).resolve().parent
_ROOT = _TESTS.parent

#: pytest's own way of saying the collection is incomplete: an ``ERROR``
#: header line, or the ``N errors`` count in the terminal summary.  Anchored,
#: because "error" occurs inside legitimate node ids all over this tree.
_ERRORS_IN_COLLECTION = re.compile(
    r"^(?:ERROR\b|E\s+ImportError)|^!+ Interrupted.*error|\b\d+ errors?\b",
    re.MULTILINE)


def _cupy_modules() -> list[pathlib.Path]:
    return sorted(p for p in _TESTS.glob("test_*.py") if _imports_cupy(str(p)))


def test_some_modules_do_import_cupy():
    """Guard the guard: if this finds nothing, the detector broke."""
    found = _cupy_modules()
    assert found, (
        "no test module appears to import cupy, which cannot be right --"
        " _imports_cupy is probably broken, and every test below would then"
        " pass vacuously"
    )


#: The zero-collect guard's own banner, verbatim from
#: tools/battery/no_silent_deselection.py.  Matched as a literal because a
#: looser match is how a probe starts accepting a different finding.
_GUARD_BANNER = ("SILENT DESELECTION -- these files were run and contributed "
                 "no tests:")

#: The guard's OTHER finding.  A file that collected FEWER tests than the
#: census records is real coverage loss, and must never be read here as "this
#: selection is legitimately empty".
_CENSUS_BANNER = "COVERAGE BELOW THE CENSUS"


def _guard_answered_an_empty_selection(returncode: int, stdout: str,
                                       relative: str) -> bool:
    """Is this rc=1 the zero-collect guard overriding pytest's own rc=5?

    THE MEASURED DEFECT, on the Windows cut box 2026-09-18, with cupy
    INSTALLED -- the configuration every battery leg runs in.
    ``_collect(tests/test_acoustic.py, "not gpu")`` is the probe behind
    ``test_no_device_touching_test_survives_the_not_gpu_selection``: it asks
    whether any device-touching test survives the CPU selection.  Every test
    in that file is device-touching, so the right answer is *none survive*,
    and pytest says exactly that -- ``no tests collected (11 deselected)``,
    exit 5, which this helper's caller has always accepted.

    ``tests/conftest.py`` registers the battery's zero-collect guard on every
    run, by design: a guard that runs only when somebody remembers ``-p`` is
    not a default.  The guard sees a file named on the command line that
    contributed nothing, prints its banner and rewrites the session's exit
    status to 1.  The caller then read 1 as a crashed collection and this
    gate went red on a correct answer.

    NOT LOOSENED, and these conditions are what make that true.  ``1`` is
    read as ``5`` only when the guard printed its zero-collect banner, only
    when the files the banner names are exactly the one file this probe
    asked about, only when pytest's own ``no tests collected`` line is there
    (so the session really did end with an EMPTY selection rather than a
    partial one), and only when nothing in the output says the collection
    errored or that a census floor was missed.  A crashed collection prints
    none of that and is still refused, which
    ``test_a_crashed_collection_is_still_refused`` holds line by line.

    The guard itself is untouched.  On a battery leg it still fails by name,
    which is what caught ``pytestmark = pytest.mark.gpu`` retiring sixty RUC
    bitwise-oracle tests.  This probe is not a leg: it is a
    ``--collect-only`` question about ONE file under a deliberately narrowing
    ``-m``, so it reads the guard's verdict instead of dying on it.
    """
    if returncode != 1:
        return False
    if _GUARD_BANNER not in stdout or _CENSUS_BANNER in stdout:
        return False
    if _ERRORS_IN_COLLECTION.search(stdout):
        return False
    # pytest's own wording for exit 5.  Without it the session ended some
    # other way and an empty set would not be an answer to anything.
    if "no tests collected" not in stdout:
        return False
    lines = stdout.splitlines()
    named = set()
    for line in lines[lines.index(_GUARD_BANNER) + 1:]:
        if not line.startswith("  ") or not line.strip().endswith(".py"):
            break
        named.add(line.strip())
    return named == {relative}


#: The exact sentence an import of the absent array library ends on.
_ARRAY_LIBRARY_ABSENT = "ModuleNotFoundError: No module named 'cupy'"


def _module_cannot_import_without_the_array_library(stdout: str,
                                                    relative: str) -> bool:
    """Did the ONE file asked about fail to import for the absent library?

    MEASURED on the Linux box in the array-library-free venv, which is the
    configuration the CPU battery leg runs in.  A device-touching module
    whose import chain reaches the library cannot be imported at all there,
    so a ``--collect-only`` on it ends in a collection ERROR rather than in
    a selection.  The caller read that as a crashed collection and refused
    to answer, which turned three gates in this file red on the one host
    they are supposed to describe.

    A module that cannot be imported collects NOTHING under any selection.
    That is a true and useful answer to both questions this file asks: it
    cannot leak into the CPU selection, and the converse gate already has
    an ``unanswerable`` arm for a module that declines to collect, which
    records it and warns rather than claiming coverage it did not see.

    NOT LOOSENED.  The library must really be absent from THIS install, the
    error must name the file this probe asked about and no other, the
    missing module must be the array library by its exact sentence, and the
    run must have ended with exactly one collection error, so a second
    module erroring for any other reason is still refused.  On an install
    that HAS the library every condition below is false at the first line
    and the caller refuses exactly as it did before.
    """
    if CUPY_INSTALLED:
        return False
    if f"ERROR collecting {relative}" not in stdout:
        return False
    if _ARRAY_LIBRARY_ABSENT not in stdout:
        return False
    if _CENSUS_BANNER in stdout:
        return False
    return re.search(r"^!+ Interrupted: 1 error during collection",
                     stdout, re.MULTILINE) is not None


def _collected_test_name(node_id: str) -> str:
    """Use the function-name scope shared by _cupy_scope and the auto-marker.

    Class methods collect as File::Class::test_method[parameters]. Taking
    the first component discarded their device coverage.
    """
    return node_id.rsplit("::", 1)[-1].split("[", 1)[0]


@pytest.mark.parametrize("node_id, expected", [
    ("tests/test_demo.py::test_plain", "test_plain"),
    ("tests/test_demo.py::Device::test_step", "test_step"),
    ("tests/test_demo.py::Device::test_step[member-2]", "test_step"),
])
def test_collection_names_preserve_class_method_device_coverage(node_id, expected):
    assert _collected_test_name(node_id) == expected


def _collect(path: pathlib.Path, selection: str | None) -> set[str]:
    """Test ids pytest would select from one file, without running anything.

    ``selection`` is a ``-m`` expression, or None to collect unfiltered.
    """
    # Exactly one -q: that is the mode that prints one node id per line.
    # Two (-qq) suppresses the listing entirely and this returns an empty set,
    # which would make every assertion below pass vacuously.
    # The silent-deselection guard is blocked BY NAME, and this is the
    # one place in the tree where blocking it is right: it fails a run
    # in which a file named on the command line contributed no test,
    # which is a leg that silently ran nothing -- but this probe is not
    # a leg.  It asks one file what survives one -m expression, and for
    # a file whose whole suite carries `pytestmark = pytest.mark.gpu`
    # the answer under `not gpu` is zero BY DESIGN, which the guard
    # reported as silent deselection at rc=1.  The check below then read
    # that rc as a crashed collection and failed this gate on every such
    # file (measured on tests/test_acoustic.py, 0 of 11 under `not
    # gpu`).  rc=5, the code for an empty selection, is what this probe
    # is built to read, and it is what it gets back now.
    command = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider",
               "-p", "no:no_silent_deselection_guard",
               "--collect-only", "-q"]
    if selection is not None:
        command += ["-m", selection]
    command.append(str(path))
    proc = subprocess.run(
        command, capture_output=True, text=True, cwd=str(_ROOT),
        env={**_environ(), "PYTHONPATH": str(_ROOT)},
    )
    # 0 = something was collected, 5 = nothing matched the selection, and
    # 1 when the zero-collect guard rewrote that 5 because this probe
    # asked about a file every one of whose tests the selection removed.
    # Any other code is a crashed or erroring collection, and the empty set it
    # yields is indistinguishable from an accurate "no such tests" -- which
    # makes the leak gate below pass VACUOUSLY, the one failure mode this
    # file exists to prevent.  Say so instead of answering nothing.
    relative = path.resolve().relative_to(_ROOT).as_posix()
    if (proc.returncode not in (0, 5)
            and not _guard_answered_an_empty_selection(
                proc.returncode, proc.stdout, relative)
            and not _module_cannot_import_without_the_array_library(
                proc.stdout, relative)):
        raise AssertionError(
            f"collection subprocess failed (rc={proc.returncode}) for "
            f"{path.name} with -m {selection!r}; treating that as 'no tests' "
            f"would make this gate pass vacuously.\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    ids = set()
    for line in proc.stdout.splitlines():
        line = line.strip()
        if "::" in line and not line.startswith(("<", "=")):
            ids.add(_collected_test_name(line))
    return ids


@functools.lru_cache(maxsize=None)
def _tree_collect(selection: str | None) -> dict[str, frozenset[str]] | None:
    """One whole-tree collection, grouped by file: ``{name: {test ids}}``.

    WHY THIS IS ONE SUBPROCESS AND NOT NINETY
    -----------------------------------------
    This file used to spawn ``pytest --collect-only`` once per
    cupy-importing module per selection -- about 260 subprocesses.  The
    2026-08-13 test-estate audit priced it at 266 s and named it the second
    most expensive file in the battery; re-measured on this branch it is
    **311 s on the Windows cut box** (process spawn is dearer there) and
    122 s on a Linux node.  pytest can collect the whole tree in one pass
    and the answers are identical, because a ``-m`` selection is applied per
    item, not per invocation.

    Returns ``None`` when the whole-tree collection cannot be trusted, in
    which case every caller falls back to the per-file subprocess.  That
    fallback is not decoration: a single module that raises at import turns
    a whole-tree collect into a partial answer, and a partial answer read as
    a complete one is exactly the vacuous green this file exists to prevent.
    """

    command = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider",
               "--collect-only", "-q"]
    if selection is not None:
        command += ["-m", selection]
    command.append(str(_TESTS))
    proc = subprocess.run(
        command, capture_output=True, text=True, cwd=str(_ROOT),
        env={**_environ(), "PYTHONPATH": str(_ROOT)})
    if proc.returncode not in (0, 5):
        return None
    # A collection ERROR is reported in the summary while OTHER files still
    # collect, so rc alone does not say the answer is complete.  Silently
    # keeping a partial result would under-report every file that failed to
    # import, and an under-report here reads as "no leaking tests" -- the
    # vacuous green this file exists to prevent.  Fall back instead.
    #
    # Matched precisely, on the summary line pytest actually writes.  A
    # substring search for "error" anywhere in stdout was tried and is
    # useless: node ids contain the word (test_..._error..., k_sflx_error),
    # so it tripped on every run and this optimisation silently did nothing
    # -- 311 s became 296 s instead of 30 s, which is how it was caught.
    if _ERRORS_IN_COLLECTION.search(proc.stdout):
        return None

    by_file: dict[str, set[str]] = {}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if "::" not in line or line.startswith(("<", "=")):
            continue
        location, _, rest = line.partition("::")
        name = pathlib.PurePath(location).name
        if not name.startswith("test_") or not name.endswith(".py"):
            continue
        by_file.setdefault(name, set()).add(_collected_test_name(rest))
    if not by_file:
        return None
    return {name: frozenset(ids) for name, ids in by_file.items()}


def _node_ids(selection: str | None, path: pathlib.Path) -> set[str]:
    """Test ids pytest would select from one file under one ``-m`` filter.

    Served from the single whole-tree collection when that succeeded, and
    from this file's original one-subprocess-per-file path when it did not.
    Both routes answer the same question; only the cost differs.
    """

    tree = _tree_collect(selection)
    if tree is None:
        return _collect(path, selection)
    return set(tree.get(path.name, frozenset()))


#: What both whole-tree gates below say when they cannot be answered.
_NO_TREE_COLLECTION = (
    "the whole-tree collection reports errors on this install, because the "
    "device modules cannot be imported without the array library, and the "
    "per-file route would ask {count} collection subprocesses a question "
    "most of them cannot answer (MEASURED: about 12 s each, so this gate "
    "alone runs near an hour to learn that they collect nothing). Every "
    "battery leg installs the library, and there one collection serves "
    "every module")


def _cannot_answer_a_whole_tree_question(selection: str) -> str | None:
    """Why neither whole-tree gate can be answered here, or None.

    Only ever true with the array library ABSENT, which is not a
    configuration any battery leg runs in, and only when the cheap route
    is genuinely unavailable: on an install that HAS the library the
    whole-tree collection succeeds and both gates run over every module
    from it.
    """
    if CUPY_INSTALLED:
        return None
    if _tree_collect(selection) is not None:
        return None
    return _NO_TREE_COLLECTION.format(count=len(_cupy_modules()))


def test_no_device_touching_test_survives_the_not_gpu_selection():
    """The property that matters, at the granularity that matters.

    Per *test*, not per module.  The coarse form -- "no test in a
    cupy-importing file survives" -- was too blunt: ``test_preflight.py`` has
    one cupy import among ~200 tests that deliberately *stub* cupy to exercise
    CPU paths, and marking the file whole deleted the VRAM preflight's only
    automated evidence.  Over-marking is not a safe failure either; it is
    coverage loss wearing a safety costume.
    """
    unanswerable = _cannot_answer_a_whole_tree_question("not gpu")
    if unanswerable is not None:
        pytest.skip(unanswerable)
    leaked = {}
    for path in _cupy_modules():
        whole, functions = _cupy_scope(str(path))
        surviving = _node_ids("not gpu", path)
        bad = surviving if whole else (surviving & set(functions))
        if bad:
            leaked[path.name] = sorted(bad)
    assert not leaked, (
        "these tests open a CUDA device yet survive -m \"not gpu\", so a run"
        f" believed to be CPU-only would use the local card: {leaked}"
    )


def test_every_device_touching_test_is_selected_under_m_gpu():
    """The converse: GPU coverage must not run nowhere.

    A test in neither selection is exercised on no machine, which is how 34
    Noah-MP CUDA gates passed locally while never running on the rented card.

    Zero-selected has two causes that must not be conflated.  A *marking* gap
    is a real defect and fails here.  A module that skipped itself at import
    -- because its module-scope ``getDeviceCount()`` probe found no device, or
    found one too busy to answer -- collected nothing under any selection, so
    this host has no evidence about its marking either way.  Reporting that as
    "runs on no machine" is a false alarm, and it fired as one: it turned this
    gate red at a release cut on a saturated card, naming two RRTMG modules
    that were byte-identical to the ones that had passed an hour earlier.
    """
    cannot = _cannot_answer_a_whole_tree_question("gpu")
    if cannot is not None:
        pytest.skip(cannot)
    missing = {}
    unanswerable = []
    for path in _cupy_modules():
        whole, functions = _cupy_scope(str(path))
        selected = _node_ids("gpu", path)
        if not selected and not _node_ids(None, path):
            unanswerable.append(path.name)
            continue
        if whole:
            if not selected:
                missing[path.name] = ["<entire module>"]
            continue
        absent = sorted(set(functions) - selected)
        if absent:
            missing[path.name] = absent
    assert not missing, (
        "these tests open a CUDA device but are selected by neither -m gpu nor"
        f" -m \"not gpu\", so they run on no machine: {missing}"
    )
    if unanswerable and len(unanswerable) == len(_cupy_modules()):
        # Every module declined to collect: this host answered nothing at all,
        # and a green here would be pure vacuum.
        pytest.skip(
            "no cupy module could be collected on this host (no device, or a"
            f" card too busy to answer): {sorted(unanswerable)}"
        )
    if unanswerable:
        warnings.warn(
            "marking unverified for modules that skip themselves at import on"
            f" this host: {sorted(unanswerable)}",
            stacklevel=2,
        )


def test_a_module_that_only_stubs_cupy_keeps_its_cpu_coverage():
    """Guard the granularity itself, with the case that motivated it.

    ``tests/test_preflight.py`` stubs cupy to test CPU failure paths and has a
    single genuine import.  If a future change re-coarsens the rule, its ~46
    CPU tests vanish silently -- and the VRAM preflight is a correctness bar
    on this hardware, not a nicety.
    """
    path = _TESTS / "test_preflight.py"
    if not path.exists():
        pytest.skip("test_preflight.py not present")
    whole, functions = _cupy_scope(str(path))
    assert not whole, (
        "test_preflight.py is marked gpu wholesale; it stubs cupy for CPU"
        " paths and only a couple of its tests really open a device"
    )
    cpu_side = _node_ids("not gpu", path)
    assert len(cpu_side) > 10, (
        f"only {len(cpu_side)} CPU test(s) survive in test_preflight.py --"
        " the VRAM preflight has effectively lost its coverage"
    )


@pytest.mark.parametrize("spelling", [
    "import cupy",
    "import cupy as cp",
    "from cupy import ndarray",
    "import cupy.cuda",
    'pytest.importorskip("cupy")',
])
def test_the_detector_sees_every_spelling(tmp_path, spelling):
    """A detector that misses a spelling is a detector that fails open."""
    module = tmp_path / "test_scratch_probe.py"
    module.write_text(
        f"import pytest\n{spelling}\n\n\ndef test_x():\n    pass\n",
        encoding="utf-8",
    )
    assert _imports_cupy(str(module)), f"missed spelling: {spelling!r}"


def test_the_detector_is_quiet_on_modules_that_only_mention_cupy(tmp_path):
    """A comment or a string is not an import; over-marking hides real tests."""
    module = tmp_path / "test_scratch_mentions.py"
    module.write_text(
        '"""Docstring mentioning cupy."""\n'
        "# cupy is not imported here\n"
        'NOTE = "cupy"\n\n\ndef test_x():\n    pass\n',
        encoding="utf-8",
    )
    assert not _imports_cupy(str(module))


# --------------------------------------------------------------------------
# the runtime ban: guards no import style can dodge
# --------------------------------------------------------------------------
#
# The AST layer above answers "which tests' own source imports cupy".  It is
# structurally blind to a test that reaches the device through an intermediary
# module, and that blindness was exploited for real: a gpu-marked test whose
# only import was a lazy ``from tilestream import multigpu`` inside the test
# body carried the marker, dodged the AST-driven skip, and ran on the local
# card during a mandated CPU-only invocation.  Two closures, each pinned here
# red-on-revert:
#
# * marker implies skip -- ``pytest_collection_modifyitems`` bans every item
#   CARRYING the gpu marker, not just its own AST hits;
# * the device-visibility backstop -- under GPUWM_NO_LOCAL_GPU=1 the conftest
#   sets ``CUDA_VISIBLE_DEVICES=-1`` before any test runs, so an escaped
#   unmarked test's first device use goes red (cudaErrorNoDevice) instead of
#   silently running on the owner's card, whatever its import style.


def _run_banned(path: pathlib.Path, cwd: pathlib.Path) -> tuple:
    """Run one test file in a subprocess WITH the local-GPU ban set.

    For a scratch file outside tests/, the tests/ conftest is loaded
    explicitly with ``-p conftest`` because directory walking would not find
    it -- and these tests exist precisely to prove what that conftest
    enforces.  A file inside tests/ picks it up normally, and loading it
    twice would double-register the plugin.
    """
    import os
    # CUDA_VISIBLE_DEVICES is deliberately stripped: the conftest under test
    # must plant it itself, and an inherited copy would mask a reverted
    # backstop.
    env = {k: v for k, v in os.environ.items()
           if k != "CUDA_VISIBLE_DEVICES"}
    # tests/ BEFORE the repository root: d51e5c2f3 (2026-08-29) planted a
    # root conftest.py (the line-ending hook armer), and with the root
    # first ``-p conftest`` imported THAT module instead of the ban
    # plugin under test -- the scratch run then saw the local card and
    # both ban assertions went red.  The order is what selects which
    # ``conftest`` the plugin flag names.
    env.update({"GPUWM_NO_LOCAL_GPU": "1",
                "PYTHONPATH": os.pathsep.join([str(_TESTS), str(_ROOT)])})
    command = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider"]
    if _TESTS not in path.parents:
        # The explicitly loaded project plugin also requires its manifest.
        # Give this scratch pytest root an exact copy of the canonical list;
        # neither the must-run checks nor the GPU-ban plugin is disabled.
        relative_manifest = pathlib.Path("tools/battery/must_run_gates.txt")
        manifest = cwd / relative_manifest
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_bytes((_ROOT / relative_manifest).read_bytes())
        command += ["-p", "conftest"]
    command += ["-q", str(path)]
    proc = subprocess.run(command, capture_output=True, text=True,
                          cwd=str(cwd), env=env)
    return proc.returncode, proc.stdout + proc.stderr


def test_the_ban_hides_every_device_from_an_unmarked_escapee(tmp_path):
    """An UNMARKED test reaching CUDA through an INTERMEDIARY sees no card.

    This is the escape the AST automation cannot see: the test file itself
    never mentions cupy -- a helper module does, imported lazily in the test
    body, which is exactly how ``test_multigpu_forced_gpu.py`` reached the
    local card (its intermediary was ``tilestream.multigpu``).  The scratch
    test asserts it CAN see a device, so under the backstop it fails
    red-loud; if the backstop is reverted on a machine with a card it
    passes -- an escaped test would once again run GPU work locally -- and
    THIS test goes red.

    The probe is ``getDeviceCount`` only: under the backstop there is no
    device to see, and on revert enumeration alone opens no context, so
    neither outcome runs work on the owner's card.
    """
    if not CUPY_INSTALLED:
        # The intermediary below IMPORTS the array library, so on an install
        # without it the scratch test ends on that import and the run
        # reports a collection error rather than the "1 failed" this gate
        # reads as the backstop working.  The gate is about the backstop
        # hiding a card that is there; an install with no array library
        # cannot reach a card at all, and there is nothing here to see.
        pytest.skip("this gate needs the array library, because the escape "
                    "it reproduces imports it")
    helper = tmp_path / "scratch_gpu_helper.py"
    helper.write_text(
        "import cupy\n\n\n"
        "def visible_devices():\n"
        "    try:\n"
        "        return cupy.cuda.runtime.getDeviceCount()\n"
        "    except Exception:\n"
        "        return 0  # cudaErrorNoDevice: the ban, doing its job\n",
        encoding="utf-8")
    scratch = tmp_path / "test_scratch_lazy_gpu.py"
    scratch.write_text(
        "def test_reaches_for_the_device():\n"
        "    import scratch_gpu_helper  # the intermediary dodge under"
        " test\n"
        "    assert scratch_gpu_helper.visible_devices() > 0\n",
        encoding="utf-8")
    rc, out = _run_banned(scratch, tmp_path)
    assert rc != 0 and "1 failed" in out, (
        "an unmarked test reaching CUDA through an intermediary could still"
        " see a device under GPUWM_NO_LOCAL_GPU=1 -- the CUDA_VISIBLE_DEVICES"
        f" backstop is gone and the local card is reachable again (rc={rc}):"
        f"\n{out}")


def test_a_marked_test_with_clean_source_is_still_skipped(tmp_path):
    """The gpu MARKER alone must trigger the ban skip, without any AST hit.

    A skip, not a run: the marked test never executes at all.  If
    marker-implies-skip is narrowed back to AST-detected items only, the
    test below RUNS (and passes, since importing cupy without touching the
    device is legal) -- turning this "1 skipped" assertion red.
    """
    helper = tmp_path / "scratch_gpu_helper2.py"
    helper.write_text("import cupy\nTOUCHED = cupy.ndarray\n",
                      encoding="utf-8")
    scratch = tmp_path / "test_scratch_marked_gpu.py"
    scratch.write_text(
        "import pytest\n\n"
        "pytestmark = pytest.mark.gpu\n\n\n"
        "def test_transitive_device_use():\n"
        "    # An intermediary, not cupy itself: no AST hit in THIS file,\n"
        "    # so only the marker can trigger the skip.  Never reached when\n"
        "    # the marker skip works.\n"
        "    import scratch_gpu_helper2\n"
        "    assert scratch_gpu_helper2.TOUCHED is not None\n",
        encoding="utf-8")
    rc, out = _run_banned(scratch, tmp_path)
    assert rc == 0 and "1 skipped" in out, (
        "a gpu-marked test with no cupy in its own source was not skipped"
        f" under the ban (rc={rc}) -- marker-implies-skip has regressed:"
        f"\n{out}")


def test_the_original_escapee_is_skipped_under_the_ban():
    """The file that actually ran on the forbidden card, pinned by name."""
    path = _TESTS / "test_multigpu_forced_gpu.py"
    if not path.exists():
        pytest.skip("test_multigpu_forced_gpu.py not present")
    rc, out = _run_banned(path, _ROOT)
    assert rc == 0 and "skipped" in out and "passed" not in out, (
        f"test_multigpu_forced_gpu.py was not skipped under the ban"
        f" (rc={rc}):\n{out}")


def _environ() -> dict:
    import os
    # Never inherit the ban into the subprocess: these are collection-only
    # runs, and the ban would skip the very items being counted.  The
    # device-visibility backstop travels with the ban and is stripped for
    # the same reason -- modules that probe the device at import would
    # otherwise all collect as "unanswerable".
    env = {k: v for k, v in os.environ.items()
           if k not in ("GPUWM_NO_LOCAL_GPU", "CUDA_VISIBLE_DEVICES")}
    return env


# --------------------------------------------------------------------------
# the second property: cupy ABSENT, and the marker one indirection out
# --------------------------------------------------------------------------
#
# Everything above is about OPENING A DEVICE.  There is a second question the
# AST detector cannot answer and, until 2026-09-17, nothing did: can this
# selection be COLLECTED at all on an install that has no cupy?
#
# It could not.  Measured on test host 1 in a venv with no cupy extra,
# ``-m "not gpu and not slow and not network"`` over the CPU battery lists
# ended in ``Interrupted: 6 errors during collection`` and ran NOTHING, and
# seven further tests were collected and failed with ModuleNotFoundError --
# five in tests/test_experiment.py and two in
# tests/test_physics_md_aerosol_claims.py, none of which names cupy in its own
# source.  They reach it through ``woof/core/physics.py`` and
# ``woof/core/mynn_pbl_runtime.py``, one indirection out, and a detector that
# reads only the module's own source is structurally blind to every one.
#
# WHY THE REMEDY IS NOT "MARK THEM gpu", which was the first thing tried.  The
# closure below flags 26 test modules whole and 120 more in part; none of them
# opens a device, and ``gpu`` implies a skip wherever local GPU work is banned,
# so marking them would retire that coverage on the very machine that runs the
# CPU battery.  Over-marking is coverage loss wearing a safety costume, in this
# file's own words.  The tree's own position is the opposite one:
# ``tools/battery/provision_battery_venv.ps1`` installs the ``all`` extra for
# the CPU legs and says in writing that stage-1 files import cupy at
# collection.  cupy is PROVISIONED there by design.
#
# So the remedy matches the property: when cupy is ABSENT the conftest does not
# import a module whose import closure needs it, and skips by name a test whose
# body reaches one.  When cupy is PRESENT -- every battery leg -- not one item
# changes.  The tests below pin both halves, and the one that matters most is
# the closure going PAST the module's own source, because that is the blindness
# the whole item is about.


def _install_closure_census() -> tuple[list, list, list]:
    """Split tests/ three ways by how it reaches cupy AT IMPORT TIME.

    ``(direct, indirect, by_function)`` -- the module's own source, one or
    more first-party hops, and a test body's own import.
    """
    direct, indirect, by_function = [], [], []
    for path in sorted(_TESTS.glob("test_*.py")):
        _edges, own = _import_time_edges(str(path))
        whole, functions = _cupy_install_scope(str(path))
        if own:
            direct.append(path.name)
        elif whole:
            indirect.append((path.name, whole))
        if functions:
            by_function.append((path.name, sorted(functions)))
    return direct, indirect, by_function


def test_the_install_closure_reaches_past_the_modules_own_source():
    """GUARD THE GUARD, and the reason this item exists.

    If this finds nothing, the closure has been narrowed back to a scan of
    the module's own source and every test below passes vacuously -- which
    is the state the tree was in when seven tests were collected on a
    CPU-only install and failed with ModuleNotFoundError.
    """
    direct, indirect, by_function = _install_closure_census()
    assert direct, (
        "no test module imports cupy in its own source, which cannot be"
        " right: the closure's base case is broken and everything below is"
        " vacuous")
    assert indirect, (
        "the install closure flags no module that does not name cupy"
        " itself, so it has been narrowed to a scan of the module's own"
        " source -- exactly the blindness that let tests/test_experiment.py"
        " and tests/test_physics_md_aerosol_claims.py be collected without"
        " cupy and fail with ModuleNotFoundError")
    assert by_function, (
        "the closure sees no test body reaching a device module, so the"
        " function-level half is gone")
    assert any(" -> " in why for _name, why in indirect), (
        "every flagged chain is one hop long, so the closure stops at the"
        f" first intermediary instead of following it: {indirect}")


def test_the_install_closure_covers_everything_the_direct_detector_sees():
    """A module-scope ``import cupy`` is a whole-module answer, always.

    The closure is the wider question, so it may never answer less than the
    narrow one.  If it does, a module that plainly cannot be imported
    without cupy is imported anyway and the collection dies.
    """
    missed = [path.name for path in sorted(_TESTS.glob("test_*.py"))
              if _import_time_edges(str(path))[1]
              and _cupy_install_scope(str(path))[0] is None]
    assert not missed, (
        "these modules import cupy at their own module scope yet the"
        f" install closure calls them importable without it: {missed}")


def test_the_closure_sees_one_indirection_in_both_import_spellings(tmp_path):
    """The synthetic case, through a REAL intermediary.

    Both spellings are pinned because reading only the left half of a
    ``from woof.core import physics`` missed precisely the route the two
    measured files take: that statement names a MODULE in its alias list,
    not an attribute.
    """
    for index, (spelling, expected) in enumerate((
            ("from woof.core import physics", "woof.core.physics"),
            ("import woof.core.mynn_pbl_runtime",
             "woof.core.mynn_pbl_runtime"),
            ("from woof.core.physics import initialize_physics",
             "woof.core.physics"))):
        # A DISTINCT file per spelling: both closure helpers are memoised by
        # path, so reusing one name would answer the first spelling three
        # times and the other two would pass on someone else's result.
        module = tmp_path / f"test_scratch_indirect_{index}.py"
        module.write_text(
            spelling + "\n\n\ndef test_x():\n    pass\n", encoding="utf-8")
        _cupy_install_scope.cache_clear()
        whole, functions = _cupy_install_scope(str(module))
        assert whole and expected in whole, (
            f"a module whose import runs {spelling!r} was called importable"
            f" without cupy: {whole!r}")
        assert not functions, (
            "a whole-module answer must not also name functions; the"
            " collection never gets that far")


def test_the_closure_sees_an_indirection_inside_a_test_body(tmp_path):
    """The function-level half, which decides a SKIP rather than a drop.

    A body-level import fails at call time, so the module collects fine and
    only that item is skipped -- with the edge that did it in the reason.
    This is the shape of the two tests in
    tests/test_physics_md_aerosol_claims.py.
    """
    module = tmp_path / "test_scratch_body_indirect.py"
    module.write_text(
        "def test_reaches_a_device_module():\n"
        "    from woof.core.mynn_pbl_runtime import mynn_flag_qs\n"
        "    assert mynn_flag_qs is not None\n\n\n"
        "def test_touches_nothing():\n"
        "    assert True\n",
        encoding="utf-8")
    _cupy_install_scope.cache_clear()
    whole, functions = _cupy_install_scope(str(module))
    assert whole is None, (
        "a body-level import is not an import-time edge; calling it one"
        " drops a module that collects perfectly well")
    named = {row.split(":", 1)[0] for row in functions}
    assert named == {"test_reaches_a_device_module"}, (
        "the function-level closure named the wrong set of tests:"
        f" {sorted(functions)}")


def test_the_two_measured_files_are_the_ones_the_closure_names():
    """The named regression, pinned by file and by test name.

    tests/test_physics_md_aerosol_claims.py is the half a source scan CAN
    reach: two tests, each importing a device module in its own body.
    tests/test_experiment.py is the half it cannot -- its five reach cupy
    through the command line at RUN time, which is a call-graph question,
    so the closure correctly says nothing and the runtime skip below is
    what covers them.  Asserting the closure's silence here is the point:
    if a later change makes it guess about runtime routes, it will start
    dropping modules that import perfectly well.
    """
    claims = _TESTS / "test_physics_md_aerosol_claims.py"
    if claims.exists():
        whole, functions = _cupy_install_scope(str(claims))
        assert whole is None, (
            "the aerosol-claims module imports no device module at module"
            f" scope; dropping it whole loses its other coverage: {whole}")
        assert len(functions) >= 2, (
            "the two tests that reached cupy through"
            " woof/core/mynn_pbl_runtime.py and woof/core/microphysics.py"
            f" are no longer named: {sorted(functions)}")
    experiment = _TESTS / "test_experiment.py"
    if experiment.exists():
        whole, _functions = _cupy_install_scope(str(experiment))
        assert whole is None, (
            "tests/test_experiment.py reaches cupy only when a test RUNS;"
            " a closure that drops it at collection is guessing")


def test_a_guarded_import_is_not_a_card_dependence(tmp_path):
    """The false-positive direction, with the module that proves it.

    ``woof/core/state.py`` imports cupy inside a ``try`` on purpose and
    records why: an absent or unloadable cupy used to kill the whole
    command line through ``cli -> downscale -> offline_child``, and
    ``run-plan --probe``, whose job is to diagnose that very install, could
    not run on it.  Reading a guarded import as a dependence would drop
    most of this tree from a cupy-less collection, which is over-marking in
    its other costume.
    """
    assert _cupy_import_chain("woof.core.state") is None, (
        "woof/core/state.py guards its cupy import deliberately; calling"
        " that a dependence drops every module that touches state")
    module = tmp_path / "test_scratch_guarded.py"
    module.write_text(
        "from woof.core import state\n\n\ndef test_x():\n"
        "    assert state is not None\n",
        encoding="utf-8")
    _cupy_install_scope.cache_clear()
    whole, _functions = _cupy_install_scope(str(module))
    assert whole is None, (
        f"a module importing a GUARDED consumer was dropped: {whole!r}")


@pytest.mark.parametrize("source, expected, cupy", [
    ("import cupy\n", set(), True),
    ("from cupy.cuda import runtime\n", set(), True),
    ("def helper():\n    import cupy\n", set(), False),
    ("async def helper():\n    import cupy\n", set(), False),
    ("class Scope:\n    import cupy\n", set(), True),
    ("class Scope:\n    def helper(self):\n        import cupy\n",
     set(), False),
    ("try:\n    import cupy\nexcept ImportError:\n    pass\n",
     set(), False),
    ("try:\n    pass\nexcept ImportError:\n    import cupy\n",
     set(), True),
    ("try:\n    pass\nexcept ImportError:\n    pass\nelse:\n    import cupy\n",
     set(), True),
    ("try:\n    pass\nfinally:\n    import cupy\n", set(), True),
    ("if enabled:\n    from woof.core import physics\n",
     {"woof.core", "woof.core.physics"}, False),
    ("try:\n    import woof.core.physics\nfinally:\n    import woof.core.state\n",
     {"woof.core.state"}, False),
])
def test_import_scope_keeps_guarded_and_function_bodies_separate(
        tmp_path, source, expected, cupy):
    """Pruning the expensive scan must preserve every import-time boundary."""
    module = tmp_path / "test_scope.py"
    module.write_text(source, encoding="utf-8")
    assert _import_time_edges(str(module)) == (frozenset(expected), cupy)


def test_import_scope_follows_siblings_only_outside_packages(tmp_path):
    """The fast scan must keep the sibling route that broke CuPy-less CI."""
    sibling = tmp_path / "test_device_helper.py"
    sibling.write_text("import cupy\n", encoding="utf-8")
    module = tmp_path / "test_consumer.py"
    module.write_text("from test_device_helper import build\n", encoding="utf-8")
    assert _import_time_edges(str(module)) == (frozenset({str(sibling)}), False)
    whole, functions = _cupy_install_scope(str(module))
    assert whole == "imports test_device_helper"
    assert not functions

    package = tmp_path / "package"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / sibling.name).write_bytes(sibling.read_bytes())
    packaged = package / module.name
    packaged.write_bytes(module.read_bytes())
    assert _import_time_edges(str(packaged)) == (frozenset(), False)


def test_nothing_is_dropped_on_an_install_that_has_cupy(monkeypatch,
                                                        tmp_path):
    """Every battery leg has cupy, and there this whole layer is dead code.

    The measured counts in the census are what they are BECAUSE cupy is
    provisioned; if the drop hook ever fired on a provisioned box it would
    silently retire 26 modules from a leg that was running them, which is
    the coverage loss this file was written to prevent.
    """
    import conftest

    module = tmp_path / "test_scratch_present.py"
    module.write_text(
        "from woof.core import physics\n\n\ndef test_x():\n    pass\n",
        encoding="utf-8")
    monkeypatch.setattr(conftest, "CUPY_INSTALLED", True)
    assert conftest.pytest_pycollect_makemodule(module, None) is None, (
        "a module was taken out of collection on an install that HAS cupy")
    _cupy_install_scope.cache_clear()
    whole, _functions = _cupy_install_scope(str(module))
    assert whole is not None, (
        "without cupy this module cannot be imported, and the hook would"
        " have nothing to act on")


def test_a_dropped_module_states_its_reason_to_the_deselection_guard(
        monkeypatch, tmp_path):
    """Loud is not silent, and the exemption is exactly the loud set.

    tools/battery/no_silent_deselection.py fails a leg when a file it was
    told to run contributes no tests, because one ``pytestmark =
    pytest.mark.gpu`` line once retired sixty bitwise-oracle tests and the
    leg stayed green. A module dropped for want of cupy contributes
    nothing either, so on a cupy-less install that guard called every drop
    a silent deselection and turned the run red -- while the drop had just
    printed its import chain, its count and its remedy.

    So the conftest writes the guard's own ZERO_COLLECT_ALLOWED entry, and
    this pins what keeps that from being a weakening: the entry carries
    the reason, and on an install that HAS cupy nothing is dropped, so the
    table stays empty and the marker fault still fails the leg by name.
    """
    import sys
    import types

    import conftest

    guard = types.ModuleType("gpuwm_no_silent_deselection")
    guard.ZERO_COLLECT_ALLOWED = {}
    monkeypatch.setitem(sys.modules, "gpuwm_no_silent_deselection", guard)
    target = _TESTS / "test_physics.py"
    conftest._state_the_reason_to_the_deselection_guard(
        target, "imports woof.core.physics")
    assert list(guard.ZERO_COLLECT_ALLOWED) == [
        "tests/test_physics.py"], guard.ZERO_COLLECT_ALLOWED
    reason = guard.ZERO_COLLECT_ALLOWED["tests/test_physics.py"]
    assert "woof.core.physics" in reason and "cupy" in reason, (
        "the exemption must carry the chain that forced it; an entry with "
        f"no reason is the silence the guard exists to catch: {reason!r}")

    # A path outside the tree writes nothing rather than an absolute key
    # the guard could never match against its own relative request set.
    outside = tmp_path / "test_elsewhere.py"
    outside.write_text("def test_x():\n    pass\n", encoding="utf-8")
    conftest._state_the_reason_to_the_deselection_guard(outside, "why")
    assert list(guard.ZERO_COLLECT_ALLOWED) == ["tests/test_physics.py"]

    # No guard loaded is a no-op, because this must never be the reason a
    # run cannot start.
    monkeypatch.delitem(sys.modules, "gpuwm_no_silent_deselection")
    conftest._state_the_reason_to_the_deselection_guard(target, "why")


def test_a_runtime_cupy_import_becomes_a_named_skip_and_nothing_else(
        monkeypatch):
    """The runtime half: the only route that can see a call-graph edge.

    The five tests in tests/test_experiment.py reach
    ``woof/core/physics.py`` through the command line and
    ``woof.core.clock.resolve_clock`` when they RUN.  No source scan can
    decide that, so the answer is taken from the exception itself -- and
    the conditions are exact, because they are the whole reason this is not
    a test loosened to pass:

    * cupy absent, so on every provisioned box the branch is dead;
    * the missing module IS cupy, so any other ModuleNotFoundError and any
      other failure of any kind re-raises untouched.

    All three directions are asserted here; drop any one condition and one
    of them goes red.
    """
    import conftest

    class _Item:
        nodeid = "tests/test_scratch.py::test_x"

    def drive(error):
        generator = conftest.pytest_runtest_call(_Item())
        next(generator)
        return generator.throw(error)

    monkeypatch.setattr(conftest, "CUPY_INSTALLED", False)
    monkeypatch.setattr(conftest, "_SKIPPED_AT_CALL_WITHOUT_CUPY", [])
    with pytest.raises(BaseException) as caught:
        drive(ModuleNotFoundError("No module named 'cupy'", name="cupy"))
    assert caught.typename == "Skipped", (
        "a test that reached cupy at RUN time on an install without it did"
        f" not become a skip: {caught.typename}")
    assert conftest._SKIPPED_AT_CALL_WITHOUT_CUPY == [_Item.nodeid], (
        "the skip was not recorded by node id, so the count is absorbed"
        " rather than printed")

    with pytest.raises(ModuleNotFoundError) as other:
        drive(ModuleNotFoundError("No module named 'zarr'", name="zarr"))
    assert other.value.name == "zarr", (
        "a DIFFERENT missing module was absorbed as a cupy skip, which"
        " would hide a real broken install")

    # THE SECOND SHAPE, measured: monkeypatch.setattr on a dotted path
    # resolves it through a loader that re-raises, so what reaches the test
    # is an ImportError and not a ModuleNotFoundError at all.
    wrapped = ImportError(
        "import error in woof.core.dycore: No module named 'cupy'")
    with pytest.raises(BaseException) as caught:
        drive(wrapped)
    assert caught.typename == "Skipped", (
        "an ImportError wrapping the absent cupy was not recognised, which"
        " left five tests failing for a module this install does not have")
    other_wrapped = ImportError(
        "import error in woof.core.dycore: No module named 'zarr'")
    with pytest.raises(ImportError) as passed_through:
        drive(other_wrapped)
    assert passed_through.value is other_wrapped, (
        "an ImportError naming a DIFFERENT module was absorbed")

    # THE THIRD SHAPE: the front door refusing ahead of the work, which is
    # not an import failure and never will be. Recognised by class name and
    # module so this file imports no part of the package it collects, and
    # built here the same way.
    missing = type("CapabilityMissing", (RuntimeError,), {})
    missing.__module__ = "woof.capabilities"
    with pytest.raises(BaseException) as caught:
        drive(missing("woof go: this command needs cupy (cupy-cuda12x)"))
    assert caught.typename == "Skipped", (
        "the command line's own cupy refusal was not recognised, which left"
        " seven tests failing on a door working exactly as designed")
    with pytest.raises(missing):
        drive(missing("woof go: this command needs a NetCDF library"))
    impostor = type("CapabilityMissing", (RuntimeError,), {})
    impostor.__module__ = "somewhere.else"
    with pytest.raises(impostor):
        drive(impostor("this command needs cupy"))

    monkeypatch.setattr(conftest, "CUPY_INSTALLED", True)
    with pytest.raises(ModuleNotFoundError) as installed:
        drive(ModuleNotFoundError("No module named 'cupy'", name="cupy"))
    assert installed.value.name == "cupy", (
        "on an install that HAS cupy a ModuleNotFoundError for it is a real"
        " failure -- absorbing it would hide a broken environment on every"
        " battery leg")


def test_the_same_answer_is_given_during_setup(monkeypatch):
    """A fixture reaching the device errors BEFORE the test body runs.

    An error in setup is reported as an ERROR rather than a failure, which
    is the shape ten tests in tests/test_domain_wizard_forcing.py took on
    this same venv for an unrelated reason, so the setup phase needs the
    same answer the call phase gives. Same conditions, and the last
    assertion is the one that keeps it exact.
    """
    import conftest

    class _Item:
        nodeid = "tests/test_scratch.py::test_setup"

    def drive(error):
        generator = conftest.pytest_runtest_setup(_Item())
        next(generator)
        return generator.throw(error)

    monkeypatch.setattr(conftest, "CUPY_INSTALLED", False)
    monkeypatch.setattr(conftest, "_SKIPPED_AT_CALL_WITHOUT_CUPY", [])
    with pytest.raises(BaseException) as caught:
        drive(ModuleNotFoundError("No module named 'cupy'", name="cupy"))
    assert caught.typename == "Skipped"
    assert conftest._SKIPPED_AT_CALL_WITHOUT_CUPY == [_Item.nodeid]
    with pytest.raises(ValueError):
        drive(ValueError("a fixture that is simply broken"))
# --------------------------------------------------------------------------
# the third property: the probe reads the guard's verdict, and nothing else
# --------------------------------------------------------------------------
#
# Measured on the Windows cut box 2026-09-18 with cupy INSTALLED, which is
# the configuration every battery leg runs in and the one this file's own
# instrument was never exercised in: the probe for
# tests/test_acoustic.py under -m "not gpu" exited 1, because the file is
# device-touching whole, the selection emptied it, and the zero-collect
# guard tests/conftest.py registers on every run rewrote pytest's exit 5.
# The probe called that a crashed collection and this file went red on the
# correct answer.  These gates hold the narrow reading that fixed it.


def _guard_output(files, *, deselected: int = 11,
                  census: bool = False, errors: bool = False) -> str:
    """The guard's stdout for a zero-collect session, as it really prints."""
    lines = ["", _GUARD_BANNER]
    lines += [f"  {name}" for name in files]
    lines.append(
        "Every test in each was deselected, skipped at collection, or "
        "removed.  The leg reported no failure for them because it ran "
        "none of them.")
    if census:
        lines += ["", _CENSUS_BANNER + " -- these files collected fewer "
                                       "tests than the census records:",
                  "  tests/test_other.py: 3 collected, 9 recorded"]
    if errors:
        lines.append("ERROR tests/test_acoustic.py")
    lines.append(f"no tests collected ({deselected} deselected)")
    return "\n".join(lines)


def test_the_guard_verdict_is_read_as_the_empty_selection_it_is():
    """The one shape that is accepted, asserted whole."""
    stdout = _guard_output(["tests/test_acoustic.py"])
    assert _guard_answered_an_empty_selection(
        1, stdout, "tests/test_acoustic.py")


def test_a_crashed_collection_is_still_refused():
    """Every shape a real breakage takes, none of them accepted.

    This is the assertion that keeps the fix from being a loosening: if any
    of these started returning True, the probe would read a broken
    collection as "no tests survive the CPU selection", which is exactly
    the vacuous green this file exists to prevent.
    """
    good = _guard_output(["tests/test_acoustic.py"])
    name = "tests/test_acoustic.py"

    # An import error, a syntax error, an internal error: no banner at all.
    assert not _guard_answered_an_empty_selection(
        1, "ImportError while importing test module", name)
    assert not _guard_answered_an_empty_selection(1, "", name)
    # A collection that errored AND deselected: the error is the finding.
    assert not _guard_answered_an_empty_selection(
        1, _guard_output([name], errors=True), name)
    assert not _guard_answered_an_empty_selection(
        1, good + "\n1 error", name)
    # Any other exit code, including the interpreter dying.
    for code in (2, 3, 4, -1, 137):
        assert not _guard_answered_an_empty_selection(code, good, name)
    # The banner without pytest's own empty-session line: the session ended
    # some other way and the empty set would not be an answer.
    partial = good.replace("no tests collected (11 deselected)",
                           "2 tests collected (9 deselected)")
    assert not _guard_answered_an_empty_selection(1, partial, name)


def test_the_verdict_must_name_the_file_the_probe_asked_about():
    """A banner about OTHER files is not an answer about this one."""
    other = _guard_output(["tests/test_other.py"])
    assert not _guard_answered_an_empty_selection(
        1, other, "tests/test_acoustic.py")
    # And a banner naming this file AND another is not either: the probe
    # named one file, so a second one means the run was not the probe's.
    both = _guard_output(["tests/test_acoustic.py", "tests/test_other.py"])
    assert not _guard_answered_an_empty_selection(
        1, both, "tests/test_acoustic.py")


def test_a_census_shortfall_is_never_read_as_an_empty_selection():
    """The guard's other finding is coverage loss and stays fatal here."""
    shrunk = _guard_output(["tests/test_acoustic.py"], census=True)
    assert not _guard_answered_an_empty_selection(
        1, shrunk, "tests/test_acoustic.py")


def _import_failure_output(named: str, *, missing: str = "cupy",
                           errors: int = 1, census: bool = False) -> str:
    """pytest's own shape for a collection stopped by one failing import."""

    lines = [_GUARD_BANNER, f"  {named}", ""]
    if census:
        lines.append(_CENSUS_BANNER)
    lines += [
        "==================================== ERRORS ====================",
        f"___________ ERROR collecting {named} ____________",
        f"ImportError while importing test module '/tree/{named}'.",
        "Traceback:",
        f"E   ModuleNotFoundError: No module named '{missing}'",
        "=========================== short test summary info ============",
        f"ERROR {named}",
        f"!!!!!!!! Interrupted: {errors} error"
        f"{'s' if errors != 1 else ''} during collection !!!!!!!!",
        f"no tests collected, {errors} error"
        f"{'s' if errors != 1 else ''} in 12.80s",
    ]
    return "\n".join(lines)


def test_only_the_absent_array_librarys_own_import_error_is_answered(
        monkeypatch):
    """The second recogniser, held line by line like the first.

    A module that cannot be imported because this install has no array
    library collects nothing under any selection, and that is a true
    answer.  Every other reason a collection errors is still a refusal,
    and on an install that HAS the library this recogniser never fires at
    all.
    """
    import test_gpu_marker_discipline as module

    name = "tests/test_noahmp_cold_start_device.py"
    monkeypatch.setattr(module, "CUPY_INSTALLED", False)
    assert module._module_cannot_import_without_the_array_library(
        _import_failure_output(name), name), (
        "the one shape this recogniser exists for was refused, which leaves"
        " three gates in this file red on the host the CPU battery runs on")

    # A DIFFERENT missing module is a broken install, not this answer.
    assert not module._module_cannot_import_without_the_array_library(
        _import_failure_output(name, missing="zarr"), name)
    # An error about some OTHER file is not an answer about this one.
    assert not module._module_cannot_import_without_the_array_library(
        _import_failure_output("tests/test_other.py"), name)
    # A second erroring module means the run was not this probe's question.
    assert not module._module_cannot_import_without_the_array_library(
        _import_failure_output(name, errors=2), name)
    # Coverage loss stays fatal here, exactly as it does for the guard.
    assert not module._module_cannot_import_without_the_array_library(
        _import_failure_output(name, census=True), name)
    # A syntax error, an internal error, an empty capture: no shape at all.
    for stdout in ("", "ImportError while importing test module",
                   "E   SyntaxError: invalid syntax"):
        assert not module._module_cannot_import_without_the_array_library(
            stdout, name)

    # ...AND ON AN INSTALL THAT HAS THE LIBRARY, never.
    monkeypatch.setattr(module, "CUPY_INSTALLED", True)
    assert not module._module_cannot_import_without_the_array_library(
        _import_failure_output(name), name), (
        "an install that HAS the array library read a failing import as an"
        " empty selection, which is the vacuous green this file prevents")


def test_the_probe_answers_on_a_module_that_is_device_touching_whole():
    """END TO END, on the real file that was red.

    Not the helper above but ``_collect`` itself, spawning the real
    subprocess against a real in-tree module every one of whose tests the
    CPU selection removes.  Before the fix this raised AssertionError with
    the guard's banner in its message; the right answer is the empty set.
    """
    if not CUPY_INSTALLED:
        # With cupy absent the module is not IMPORTED at all: the conftest
        # drops it and states its reason to the same guard, so the probe
        # gets a clean exit 5 and the -m gpu converse below collects
        # nothing either.  That is a different route with its own gates
        # above; this one is about the guard firing, which needs the
        # module to collect its tests and then have them deselected.
        pytest.skip("this route needs the module to import, so it needs "
                    "cupy; the cupy-absent route is gated above")
    whole_modules = [p for p in _cupy_modules() if _cupy_scope(str(p))[0]]
    if not whole_modules:
        pytest.skip("no test module is device-touching whole on this tree")
    path = next((p for p in whole_modules if p.name == "test_acoustic.py"),
                whole_modules[0])
    assert _collect(path, "not gpu") == set(), (
        f"{path.name} is device-touching whole, so the CPU selection must "
        "leave nothing behind")
    # The converse in the same breath, so the empty set above cannot be the
    # answer to every question: -m gpu keeps them.
    assert _collect(path, "gpu"), (
        f"{path.name} collected nothing under -m gpu either, so the empty "
        "set above says nothing about the marking")
