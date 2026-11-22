"""``tools/import_sweep.py`` imports the package and tells an absence from a fault.

THE BREAKAGE THIS PREVENTS
--------------------------
xc-06-01 of the 2026-09 audit: `.github/workflows/` held one workflow, whose
``test`` job names 17 test files by hand.  Resolving every ``woof.*`` import
in those 17 files and taking the module-scope transitive closure reaches 221
of 525 package modules -- 209,243 of 433,411 lines.  The other half of the
package was never IMPORTED by anything an automated gate ran, so a
``SyntaxError``, a bad module-scope constant or a broken internal import
there produced a green check on the pull request.

The sweep answers that in seconds with no GPU and no fixtures.  What makes it
a gate rather than a smoke test is the distinction this file exercises: a
``ModuleNotFoundError`` for a DECLARED extra is tolerated and reported, and
everything else -- including a ``ModuleNotFoundError`` for a ``woof.*``
name, which is a broken internal import wearing the same exception class --
fails.  An opportunistic ``except Exception: continue`` walk (the shape
``tests/test_doctor.py:1031-1035`` uses, correctly, for a different question)
would pass on a package that ships entirely broken.

The second breakage, found by review of the first fix: the walk caught
``Exception``, so one module's import-scope ``sys.exit()`` ended the whole
sweep with that module's status and no report -- measured as exit 2 and no
output on a box with a visible CUDA device.  A gate that the thing it
measures can terminate is worse than a narrow one, because a narrow gate at
least prints what it did check.
"""

from __future__ import annotations

import importlib
import importlib.util
import pathlib
import re
import sys

import pytest

from tools.import_sweep import (
    OPTIONAL_MODULES,
    SCRIPT_MODULES,
    main,
    sweep,
    tolerated,
)

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]

#: A top-level name no distribution provides, registered in
#: ``OPTIONAL_MODULES`` for the length of one test.  The synthetic fixture
#: below used to ``import cupy``, which made its assertion a statement about
#: the MACHINE rather than about the sweep: on a box that HAS cupy -- which,
#: for a GPU-native project, is most developer boxes -- the fixture module
#: imported cleanly, ``absences`` came back empty and the negative control
#: failed for a reason with nothing to do with the code under test.  A gate
#: whose verdict depends on what happens to be installed is the shape this
#: whole file exists to refuse, so the absence is made a certainty instead.
ABSENT_EXTRA = "sweepfixture_extra_no_distribution_provides"


def test_a_declared_extra_is_tolerated_and_nothing_else_is() -> None:
    """The classifier, in both directions, one shape per line.

    ``netCDF4`` is the essential negative: it is a BASE dependency
    (``pyproject.toml``'s ``dependencies``), so a runner that cannot import it
    has an environment that does not match what the distribution declares,
    and that is a red rather than a shrug.
    """

    assert tolerated(ModuleNotFoundError("no cupy", name="cupy"))
    assert tolerated(ModuleNotFoundError("no cupyx", name="cupyx"))
    assert tolerated(ModuleNotFoundError("broken cupyx", name="cupyx.scipy")) is None
    assert tolerated(ModuleNotFoundError("no rasterio", name="rasterio")) == \
        OPTIONAL_MODULES["rasterio"]

    assert tolerated(ModuleNotFoundError("no netCDF4", name="netCDF4")) is None
    assert tolerated(
        ModuleNotFoundError("no woof.core.nope", name="woof.core.nope")) \
        is None
    assert tolerated(ImportError("cannot import name 'X' from 'woof.core'")) \
        is None
    assert tolerated(ModuleNotFoundError("nameless")) is None
    assert tolerated(ValueError("a module-scope constant is wrong")) is None


def _synthetic_package(root: pathlib.Path, name: str) -> None:
    package = root / name
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "fine.py").write_text("VALUE = 1\n", encoding="utf-8")
    (package / "needs_an_extra.py").write_text(
        f"import {ABSENT_EXTRA}\n", encoding="utf-8")
    (package / "raises.py").write_text(
        "raise ValueError('a bad module-scope constant')\n", encoding="utf-8")
    (package / "broken_internal.py").write_text(
        f"from {name} import nothing_of_the_sort\n", encoding="utf-8")
    # The finder caches a directory listing per sys.path entry, and anything
    # that resolves a name before these files exist caches their absence.
    importlib.invalidate_caches()


def _declare_the_absent_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register :data:`ABSENT_EXTRA` as an extra, for one test."""

    assert importlib.util.find_spec(ABSENT_EXTRA) is None, (
        f"{ABSENT_EXTRA} resolves to something on this machine, so the "
        "fixture below is no longer a certain absence; rename it")
    monkeypatch.setitem(OPTIONAL_MODULES, ABSENT_EXTRA, "recast-woof[nothing]")


def test_the_sweep_names_every_module_that_does_not_import(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """THE NEGATIVE CONTROL.  Three faults, one shrug, and they are sorted.

    ``broken_internal`` is the one that matters: it raises
    ``ModuleNotFoundError`` exactly like the missing extra does, and only the
    name distinguishes them.  A sweep that tolerated the class rather than the
    name would call this package healthy.
    """

    monkeypatch.syspath_prepend(str(tmp_path))
    _declare_the_absent_extra(monkeypatch)
    _synthetic_package(tmp_path, "sweepfixture")
    for module in [name for name in sys.modules
                   if name.startswith("sweepfixture")]:
        monkeypatch.delitem(sys.modules, module, raising=False)

    walked, absences, failures = sweep(("sweepfixture",))

    assert "sweepfixture.fine" in walked
    assert sorted(absences) == [ABSENT_EXTRA], absences
    assert absences[ABSENT_EXTRA] == ["sweepfixture.needs_an_extra"]

    named = sorted(failure.module for failure in failures)
    assert named == ["sweepfixture.broken_internal", "sweepfixture.raises"], (
        "the sweep must fail on a module that raises at import AND on one "
        f"whose internal import does not resolve; it reported {named}")
    detail = " ".join(str(failure) for failure in failures)
    assert "ValueError" in detail and "nothing_of_the_sort" in detail, detail


def _package_of(root: pathlib.Path, name: str, **modules: str) -> None:
    """A one-off package whose modules are exactly ``modules``."""

    package = root / name
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    for leaf, body in modules.items():
        (package / f"{leaf}.py").write_text(body, encoding="utf-8")
    importlib.invalidate_caches()


def test_a_module_that_exits_at_import_is_a_failure_not_the_end_of_the_sweep(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """THE GATE MUST NOT BE KILLABLE BY THE THING IT IS MEASURING.

    The walk caught ``Exception``, and ``SystemExit`` is not one.  Measured on
    a box with a visible CUDA device: ``python -m tools.import_sweep`` exited
    2 and printed NO REPORT AT ALL, because ``tilestream/attack_real12gb.py``
    raises ``SystemExit(2)`` at import scope when the card is not idle and the
    sweep died inside it.  Every module after it went unmeasured and the lane
    saw an exit status with no verdict attached -- the audit's own defect
    class, one level up: a gate that stopped measuring and said nothing.

    ``z_after`` is the essential half.  A sweep that merely recorded the
    exit and then stopped would still name ``a_exits``.
    """

    monkeypatch.syspath_prepend(str(tmp_path))
    _package_of(tmp_path, "sweepexits",
                a_exits="raise SystemExit(2)\n",
                z_after="VALUE = 1\n")
    for module in [name for name in sys.modules
                   if name.startswith("sweepexits")]:
        monkeypatch.delitem(sys.modules, module, raising=False)

    walked, absences, failures = sweep(("sweepexits",))

    assert "sweepexits.z_after" in walked, walked
    assert not absences, absences
    named = [failure.module for failure in failures]
    assert named == ["sweepexits.a_exits"], named
    assert "SystemExit" in str(failures[0]), str(failures[0])


def test_a_keyboard_interrupt_still_stops_the_sweep(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The other direction, and the reason the catch is a named pair.

    This does not fail on the code the test above fixes -- it fails on the
    cheap version of that fix.  ``except BaseException`` would make the sweep
    swallow Ctrl-C, record the interrupted module as broken and carry on
    through the remaining 656, which is a gate that cannot be stopped and a
    verdict nobody asked for.
    """

    monkeypatch.syspath_prepend(str(tmp_path))
    _package_of(tmp_path, "sweepinterrupt",
                a_interrupt="raise KeyboardInterrupt\n",
                z_after="VALUE = 1\n")
    for module in [name for name in sys.modules
                   if name.startswith("sweepinterrupt")]:
        monkeypatch.delitem(sys.modules, module, raising=False)

    with pytest.raises(KeyboardInterrupt):
        sweep(("sweepinterrupt",))


def test_the_exit_code_is_the_verdict(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture) -> None:
    """A gate whose exit code does not follow its finding is not a gate."""

    monkeypatch.syspath_prepend(str(tmp_path))
    _declare_the_absent_extra(monkeypatch)
    _synthetic_package(tmp_path, "sweepverdict")
    healthy = tmp_path / "sweephealthy"
    healthy.mkdir()
    (healthy / "__init__.py").write_text("", encoding="utf-8")
    (healthy / "fine.py").write_text("VALUE = 1\n", encoding="utf-8")
    importlib.invalidate_caches()
    for module in [name for name in sys.modules
                   if name.startswith(("sweepverdict", "sweephealthy"))]:
        monkeypatch.delitem(sys.modules, module, raising=False)

    assert main(["sweepverdict"]) == 1
    assert "IMPORT SWEEP FAILED" in capsys.readouterr().out
    assert main(["sweephealthy"]) == 0
    assert "import sweep: OK" in capsys.readouterr().out


def test_a_declared_script_is_walked_but_not_imported_and_is_named(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture) -> None:
    """A module the sweep leaves out has to be VISIBLE in the sweep's report.

    ``SCRIPT_MODULES`` is an exclusion, and an exclusion nobody can read is
    the fault the whole audit is about.  So both halves are asserted: the
    module is not imported (its body would have ended the run), and the
    report says which module was skipped and why.
    """

    monkeypatch.syspath_prepend(str(tmp_path))
    _package_of(tmp_path, "sweepscript",
                program="raise SystemExit('the module body is the program')\n",
                fine="VALUE = 1\n")
    monkeypatch.setitem(SCRIPT_MODULES, "sweepscript.program",
                        "a synthetic program")
    for module in [name for name in sys.modules
                   if name.startswith("sweepscript")]:
        monkeypatch.delitem(sys.modules, module, raising=False)

    walked, _absences, failures = sweep(("sweepscript",))

    assert "sweepscript.program" in walked, walked
    assert not failures, [str(failure) for failure in failures]
    assert "sweepscript.program" not in sys.modules

    assert main(["sweepscript"]) == 0
    out = capsys.readouterr().out
    assert "not imported" in out and "sweepscript.program" in out, out
    assert "a synthetic program" in out, out


def test_no_shipped_module_imports_a_declared_script() -> None:
    """The exclusion may not cost the sweep any transitive coverage.

    Leaving a module unimported is only free while nothing the sweep DOES
    import depends on it.  The day a library module grows
    ``from package import script_only``, the table stops being a list of
    programs and becomes a hole in the middle of the package -- and this is
    the line that says so.
    """

    leaves = {module.rsplit(".", 1)[1]: module for module in SCRIPT_MODULES}
    offenders = []
    for package in ("woof", "tilestream"):
        for path in sorted((REPOSITORY_ROOT / package).rglob("*.py")):
            relative = path.relative_to(REPOSITORY_ROOT).as_posix()
            if relative[:-len(".py")].replace("/", ".") in SCRIPT_MODULES:
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                # (WOOF 1.1.7) an import statement, not a comment or docstring that says "import"
                if not re.match(r"\s*(?:from\s+[\w.]+\s+)?import\s", line):
                    continue
                for leaf, module in leaves.items():
                    if re.search(rf"\b{leaf}\b", line):
                        offenders.append(f"{relative}: {line.strip()}"
                                         f"  ({module})")
    assert not offenders, offenders


def test_every_declared_script_is_a_file_this_repository_ships() -> None:
    """A table entry for a module that no longer exists excludes nothing."""

    for module in SCRIPT_MODULES:
        path = REPOSITORY_ROOT / (module.replace(".", "/") + ".py")
        assert path.is_file(), (
            f"{module} is in tools/import_sweep.py::SCRIPT_MODULES and there is "
            f"no {path.relative_to(REPOSITORY_ROOT)}; a stale exclusion reads "
            "like a considered one")


def test_the_sweep_reaches_the_data_assimilation_stack() -> None:
    """Against the real package, on the subsystem the audit measured as unrun.

    ``woof/da/`` is 16,806 lines with zero entries on either GPU shard and
    none of its device suites on any lane (xc-06-02).  It is also CPU-
    importable, which is the whole point: the cheapest gate in the repository
    can cover it, and until this sweep existed nothing did.
    """

    walked, _absences, failures = sweep(("woof.da",))

    # 3.0.3 removed the local data assimilation; what stays is the analysis
    # door, IAU, radar heating, perturbations, products and operators.
    assert "woof.da.analysis_door" in walked, sorted(walked)
    assert "woof.da.letkf" not in walked, sorted(walked)
    assert len(walked) >= 15, walked
    assert not failures, [str(failure) for failure in failures]


def test_a_nested_package_exit_cannot_abort_enumeration(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    _package_of(tmp_path, "nestedexit", z_after="VALUE = 1\n")
    nested = tmp_path / "nestedexit" / "a_exits"
    nested.mkdir()
    (nested / "__init__.py").write_text("raise SystemExit(7)\n", encoding="utf-8")
    (nested / "child.py").write_text("VALUE = 2\n", encoding="utf-8")
    importlib.invalidate_caches()
    walked, absences, failures = sweep(("nestedexit",))
    assert "nestedexit.z_after" in walked and "nestedexit.a_exits.child" in walked
    assert {failure.module for failure in failures} == {
        "nestedexit.a_exits", "nestedexit.a_exits.child"}
    assert all("SystemExit" in str(failure) for failure in failures)
    assert not absences


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_linux_probe_is_importable_but_refuses_non_linux_execution(monkeypatch, capsys, platform):
    from tilestream import linux_probe
    monkeypatch.setattr(linux_probe.sys, "platform", platform)
    monkeypatch.setattr(linux_probe.sys, "argv", ["linux_probe"])
    with pytest.raises(SystemExit) as caught:
        linux_probe.main()
    assert caught.value.code == 2
    assert "requires Linux" in capsys.readouterr().err
    with pytest.raises(RuntimeError, match="requires Linux"):
        linux_probe.container_evidence()
