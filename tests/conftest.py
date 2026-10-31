"""Shared fixtures, and the guarantee that ``-m "not gpu"`` touches no device.

The GPU exclusion used to rest on every author remembering
``@pytest.mark.gpu``.  That failed silently and expensively: five Noah-MP CUDA
modules gated only on ``pytest.importorskip("cupy")`` and carried no marker, so
``-m "not gpu"`` *collected* them -- an unmarked test is not excluded by
``not gpu`` -- and 34 CUDA gates compiled and ran on a machine whose owner had
asked that no GPU work run there at all.  A convention that fails open is not a
convention; it is a hope.

Three things close it, in increasing order of strength:

* every test whose module imports cupy is marked ``gpu`` **automatically**, so
  the exclusion no longer depends on anyone remembering;
* ``GPUWM_NO_LOCAL_GPU=1`` skips those tests outright and stops this file
  importing cupy at all, so no device is opened even to ask whether one exists;
* ``tests/test_gpu_marker_discipline.py`` fails if any cupy-importing module
  would survive ``-m "not gpu"``.

Detection is by AST over the module's own source, not by inspecting
``sys.modules``: a module that imports cupy inside a function still needs the
marker, and reading the source cannot itself trigger an import.
"""

import ast
import functools
import os
import pathlib
import re

# Kernel numerics for the battery (owner ruling 2026-10-16: fast is the default, strict the opt-in).  The
# column oracles grade units word for word against WRF, so the battery names strict before woof loads;
# tests of the fast default select it themselves (tests/test_numerics_default.py).
os.environ["GPUWM_NUMERICS"] = "strict"

import pytest

#: Set to 1 to guarantee no local device is opened, whatever is collected.
#: The rented-GPU workflow leaves this set on the user's own machine.
NO_LOCAL_GPU = os.environ.get("GPUWM_NO_LOCAL_GPU", "") not in ("", "0")

if NO_LOCAL_GPU:
    # RUNTIME BACKSTOP, because source inspection is not a guarantee.  The
    # AST detector below marks tests whose *own* source imports cupy, but a
    # test can reach the device through an intermediary --
    # ``test_multigpu_forced_gpu.py`` did exactly that with a lazy
    # ``from tilestream import multigpu`` inside the test body, dodged the
    # marker automation, and RAN ON THE LOCAL CARD during a mandated
    # CPU-only invocation.  No enumeration of intermediaries can close that,
    # so the guarantee is planted where every route converges: the CUDA
    # runtime reads this variable at initialisation, and "-1" is an invalid
    # ordinal that leaves NOTHING visible.  Any escaped test's first device
    # use then fails loudly (cudaErrorNoDevice) instead of silently running
    # on the owner's card, whatever its import style, and subprocesses
    # inherit the ban.
    #
    # An import-level ban was tried first and rejected: merely importing
    # cupy is NOT the crime (module-scope ``import cupy`` sits under swaths
    # of legitimate CPU coverage, and zarr pulls the full package into every
    # pytest process before any conftest runs) -- opening the device is.
    # ``tests/test_gpu_marker_discipline.py`` pins this backstop
    # red-on-revert by asserting the banned process really sees zero
    # devices.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def has_gpu() -> bool:
    """True when a usable CUDA device is present.

    Under ``GPUWM_NO_LOCAL_GPU`` this answers False *without importing cupy*.
    Importing it and calling ``getDeviceCount()`` is itself device contact, and
    it used to happen on every single pytest invocation, including runs that
    had explicitly excluded the GPU.
    """
    if NO_LOCAL_GPU:
        return False
    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        return False


HAS_GPU = has_gpu()
requires_gpu = pytest.mark.skipif(not HAS_GPU, reason="no CUDA GPU / cupy")


def _is_cupy_import(node: ast.AST) -> bool:
    """Whether this single node imports cupy, however it is spelled."""
    if isinstance(node, ast.Import):
        return any(a.name.split(".")[0] == "cupy" for a in node.names)
    if isinstance(node, ast.ImportFrom):
        return (node.module or "").split(".")[0] == "cupy"
    # pytest.importorskip("cupy") is an import in every sense that matters.
    if isinstance(node, ast.Call):
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(
            func, "id", "")
        if name == "importorskip" and node.args:
            first = node.args[0]
            return isinstance(first, ast.Constant) and first.value == "cupy"
    return False


def _is_fixture(node: ast.AST) -> bool:
    for deco in getattr(node, "decorator_list", []):
        target = deco.func if isinstance(deco, ast.Call) else deco
        name = target.attr if isinstance(target, ast.Attribute) else getattr(
            target, "id", "")
        if name == "fixture":
            return True
    return False


@functools.lru_cache(maxsize=None)
def _cupy_scope(path: str) -> tuple[bool, frozenset[str]]:
    """Which parts of a module open a CUDA device.

    Returns ``(whole_module, {function names})``.

    Granularity is the whole point.  A first version answered only "does this
    file mention cupy anywhere", which marked ``tests/test_preflight.py``
    entirely ``gpu`` on the strength of **one** import at ``:1548`` -- and that
    module's other ~200 tests deliberately *stub* cupy to exercise CPU paths
    ("no device touched", says its own comment).  Skipping them to protect one
    test removed the VRAM preflight's only automated evidence, which is a
    correctness bar on this hardware.  So:

    * module-level import -> the whole module, because import happens at
      collection and every test in it pays;
    * inside a fixture -> the whole module, because which tests request that
      fixture is not decidable from the AST alone, and over-marking is the
      safe direction;
    * inside one test function -> that function only.
    """
    try:
        tree = ast.parse(pathlib.Path(path).read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return False, frozenset()

    functions: set[str] = set()
    defs = [n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    owned = {id(inner) for fn in defs for inner in ast.walk(fn)}

    for node in ast.walk(tree):
        if not _is_cupy_import(node):
            continue
        if id(node) not in owned:
            return True, frozenset()          # module scope: everything pays

    for fn in defs:
        if not any(_is_cupy_import(n) for n in ast.walk(fn)):
            continue
        # A fixture, or any non-test helper, has callers this cannot see.  Its
        # device use belongs to whoever calls it, so the only safe answer is
        # the whole module.  Narrowing to the helper's own name would be worse
        # than the coarse rule: the helper is not a collected test, so nothing
        # would ever be marked and the leak would reopen silently.
        if _is_fixture(fn) or not fn.name.startswith("test_"):
            return True, frozenset()
        functions.add(fn.name)

    return False, frozenset(functions)


def _imports_cupy(path: str) -> bool:
    """Whether any part of this module opens a CUDA device."""
    whole, functions = _cupy_scope(path)
    return whole or bool(functions)


# --------------------------------------------------------------------------
# the second property: cupy INSTALLED, which is not the same as a device
# --------------------------------------------------------------------------
#
# The marker above is about OPENING A DEVICE, and this file's own reasoning
# says why merely importing cupy is not the crime.  There is a separate
# question it does not answer, and until 2026-09-17 nothing did: can this
# selection even be COLLECTED on an install that has no cupy at all?
#
# MEASURED on a development machine in a venv built from the runtime dependencies with no
# cupy extra: ``-m "not gpu and not slow and not network"`` over the stage-1
# and always lists ends in ``Interrupted: 8 errors during collection`` and
# runs NOTHING.  Six of those eight are cupy: a test module imports a
# first-party module which imports cupy UNGUARDED at its own module scope,
# so the import fails at collection and takes the session with it.  Seven
# further tests are collected and fail with ModuleNotFoundError, five of
# them through a route no source scan can see (the command line, then
# ``woof.core.clock.resolve_clock``, then ``woof/core/physics.py:37``).
#
# THIS IS NOT FIXED BY MARKING THEM ``gpu``, and that was the first thing
# tried.  It would mark 156 tests across 59 files, none of which opens a
# device, and remove every one of them from the CPU legs on a properly
# provisioned box -- which is coverage loss wearing a safety costume, in
# this file's own words, and the tree's position is the opposite one:
# ``tools/battery/provision_battery_venv.ps1`` installs the ``all`` extra
# for the CPU legs and says in writing that "six stage-1 files import cupy
# at collection".  Six is exactly what was measured.  cupy is PROVISIONED
# for those legs by design.
#
# So the remedy matches the property.  When cupy is ABSENT, a module that
# cannot be imported without it is not collected, and a test whose body
# reaches such a module is skipped -- by name, counted, and printed in the
# terminal summary, never silently.  When cupy is PRESENT, which is every
# battery leg, nothing below changes anything at all: no item is skipped,
# no module is dropped, and every count stays what the census recorded.
# What changes is that an install without cupy now produces a CPU run with
# a named set of skips instead of a collection that dies.

#: Whether ``import cupy`` would succeed here.  Asked of the FINDER, so
#: nothing is imported and no device is contacted even to answer: the
#: question is about the install, not about a card.
def _cupy_installed() -> bool:
    import importlib.util
    try:
        return importlib.util.find_spec("cupy") is not None
    except Exception:
        return False


CUPY_INSTALLED = _cupy_installed()

#: Skip a test that needs cupy IMPORTABLE and opens no device.  The sibling
#: of ``requires_gpu`` above, and spelled the same way, because the tree
#: already says a skipif helper is how a missing optional dependency is
#: expressed.  It is deliberately NOT a marker: adding one to the ``-m``
#: vocabulary would change what every battery leg selects, and these tests
#: must keep running wherever cupy is installed.
requires_cupy = pytest.mark.skipif(
    not CUPY_INSTALLED,
    reason="needs cupy importable (opens no device); this install has none")

#: The package roots whose import edges are followed.
_FIRST_PARTY = ("woof", "tilestream", "tools")
_TREE_ROOT = pathlib.Path(__file__).resolve().parents[1]


@functools.lru_cache(maxsize=None)
def _first_party_file(dotted: str) -> str | None:
    """The source file of a first-party module name, or None."""
    base = _TREE_ROOT.joinpath(*dotted.split("."))
    module = base.with_suffix(".py")
    if module.is_file():
        return str(module)
    package = base / "__init__.py"
    return str(package) if package.is_file() else None


def _first_party_targets(node: ast.AST) -> set[str]:
    """The first-party module names one import node binds."""
    names: set[str] = set()
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name.split(".")[0] in _FIRST_PARTY:
                names.add(alias.name)
    elif isinstance(node, ast.ImportFrom):
        module = node.module or ""
        if module.split(".")[0] in _FIRST_PARTY:
            names.add(module)
            # ``from woof.core import physics`` names a MODULE in its
            # alias list, not an attribute, and reading only the left half
            # missed exactly that spelling on the measured routes.
            for alias in node.names:
                names.add(module + "." + alias.name)
    return names


def _sibling_targets(node: ast.AST, path: str) -> set[str]:
    """The SIBLING test modules one import node binds, as absolute paths.

    ``tests/`` is not a package, so pytest's default ``prepend`` import mode
    puts the directory itself on ``sys.path`` and ``from test_noahmp_runtime
    import _build`` is an ordinary import of ``tests/test_noahmp_runtime.py``.
    The closure followed only woof, tilestream and tools, so three modules
    that reach cupy through a sibling (test_da_cycle_join_gpu,
    test_noahmp_cold_start_device, test_noahmp_device_wiring) were imported
    on a cupy-less install and ended the public CI's CPU job with
    ``Interrupted: 3 errors during collection`` on 2.7.6, 2.7.7 and 2.8.0.
    A file inside a package is not resolved this way, so only a directory
    without ``__init__.py`` contributes siblings.  The key is the file path,
    which no dotted first-party name can be mistaken for.
    """
    if isinstance(node, ast.Import):
        tops = {alias.name.split(".")[0] for alias in node.names}
    elif isinstance(node, ast.ImportFrom) and not node.level:
        tops = {(node.module or "").split(".")[0]}
    else:
        return set()
    folder = pathlib.Path(path).resolve().parent
    if (folder / "__init__.py").is_file():
        return set()
    found: set[str] = set()
    for top in tops:
        if not top or top in _FIRST_PARTY or top == "cupy":
            continue
        module = folder / (top + ".py")
        package = folder / top / "__init__.py"
        if module.is_file():
            found.add(str(module))
        elif package.is_file():
            found.add(str(package))
    return found


@functools.lru_cache(maxsize=None)
def _import_time_edges(path: str) -> tuple[frozenset[str], bool]:
    """What runs when this file is imported: ``(first-party, cupy)``.

    Module scope only, because that is what executes at import, and
    UNGUARDED only.  A ``try: import cupy`` is not a card dependence and
    must not be read as one: ``woof/core/state.py`` guards its import
    deliberately, with its own record of why -- an absent or unloadable
    cupy used to kill the whole command line through
    ``cli -> downscale -> offline_child -> here``, and ``run-plan
    --probe``, whose job is to diagnose exactly that install, could not
    run on it.  Reading a guarded import as a dependence would mark most
    of this tree and would be measurably wrong.
    """
    try:
        tree = ast.parse(pathlib.Path(path).read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return frozenset(), False
    imports: set[str] = set()
    cupy = False
    # Visit only the same unguarded import-time scope as before. Walking
    # every function subtree to build exclusion sets made a small CPU-only
    # collection traverse millions of irrelevant nodes. In particular, a
    # try's handlers/else/finally still count; only its body is guarded.
    pending = [tree]
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if isinstance(node, ast.Try):
            pending.extend(node.handlers)
            pending.extend(node.orelse)
            pending.extend(node.finalbody)
            continue
        pending.extend(ast.iter_child_nodes(node))
        if isinstance(node, ast.Import):
            if any(a.name.split(".")[0] == "cupy" for a in node.names):
                cupy = True
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] == "cupy":
                cupy = True
        else:
            continue
        imports |= _first_party_targets(node)
        imports |= _sibling_targets(node, path)
    return frozenset(imports), cupy


#: Memo for the closure below.  A plain dict rather than lru_cache because
#: the recursion carries a cycle stack that must not become part of the key.
_IMPORT_CLOSURE: dict[str, tuple[str, ...] | None] = {}


def _cupy_import_chain(dotted: str,
                       stack: tuple[str, ...] = ()) -> tuple[str, ...] | None:
    """The import chain by which this module needs cupy, or None.

    One indirection or ten: the answer is the same shape, and the chain is
    returned rather than a boolean so a skip can say which edge did it.
    """
    if dotted in _IMPORT_CLOSURE:
        return _IMPORT_CLOSURE[dotted]
    if dotted in stack:
        return None                      # a cycle proves nothing by itself
    # A sibling test module arrives as its own file path (see
    # _sibling_targets) and is named in the chain by its module name.
    sibling = dotted.endswith(".py")
    path = dotted if sibling else _first_party_file(dotted)
    if path is None:
        _IMPORT_CLOSURE[dotted] = None
        return None
    label = dotted
    if sibling:
        located = pathlib.Path(dotted)
        label = (located.parent.name if located.name == "__init__.py"
                 else located.stem)
    imports, cupy = _import_time_edges(path)
    if cupy:
        _IMPORT_CLOSURE[dotted] = (label,)
        return _IMPORT_CLOSURE[dotted]
    answer = None
    for module in sorted(imports):
        found = _cupy_import_chain(module, stack + (dotted,))
        if found:
            answer = (label,) + found
            break
    _IMPORT_CLOSURE[dotted] = answer
    return answer


@functools.lru_cache(maxsize=None)
def _cupy_install_scope(path: str) -> tuple[str | None, frozenset[str]]:
    """Which parts of a test module cannot run without cupy INSTALLED.

    Returns ``(whole-module reason, {function name: reason})``.  The
    whole-module answer is what decides collection, because a module-scope
    edge fails during import and no marker can rescue it.
    """
    try:
        tree = ast.parse(pathlib.Path(path).read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None, frozenset()
    imports, cupy = _import_time_edges(path)
    if cupy:
        return "imports cupy at module scope", frozenset()
    for module in sorted(imports):
        chain = _cupy_import_chain(module)
        if chain:
            return "imports " + " -> ".join(chain), frozenset()
    reasons: dict[str, str] = {}
    for fn in (node for node in ast.walk(tree)
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))):
        if not fn.name.startswith("test_"):
            continue
        for node in ast.walk(fn):
            for module in sorted(_first_party_targets(node)):
                chain = _cupy_import_chain(module)
                if chain:
                    reasons[fn.name] = "imports " + " -> ".join(chain)
                    break
            if fn.name in reasons:
                break
    return None, frozenset(f"{name}: {why}" for name, why in reasons.items())


#: Modules not collected for want of cupy, printed in the terminal summary.
#: A dropped module that nothing announces is the silent deselection this
#: repository has a whole plugin about.
_UNCOLLECTED_WITHOUT_CUPY: list[str] = []

#: Tests that reached for cupy at CALL time on an install without it.
_SKIPPED_AT_CALL_WITHOUT_CUPY: list[str] = []


#: The text every route prints for the one absence this file is about.
_CUPY_IS_ABSENT = "No module named 'cupy'"


def _is_the_missing_cupy(error: BaseException) -> str | None:
    """Why this failure is only the absent cupy, or None for everything else.

    MEASURED, on the cupy-less venv on a development machine: the absence arrives in three
    shapes, and a hook that knew only the first left twelve tests failing
    for it.

    * ``ModuleNotFoundError`` named ``cupy`` -- an engine function
      importing a device module when called.
    * ``ImportError`` re-raised around it.  ``monkeypatch.setattr(
      "woof.core.dycore.step", ...)`` resolves the dotted path through a
      loader that wraps the original: ``import error in woof.core.dycore:
      No module named 'cupy'``, which is not a ModuleNotFoundError at all.
    * ``woof.capabilities.CapabilityMissing`` -- the front door refusing
      ahead of the work, exactly as designed, which is not an import
      failure and never will be.

    Each arm is exact in the same way. The module named must BE cupy, the
    wrapped text must name cupy, and the refusal must name cupy; anything
    else, and every other exception of every other kind, returns None and
    is raised unchanged. The class is recognised by name and module rather
    than imported, so this file still imports no part of the package it is
    collecting.
    """
    if isinstance(error, ModuleNotFoundError):
        if (getattr(error, "name", None) or "").split(".")[0] == "cupy":
            return "imports cupy when it is called"
        return None
    if isinstance(error, ImportError):
        if _CUPY_IS_ABSENT in str(error):
            return ("reaches a module whose own import of cupy is re-raised "
                    "as an ImportError")
        return None
    if any(cls.__module__ == "woof.capabilities"
           and cls.__name__ == "CapabilityMissing"
           for cls in type(error).__mro__) and "cupy" in str(error):
        return "meets the command line's own refusal for the absent cupy"
    return None


def _skip_for_the_absent_cupy(item, error: BaseException) -> None:
    """Record and skip, or return so the caller re-raises untouched."""
    if CUPY_INSTALLED:
        return
    why = _is_the_missing_cupy(error)
    if why is None:
        return
    _SKIPPED_AT_CALL_WITHOUT_CUPY.append(item.nodeid)
    pytest.skip(f"no cupy in this install; this test opens no device but "
                f"{why}. The battery legs install cupy "
                "(tools/battery/provision_battery_venv.ps1); remedy here is "
                "pip install 'recast-woof[gpu-cu12]'")


@pytest.hookimpl(wrapper=True)
def pytest_runtest_setup(item):
    """The same answer during SETUP, where a fixture reaches the device.

    A module-scope or function-scope fixture that imports an engine module
    fails before the test body runs, and an error in setup is reported as
    an ERROR rather than a failure -- the shape ten tests in
    tests/test_domain_wizard_forcing.py took on this same venv for an
    unrelated reason. Same conditions as the call wrapper below, same
    exactness, same dead branch wherever cupy is installed.
    """
    try:
        return (yield)
    except BaseException as error:
        _skip_for_the_absent_cupy(item, error)
        raise


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    """The runtime half, which no source scan can reach.

    THE MEASURED REMAINDER.  With the import-time closure above in place,
    the cupy-less venv on a development machine still reported 214 failures, and 136 of
    them were ``ModuleNotFoundError: No module named 'cupy'`` raised
    DURING the test -- an engine function importing a device module when
    called, most often through the command line and
    ``woof.core.clock.resolve_clock`` into ``woof/core/physics.py:37``.
    No AST closure can see that route: whether a given test reaches a
    lazily imported module is a call-graph question, and this file already
    records why enumerating intermediaries does not work.

    So the answer is taken where the answer is: the exception itself.  The
    conditions are exact, and each one is what keeps this from hiding
    anything.

    * ``CUPY_INSTALLED`` is False.  On every provisioned box -- which is
      every battery leg -- this branch is dead and the failure is raised
      unchanged.
    * The exception is one of the three shapes
      :func:`_is_the_missing_cupy` recognises, each of which has to NAME
      cupy.  A different missing module, and any other failure of any
      kind, re-raises untouched.
    * The skip is recorded by node id and printed in the summary, so the
      count is visible rather than absorbed.

    This is not a test loosened to pass: on an install that can run these
    tests, nothing here runs at all.
    """
    try:
        return (yield)
    except BaseException as error:
        _skip_for_the_absent_cupy(item, error)
        raise


def _register_silent_deselection_guard(config):
    """Load tools/battery/no_silent_deselection BY PATH, on every run.

    THE BREAKAGE, measured 2026-08-28.  ``pytestmark = pytest.mark.gpu`` at
    the top of tests/test_ruc.py retired all sixty RUC bitwise-oracle tests --
    the suite that detects a ONE-ULP change to Stefan-Boltzmann -- and the leg
    reported rc=0 with 8 passed, 61 deselected.  Nothing saw it: not
    test_gpu_marker_discipline.py, not test_module_skip_placement.py, not
    test_stage1_manifest.py.  One line retires any suite in this repository.

    Registered here rather than left to the battery's command line because a
    guard that runs only when somebody remembers to pass ``-p`` is not a
    default, and the battery is driven from queue scripts that do not live in
    this repository.  It is loaded BY PATH rather than as ``tools.battery.*``
    because that import fails whenever pytest is run from a subdirectory --
    measured: ImportError, and it takes the whole session with it.

    Failure to load is reported and not fatal: this hook must never be the
    reason a test run cannot start.
    """

    import importlib.util
    import sys

    path = (pathlib.Path(__file__).resolve().parents[1]
            / "tools" / "battery" / "no_silent_deselection.py")
    if not path.is_file():
        print()
        print(f"no_silent_deselection guard NOT LOADED: {path} is "
              "missing; "
              "a whole suite can be retired by one marker line and this run "
              "would not see it")
        return
    spec = importlib.util.spec_from_file_location(
        "gpuwm_no_silent_deselection", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["gpuwm_no_silent_deselection"] = module
    try:
        spec.loader.exec_module(module)
        module.pytest_configure(config)
    except Exception as error:                       # pragma: no cover
        print()
        print(f"no_silent_deselection guard NOT LOADED: {error!r}")


def _register_silent_skip_guard(config):
    """Load the default skip guard by path; broken policy cannot read green."""
    import importlib.util
    import sys

    path = (pathlib.Path(__file__).resolve().parents[1]
            / "tools" / "battery" / "no_silent_skip.py")
    if not path.is_file():
        raise pytest.UsageError(f"required no_silent_skip guard is missing: {path}")
    spec = importlib.util.spec_from_file_location("gpuwm_no_silent_skip", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["gpuwm_no_silent_skip"] = module
    spec.loader.exec_module(module)
    module.pytest_configure(config)


def _tree_under_test():
    """Load ``tools/tree_under_test`` BY PATH, never by name.

    ``import tools.tree_under_test`` would be resolved by the same broken
    machinery the module exists to detect, and would happily hand back the
    OTHER checkout's copy -- a detector that answers from the tree it is
    supposed to be accusing.  So it is loaded from this file's own
    location, which is the only thing in the process that is certainly
    part of the tree pytest collected.
    """

    import importlib.util

    path = pathlib.Path(__file__).resolve().parents[1] / "tools" \
        / "tree_under_test.py"
    spec = importlib.util.spec_from_file_location(
        "_gpuwm_tree_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pytest_configure(config):
    # WHICH TREE IS THIS, before anything else has a chance to report a
    # verdict about it.  An editable install binds `woof` and `tools` to
    # the main checkout through a sys.meta_path finder, which answers
    # ahead of sys.path, so a lane worktree's suite silently imports the
    # main checkout and reports green about edits it never executed.
    # Found the hard way: a committed fix to tools/check_negation_invariant
    # .py failed its own tests in the worktree holding the fix, because
    # the worktree was running the unfixed copy from the main checkout.
    # UsageError rather than a warning -- a run that measured the wrong
    # tree has no result worth printing, and every previous lane on this
    # box was one variable away from believing one.
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    refusal = _tree_under_test().check(repo_root)
    if refusal is not None:
        raise pytest.UsageError(refusal)

    # AFTER the tree check and not before it: this guard reports on the
    # suite that is about to run, and reporting on a suite collected from
    # the wrong checkout is the thing the check above refuses.
    if not config.pluginmanager.hasplugin("no_silent_deselection_guard"):
        _register_silent_deselection_guard(config)

    # The other half of the same question.  The guard above asks whether a
    # listed file still CONTRIBUTES tests; this one asks whether the tests it
    # contributes still RUN, which is the half a skip walks straight through.
    if not config.pluginmanager.hasplugin("no_silent_skip_guard"):
        _register_silent_skip_guard(config)

    config.addinivalue_line(
        "markers",
        "slow_acceptance: multi-minute end-to-end acceptance runs; excluded "
        "during fix-round iteration (-m 'not slow_acceptance'), REQUIRED in "
        "the full suite before any task's final commit")
    config.addinivalue_line(
        "markers",
        "gpu: opens a CUDA device.  Applied automatically to every test whose "
        "module imports cupy -- do not rely on writing it by hand, and do not "
        "remove the automation to 'clean up' a redundant-looking marker.")
    config.addinivalue_line(
        "markers",
        "requires_capability(name): reads a staged Rust artifact; skipped, "
        "with the command that stages it, when the probe in tests/conftest.py "
        "CAPABILITY_PROBES finds this box cannot do it.  Spelled through the "
        "requires_* marks defined beside the probes.")
    config.addinivalue_line(
        "markers",
        "requires_case_inputs(config): loads a shipped case configuration with "
        "its declared inputs required; skipped, naming the file the case "
        "loader finds absent, on a machine that does not hold them.")


def _state_the_reason_to_the_deselection_guard(path, why: str) -> None:
    """Tell the silent-deselection guard why this file said nothing.

    That guard fails a leg when a file it was told to run contributes no
    tests, because ``pytestmark = pytest.mark.gpu`` at the top of
    tests/test_ruc.py once retired sixty bitwise-oracle tests and the leg
    stayed green.  A module dropped above contributes nothing either, so
    without this the guard turns every cupy-less run red -- which is the
    same wrong answer in the other direction: the run is told the file was
    silently retired when the file has just said, in the terminal summary
    and in its own skip reason, exactly why it cannot be imported here and
    what to install.

    NOT A WEAKENING, and the conditions are what make that true.  The entry
    is written only for a module THIS SESSION dropped, only on an install
    with no cupy at all, and only with the import chain that forced it.  On
    every provisioned battery leg nothing above ever runs, so the guard's
    table stays empty and the marker fault it was built for still fails the
    leg by name.  The guard's own contract for an entry is a claim about
    where the coverage lives; this one names the leg: the battery installs
    cupy, so it runs there.
    """
    import sys

    guard = sys.modules.get("gpuwm_no_silent_deselection")
    allowed = getattr(guard, "ZERO_COLLECT_ALLOWED", None)
    if allowed is None:
        return
    root = pathlib.Path(__file__).resolve().parents[1]
    try:
        relative = pathlib.Path(path).resolve().relative_to(root).as_posix()
    except (OSError, ValueError):
        return
    allowed[relative] = (
        f"no cupy in this install and this module {why}; its coverage runs "
        "on the battery legs, which install cupy "
        "(tools/battery/provision_battery_venv.ps1)")


class _ModuleNeedingCupy(pytest.Module):
    """A module reported as SKIPPED instead of imported and errored.

    The drop is RECORDED here and not where the node is made, measured:
    pytest builds a node for every file in the collected directory and
    then prunes to the paths the session asked for, so recording at node
    creation made ``pytest tests/test_config.py`` report 37 modules
    skipped in a run that had asked for one file and would never have
    imported any of them.  ``collect`` runs only for a node that survived
    the pruning, so the count is the number of modules this session
    really lost.
    """

    def collect(self):
        why = _SKIP_REASON.get(str(self.path), "needs cupy importable")
        _UNCOLLECTED_WITHOUT_CUPY.append(f"{self.path.name}: {why}")
        _state_the_reason_to_the_deselection_guard(self.path, why)
        pytest.skip(f"no cupy in this install; this module {why}. It opens "
                    "no device: the battery legs install cupy "
                    "(tools/battery/provision_battery_venv.ps1), remedy here "
                    "is pip install 'recast-woof[gpu-cu12]'",
                    allow_module_level=True)


#: Why each dropped module needs cupy, for the skip reason it reports.
_SKIP_REASON: dict[str, str] = {}


def pytest_pycollect_makemodule(module_path, parent):
    """Do not IMPORT a module that cannot be imported without cupy.

    Only when cupy is absent, and only for a module whose import-time
    closure reaches an unguarded ``import cupy``.  With cupy installed --
    which is every provisioned battery leg -- this returns None for
    everything and changes nothing whatsoever.

    A MARKER CANNOT DO THIS JOB: pytest imports a module to collect it, so
    the failure lands before any marker is consulted, and six modules
    ended the whole session with ``Interrupted: 8 errors during
    collection`` on the measured cupy-less venv -- nothing ran at all.

    And it is this hook rather than ``pytest_ignore_collect``, measured:
    a path named on the command line is collected without consulting the
    ignore hook, so the six errors survived that version of the fix, and
    the battery names every file on its list explicitly.  This hook is
    consulted for a walked directory and for an explicit argument alike,
    and it can answer SKIPPED, which an ignore cannot.
    """
    if CUPY_INSTALLED:
        return None
    whole, _functions = _cupy_install_scope(str(module_path))
    if whole is None:
        return None
    _SKIP_REASON[str(module_path)] = whole
    return _ModuleNeedingCupy.from_parent(parent, path=module_path)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Say what was dropped for want of cupy, every time it happens."""
    if _SKIPPED_AT_CALL_WITHOUT_CUPY:
        terminalreporter.write_sep(
            "=", f"{len(_SKIPPED_AT_CALL_WITHOUT_CUPY)} test(s) SKIPPED at "
                 "call: this install has no cupy")
        terminalreporter.write_line(
            "  They reach a device module through a runtime route no source "
            "scan can see. Each raised ModuleNotFoundError('cupy') and "
            "nothing else was absorbed.")
    if not _UNCOLLECTED_WITHOUT_CUPY:
        return
    terminalreporter.write_sep(
        "=", f"{len(set(_UNCOLLECTED_WITHOUT_CUPY))} module(s) SKIPPED whole: "
             "this install has no cupy")
    for row in sorted(set(_UNCOLLECTED_WITHOUT_CUPY)):
        terminalreporter.write_line("  " + row)
    terminalreporter.write_line(
        "  These open no device; they cannot be IMPORTED without cupy. "
        "The battery legs install it (tools/battery/provision_battery_venv"
        ".ps1), so this list is empty there; remedy here: pip install "
        "'recast-woof[gpu-cu12]'.")


def pytest_collection_modifyitems(config, items):
    """Mark every cupy-importing test ``gpu``, and skip them when banned.

    Marking is unconditional so that ``-m "not gpu"`` is accurate on any
    machine, with or without a device.

    The ban-skip applies to every item CARRYING the marker, not only to the
    items this hook marked.  The old form skipped exactly its own AST hits,
    so a hand-marked test whose device use is transitive (no cupy in its own
    source, a lazy ``from tilestream import multigpu`` in the body) was
    marked ``gpu`` yet NOT skipped, and ran on the local card during a
    CPU-only invocation.  Belt: marker implies skip.  Braces: the
    ``CUDA_VISIBLE_DEVICES=-1`` backstop planted at import above, for tests
    carrying no marker at all.
    """
    skip_local = pytest.mark.skip(
        reason="GPUWM_NO_LOCAL_GPU=1: GPU work belongs on the rented device")
    for item in items:
        path = getattr(item, "fspath", None)
        if path is None:
            continue
        whole, functions = _cupy_scope(str(path))
        detected = whole
        if not whole:
            # ``originalname`` is the undecorated name for parametrised items.
            name = getattr(item, "originalname", None) or item.name
            detected = name.split("[")[0] in functions
        if detected:
            item.add_marker(pytest.mark.gpu)
        if NO_LOCAL_GPU and (detected
                             or item.get_closest_marker("gpu") is not None):
            item.add_marker(skip_local)
        if not CUPY_INSTALLED:
            # The function-level half of the install question.  A test body
            # that imports a module which needs cupy fails at CALL time,
            # not at collection, so the module is collected and only this
            # item is skipped -- with the edge that did it in the reason.
            _whole, functions = _cupy_install_scope(str(path))
            if functions:
                name = getattr(item, "originalname", None) or item.name
                bare = name.split("[")[0]
                for row in functions:
                    if row.split(":", 1)[0] == bare:
                        item.add_marker(pytest.mark.skip(
                            reason="no cupy in this install; this test "
                                   "opens no device but " + row))
                        break
    _gate_on_capabilities(items)
    _gate_on_case_inputs(items)


@pytest.fixture
def classic_history_frames(monkeypatch):
    """History frames written and priced as classic CDF-2 (``--history-compression none``).

    For tests of the classic frame layout and of frame counting, whose expected bytes are
    classic frame bytes. The default (lossless compressed frames, priced at
    ``disk_budget.COMPRESSED_HISTORY_FRACTION``) is tested in
    tests/test_compressed_history_pricing.py.
    """
    monkeypatch.setenv("GPUWM_HISTORY_COMPRESSION", "none")


@pytest.fixture(autouse=True, scope="session")
def _isolated_fetch_lock_root(tmp_path_factory):
    """Keep every test's output locks out of the machine-wide lock root.

    ``woof.fetch_guard`` keys its lock files on the resolved output path
    and keeps them outside the output tree, which means a suite that
    fetches into a hundred ``tmp_path`` directories leaves a hundred
    dead lock files under ``%PROGRAMDATA%/woof/locks``.  Pointing the
    root at the session's own temp directory makes the suite hermetic
    and leaves the user's machine alone.  Tests that need their own root
    (the lock gates themselves) override the variable per test.
    """
    from woof import fetch_guard

    root = tmp_path_factory.mktemp("fetch-locks")
    previous = os.environ.get(fetch_guard.LOCK_ROOT_ENV)
    os.environ[fetch_guard.LOCK_ROOT_ENV] = str(root)
    yield root
    if previous is None:
        os.environ.pop(fetch_guard.LOCK_ROOT_ENV, None)
    else:
        os.environ[fetch_guard.LOCK_ROOT_ENV] = previous


@pytest.fixture(autouse=True, scope="session")
def _isolated_fetch_connection_slots(tmp_path_factory):
    """Keep the suite's fetch connection slots off the box-wide /tmp/gpuwm-fetch-connections.

    Pools size themselves from the slots free on the box
    (:func:`woof.fetch_connections.process_share`), so on a farm host where
    real fetches run the expected widths would depend on whoever else holds
    slots.  Tests that need their own slot directory set the variable per test.
    """
    from woof import fetch_connections

    root = tmp_path_factory.mktemp("fetch-connections")
    previous = os.environ.get(fetch_connections.DIRECTORY_ENV)
    os.environ[fetch_connections.DIRECTORY_ENV] = str(root)
    yield root
    if previous is None:
        os.environ.pop(fetch_connections.DIRECTORY_ENV, None)
    else:
        os.environ[fetch_connections.DIRECTORY_ENV] = previous


@pytest.fixture(autouse=True)
def _wizard_probe_pinned_to_a_24gib_card(monkeypatch):
    """Pin the ONE number the outside world moves in the domain wizard.

    With neither ``--card`` nor ``--vram-gib``, ``woof domain`` measures
    the local card through its probe seam and refuses when nothing is
    measurable.  Unpinned, every bare-wizard fixture emission in this
    suite (117 of them at the time of writing) would size against
    whatever card the box happens to hold -- or refuse outright on the
    CPU legs -- turning grid dimensions machine-dependent.  The pin is a
    24 GiB card with that tier's assumed free memory, so historical
    geometry fixtures keep their exact bytes. Constrained availability
    is exercised separately by the sizing-authority regressions.

    In-process invocations only; a test that drives the real CLI in a
    subprocess bypasses this and must declare its card (or pin its own
    probe).  The sizing-authority tests that exercise the measure and
    refuse paths re-monkeypatch this same seam with their own answers.
    """

    from woof import domain_wizard

    monkeypatch.setattr(
        domain_wizard, "device_memory_probe_subprocess",
        lambda **_kwargs: {"free_bytes": int(domain_wizard.card_assumed_free_gib(24) * 1024 ** 3),
                           "total_bytes": 24 * 1024 ** 3,
                           "profile": None})


@pytest.fixture(autouse=True)
def _multi_card_doors_pinned_to_a_free_threaded_interpreter(monkeypatch):
    """Pin the interpreter-lock answer every multi-card door refuses on.

    ``woof go``/``check``/``sim``/ensemble refuse ``--devices N>1`` on an
    interpreter whose lock is on (woof.free_threading.
    refuse_multi_card_under_gil).  This suite and its CI legs run on GIL
    CPython, and its [devices] tests exercise plumbing and admission, not
    stepping threads, so unpinned every one of them would stop at the
    refusal.  The refusal itself is tested in tests/test_free_threading.py,
    which restores the real answer.  In-process invocations only.
    """

    from woof import free_threading

    monkeypatch.setattr(free_threading, "multi_card_lock_reason", lambda: None)


@pytest.fixture(autouse=True)
def _hydro_door_tools_unprobed_outside_hydro_tests(request, monkeypatch):
    """Outside the hydro test modules, the default-on hydro door sees no builder.

    A natives env.sh exports WOOF_RW_HYDRO_WORLD and the other hydro provider
    tools, so every `woof go` a test drives (33 test files) reached the real
    door: its tool probe ran through the tests' stage-counting Popen doubles
    (test_a_failing_stage_replays_its_output_and_stops returned 2, not 3), and
    non-dry launches prepared a real continental fabric, its parameters and a
    90-day spin-up in the shared cache (test_the_memory_gate_warns_*,
    2026-10-21). Without the env the door finds no tools; this makes
    the env battery see the same. Hydro behaviour is tested in the modules
    named *hydro*, which keep the real probe (or pass their own probe_row).
    """

    module = getattr(request.node, "module", None)
    if "hydro" in getattr(module, "__name__", "hydro"):
        return
    from woof.hydro import door as hydro_door
    monkeypatch.setattr(hydro_door, "probe", lambda row: [
        "hydro provider tools not probed outside the hydro test modules"])


@pytest.fixture(autouse=True)
def _run_disk_pinned_to_ample_free_space(monkeypatch):
    """Pin the free disk run-plan's refusal before the download compares with.

    ``woof run-plan`` refuses a run whose projection (download,
    preparation, history, checkpoints and pictures) is larger than the
    free space on the run directory's disk.  The plans this suite executes
    stub every stage and write almost nothing, yet unpinned their verdict
    followed the host's disk: on a development machine with 12 GB free, 17 run-plan tests
    failed with "This run would write about 29.0 GiB" before reaching the
    stage they test.  The pin is a petabyte; the refusal itself is tested
    by the disk-budget tests, which pin their own free space.
    """

    from woof import disk_budget

    monkeypatch.setattr(disk_budget, "free_bytes", lambda path: 10 ** 15)


@pytest.fixture(autouse=True)
def _chain_static_staging_kept_off_the_network(monkeypatch):
    """Keep the chains' static-file staging from downloading in this suite.

    Since 2.8.7 each native chain stages the published static files its
    configuration reads (``stage_required_static_sources``) before its
    forcing fetch.  The chain tests here drive HRRR-cone configurations
    against a fake GEOG root with every stage stubbed, so unpinned they
    started the real 1.11 GB hrrr-conus-v4 download from NCO inside
    tests/test_preparation_assets_wif.py (measured 2026-10-06).  The pin
    stages nothing; no stubbed stage reads the file.  The staging itself is
    tested in tests/test_recipe_route_inputs.py, which imports the real
    function at module load and puts it back where it drives a chain.
    """

    from woof.static import external_source

    monkeypatch.setattr(external_source, "stage_required_static_sources",
                        lambda *args, **kwargs: [])


@pytest.fixture(autouse=True)
def _wif_climatology_download_kept_off_the_network(monkeypatch):
    """Keep mp=28's automatic WIF climatology download out of this suite.

    Since 2.9.7 an mp=28 preparation with no copy of
    QNWFA_QNIFA_SIGMA_MONTHLY.dat downloads the pinned 215 MiB dataset
    (woof.ingest.wif_climatology.acquire_wif_climatology) instead of
    falling back to the synthetic profile.  Here the download answers as an
    offline machine does, so the refusal path is what a test without a
    staged dataset sees.  The staging itself is tested through
    woof.table_assets.stage_wif_dataset directly.
    """

    from woof.ingest import wif_climatology

    monkeypatch.setattr(wif_climatology, "_download_wif_dataset",
                        lambda root: 2)


@pytest.fixture(autouse=True)
def _lake_ice_analysis_kept_off_the_network(monkeypatch, tmp_path_factory):
    """Keep the GFS route's observed lake ice from downloading in this suite.

    Since 2.9.1 a GFS preparation reads the daily IMS snow and ice analysis
    (woof.ingest.lake_ice) into ~/.woof/cache/surface-obs.  Pointed at an
    empty per-session cache and with no network answer, every preparation
    here falls back to the forcing's ice exactly as an offline run does,
    so fixtures recorded before the change keep their bytes.  The
    acquisition and decision are tested in tests/test_lake_ice.py.
    """

    from urllib.error import URLError

    from woof.ingest import lake_ice

    monkeypatch.setenv(lake_ice.CACHE_ENV,
                       str(tmp_path_factory.getbasetemp() / "surface-obs-empty"))
    original = lake_ice.acquire_observed_lake_ice

    def offline(valid_time, **kwargs):
        def refuse(request, timeout):
            raise URLError("network kept off in the test suite")
        kwargs.setdefault("opener", refuse)
        return original(valid_time, **kwargs)

    monkeypatch.setattr(lake_ice, "acquire_observed_lake_ice", offline)


@pytest.fixture(autouse=True)
def _the_default_geography_root_counts_as_staged(monkeypatch):
    """Answer the DEFAULT WPS_GEOG root as staged, on every test machine.

    Since 2.8.8 ``woof go --dry-run`` asks the same geography question the
    launch asks (it exited 0 on a tree the launch refused).  The dry-run
    tests here plan configurations against whatever geography the machine
    has at ``default_geog_root()``: none on public CI and a development machine, 16 GB on
    a workstation, so without this pin the same test passed or refused by
    host.  Only the default root is answered; a root a test NAMES
    (``--geog-root``, ``[case_data].geog_root``, a direct call) gets the
    real check, which is what the negative controls use
    (tests/test_go_dry_run_geography.py,
    tests/test_interrupt_contract.py::test_the_geography_refusal_still_fires_when_it_should),
    and a test that patches ``geography_refusal`` itself still wins.
    """

    from woof import go_cli
    from woof.geog_assets import default_geog_root

    real = go_cli.geography_refusal

    def refusal(geog_root, **kwargs):
        if pathlib.Path(geog_root) == default_geog_root():
            return None
        return real(geog_root, **kwargs)

    monkeypatch.setattr(go_cli, "geography_refusal", refusal)


@pytest.fixture
def pinned_thompson_tables(monkeypatch):
    """Resolve the mp8 tables to the checkout's pinned set when it has one.

    A non-editable recast-woof-data carries neither externalized Thompson table
    (qr_acr_qg_V4.dat, freezeH2O.dat), so on a development tree whose
    machine never ran ``woof fetch-tables`` every test that binds the
    Thompson table authority stopped in MissingTableAssets before the gate
    it asserts on, and its result depended on the box running it.  When
    the checkout's own recast-woof-data directory holds the complete pinned set,
    this stages it as the user root (the mechanism
    tests/test_prepared_domain_tree_forecast.py's missing-table control
    uses) and clears the override so no other root answers first.

    Only a complete checkout set is staged.  RELEASE-EXCLUDE.txt drops
    freezeH2O.dat from the public tree (GitHub caps a blob at 100 MiB),
    and the public developer install stages the whole set with ``woof
    fetch-tables`` under ~/.woof/tables/thompson instead.  Staging that
    checkout's short directory would hide the machine's complete set and
    turn 28 passing tests into MissingTableAssets refusals, so there the
    fixture changes nothing and returns None: the machine's own ladder
    (override, complete packaged root, staged root) answers as it does
    for a user.  Either way the binding still checks every asset's size
    and SHA-256 against the pins.  The checkout set is read in place and
    never written.  Not autouse: the tests of the resolver and of the
    missing-table refusals need the machine's own answer.
    """

    from woof import physics_compat

    checkout = (pathlib.Path(__file__).resolve().parents[1] / "recast-woof-data"
                / "woof_data" / "data" / "thompson" / "tables")
    if not physics_compat._table_root_is_complete(checkout):
        return None
    monkeypatch.delenv(physics_compat.THOMPSON_TABLE_ROOT_ENV, raising=False)
    monkeypatch.setattr(physics_compat, "user_thompson_table_root",
                        lambda: checkout)
    return checkout


def complete_runtime_manifest(payload: dict | None = None,
                              *, platform_name: str = "linux-x86_64",
                              **overrides) -> dict:
    """A sealed-runtime manifest that meets the WHOLE required schema.

    Fixtures used to declare only the keys the assertion under test
    happened to read, which is exactly how a two-key manifest reached a
    field user's preparation and died several minutes in, at a third
    consumer, on a key nobody had validated.  Now that
    :func:`woof.runtime_manifest.validate_manifest` is the one gate,
    a fixture that skips a field is testing a document no consumer will
    ever accept.  Build from here and override deliberately.
    """

    from woof import __version__
    from woof.native_wrf_distribution import distribution_contract
    from woof.runtime_manifest import RUNTIME_SCHEMA

    manifest = {
        "schema": RUNTIME_SCHEMA,
        "status": "READY",
        "artifact": {"name": "gpuwm-native-wrf",
                     "gpuwm_version": __version__},
        "source": {"commit": "0" * 40, "tree": "1" * 40,
                   "worktree_clean": True},
        "contract": distribution_contract(platform_name),
        "payload": payload if payload is not None else {
            "libexec/bridges/placeholder": {"bytes": 0, "sha256": "0" * 64},
        },
    }
    manifest.update(overrides)
    return manifest


def assert_gates(case: str, metrics: dict) -> None:
    """Assert every benchmark gate for ``case`` passes.

    Single-sourcing (Phase 2 Task 12): the gate intervals live in the case
    module's ``GATES`` export, and both the CLI and the benchmark tests
    consume them through ``woof.cli._failing_gate`` -- one table, one
    checker.  The failure message names the failing gate and dumps the
    metrics (list-valued time series dropped for readability).
    """
    from woof.cli import _failing_gate
    bad = _failing_gate(case, metrics)
    assert bad is None, (bad, {k: v for k, v in metrics.items()
                               if not isinstance(v, list)})


# ---------------------------------------------------------------------------
# STAGED RUST ARTIFACTS: one probe per capability, and a skip that says how
# to stage it.
#
# 110 cases in thirteen files read a staged Rust artifact and had no gate at
# all, so on a box where the artifact is absent -- or, worse, STALE -- they
# reported a red suite that says nothing about the tree.  Measured on the
# Linux CPU box (2026-09-11): 54 cases died on `rw_netcdf: Times is a String
# variable` from a bridge staged months ago, 25 on a CPU preprocessing
# library too old to export the symbol the call needs, 11 on a grib1 bridge
# that was never built, and 20 on the absent wrf-rust distribution.  A red
# for a missing tool is indistinguishable from a red for a defect, which is
# the whole reason to gate.
#
# The probes ask about the CAPABILITY, never merely about the file.  An
# artifact that is present and too old is the case that actually happened,
# and "the binary exists" answers it wrong.  Each reason names the command
# that stages the artifact, because a skip a reader cannot act on is a
# silence.
# ---------------------------------------------------------------------------

def _never_raises(probe):
    """A capability probe answers, or says why it could not answer.

    An exception out of a probe would take the session down with every
    unrelated test in it.  That happened while the probes still ran at
    import: ``WOOF_RW_NETCDF`` naming a path that no longer exists makes
    woof.netcdf_bridge.find_netcdf_bin raise FileNotFoundError
    deliberately, and a stale override then collected nothing at all.  A
    probe that cannot reach its artifact has found a gap, which is an
    answer; the text carries the error so the reader can act on it.
    """

    @functools.wraps(probe)
    def answer(*args):
        try:
            return probe(*args)
        except Exception as error:                      # noqa: BLE001
            detail = " ".join(str(error).split())[:200]
            named = f"{probe.__name__}{args!r}" if args else probe.__name__
            return (f"this capability could not be resolved here "
                    f"({named}: {detail}); fix or unset whatever names it "
                    "-- WOOF_RW_NETCDF and the staged estate under "
                    "~/.woof/bridges are the usual answers -- and stage "
                    "it with `python tools/stage_wheel_bridges.py`")

    return answer


@functools.lru_cache(maxsize=1)
@_never_raises
def netcdf_bridge_gap() -> str | None:
    """Why the staged ``rw_netcdf`` cannot read a WRF file, or None.

    A real round trip, not a version string: a NETCDF3_CLASSIC file with a
    WRF ``Times`` character array, written with netCDF4 and read back
    through the bridge.  That is what every gated case does first, and an
    older bridge reads the character array as a String variable and
    refuses.
    """

    import tempfile

    try:
        import netCDF4  # noqa: F401
    except Exception as error:                          # noqa: BLE001
        return f"netCDF4 is not importable here ({error})"
    from woof import netcdf_bridge

    if netcdf_bridge.find_netcdf_bin() is None:
        return ("rw_netcdf is not staged; build it with `cd tools/rustwx && "
                "cargo build --release -p rw-netcdf --offline` and stage it "
                "with `python tools/stage_wheel_bridges.py`")
    import numpy as _np

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "probe.nc")
        stamp = "2021-06-01_00:00:00"
        with netCDF4.Dataset(path, "w", format="NETCDF3_CLASSIC") as ds:
            ds.createDimension("Time", 1)
            ds.createDimension("DateStrLen", len(stamp))
            times = ds.createVariable("Times", "S1",
                                      ("Time", "DateStrLen"))
            times[0, :] = _np.asarray(list(stamp), dtype="S1")
        try:
            with netcdf_bridge.open_dataset(path) as dataset:
                _np.asarray(dataset.variables["Times"][...])
        except Exception as error:                      # noqa: BLE001
            return (f"the staged rw_netcdf cannot read a WRF Times array "
                    f"({error}); restage it with `cd tools/rustwx && cargo "
                    "build --release -p rw-netcdf --offline` followed by "
                    "`python tools/stage_wheel_bridges.py`")
    return None


@functools.lru_cache(maxsize=None)
@_never_raises
def cpu_preprocess_gap(symbol: str) -> str | None:
    """Why the staged CPU preprocessing library lacks ``symbol``, or None."""

    try:
        from woof.ingest.cpu_backend import CpuPreprocessBackend
        backend = CpuPreprocessBackend()
    except Exception as error:                          # noqa: BLE001
        return (f"the CPU preprocessing bridge is not usable here ({error}); "
                "build it with `cd tools/grib1_bridge && cargo build "
                "--release --locked --offline`")
    try:
        getattr(backend._library, symbol)
    except AttributeError:
        return (f"the staged CPU preprocessing bridge exports no {symbol}, "
                "so it predates this capability; rebuild it with `cd "
                "tools/grib1_bridge && cargo build --release --locked "
                "--offline` and restage with `python "
                "tools/stage_wheel_bridges.py`")
    return None


@functools.lru_cache(maxsize=1)
@_never_raises
def grib1_bridge_gap() -> str | None:
    """Why no usable grib1 bridge can be found here, or None.

    A probe FINDS; it never builds.  The module's own resolver,
    woof.ingest.grib.build_rust_bridge, runs ``cargo build`` inside any
    checkout before it looks anywhere else, and the first shape of this
    probe called it: every pytest collection on a box with cargo on PATH
    then compiled a crate before a single test ran, and wrote
    tools/grib1_bridge/target/ into trees that must not carry one (an
    exported release tree read by tests/test_release_snapshot_machine_paths
    .py went red with 316 machine paths from that directory alone).

    woof.bridges.find_bridge is the ladder without the build step, in
    the order the resolver itself uses once it stops building: the
    WOOF_GRIB1_BRIDGE override (a missing file it names is a gap, not a
    fall-through), the checkout's own target/release and target/debug,
    libexec beside the package, the wheel bundle, ~/.woof/bridges.  What
    it finds is then RUN: ``--era5-member-capabilities`` is the bridge's
    own no-input self-description, so a stale or broken executable
    answers here rather than inside the first gated case.
    """

    import subprocess

    from woof import bridges

    found = bridges.find_bridge("grib1_bridge")
    remedy = (" -- build it with `cd tools/grib1_bridge && cargo build "
              "--release --locked --offline` and stage it with `python "
              "tools/stage_wheel_bridges.py`")
    if found is None:
        return "no grib1 bridge executable is staged or built here" + remedy
    completed = subprocess.run(
        [str(found), "--era5-member-capabilities"], capture_output=True,
        text=True, timeout=60)
    if completed.returncode != 0 or '"schema"' not in completed.stdout:
        first = (completed.stderr or completed.stdout).strip().splitlines()
        detail = first[0] if first else f"exit status {completed.returncode}"
        return (f"the grib1 bridge at {found} does not answer its own "
                f"capability query ({detail})" + remedy)
    return None


@functools.lru_cache(maxsize=1)
@_never_raises
def wrf_rust_gap() -> str | None:
    """Why the mandated science core wrf-rust is unavailable, or None."""

    try:
        import wrf  # noqa: F401
    except Exception as error:                          # noqa: BLE001
        return (f"the mandated science core wrf-rust is not installed here "
                f"({error}); `pip install wrf-rust` into this interpreter")
    return None


#: capability name -> the probe that answers for it.  Resolved LAZILY, in
#: pytest_collection_modifyitems, and only for a capability some collected
#: item actually carries: a skipif evaluated at import ran every probe on
#: every collection, so `pytest tests/test_config.py` opened the NetCDF
#: bridge, loaded the CPU library and (see grib1_bridge_gap) built a crate,
#: for a file that reads none of them.
def netcdf_writer_gap() -> str | None:
    """Why the Rust NetCDF writer library cannot load here, or None."""

    try:
        from woof.io.nc_writer_bridge import resolve_ncwrite_bridge
        resolve_ncwrite_bridge()
    except Exception as error:                          # noqa: BLE001
        return f"the Rust NetCDF writer library is not usable here ({str(error).splitlines()[0]})"
    return None


CAPABILITY_PROBES = {
    "netcdf_bridge": netcdf_bridge_gap,
    # The ensemble product collector writes its NetCDF through this library.
    "netcdf_writer_bridge": netcdf_writer_gap,
    "wrf_eta_bridge": functools.partial(cpu_preprocess_gap,
                                       "gpuwm_wrf_eta_f32"),
    "wrf_sfcprs_bridge": functools.partial(cpu_preprocess_gap,
                                          "gpuwm_wrf_sfcprs3_from_f64"),
    # Every masked surface field (soil, snow, skin, sea ice) maps through
    # this entry under both backends; without it they cannot be mapped.
    "wps_masked_chain_bridge": functools.partial(
        cpu_preprocess_gap, "gpuwm_wps_masked_chain_f64"),
    # The native HRRR route's soil maps through this entry under both
    # backends; without it that route cannot map its soil.
    "masked_stencil_bridge": functools.partial(
        cpu_preprocess_gap, "gpuwm_masked_bilinear_stencil_f64"),
    # The lake skin search, the water-temperature blends, the labelling,
    # the water repairs and the source owner run through these entries
    # under both backends; without them no water temperature can be
    # assembled.  Probed by the newest entry, the CPU backend's
    # surface-nearest search, built after all of them.
    "water_blend_bridge": functools.partial(
        cpu_preprocess_gap, "gpuwm_masked_nearest_f32"),
    "wrf_rust": wrf_rust_gap,
    "grib1_bridge": grib1_bridge_gap,
}

CAPABILITY_MARKER = "requires_capability"

requires_netcdf_bridge = pytest.mark.requires_capability("netcdf_bridge")
requires_netcdf_writer_bridge = pytest.mark.requires_capability("netcdf_writer_bridge")
requires_wrf_eta_bridge = pytest.mark.requires_capability("wrf_eta_bridge")
requires_wrf_sfcprs_bridge = pytest.mark.requires_capability(
    "wrf_sfcprs_bridge")
requires_wrf_rust = pytest.mark.requires_capability("wrf_rust")
requires_wps_masked_chain_bridge = pytest.mark.requires_capability(
    "wps_masked_chain_bridge")
requires_masked_stencil_bridge = pytest.mark.requires_capability(
    "masked_stencil_bridge")
requires_water_blend_bridge = pytest.mark.requires_capability(
    "water_blend_bridge")
requires_grib1_bridge = pytest.mark.requires_capability("grib1_bridge")


def capability_gap(name: str) -> str | None:
    """The probe's verdict for ``name``, cached for the session."""

    return CAPABILITY_PROBES[name]()


def _gate_on_capabilities(items) -> None:
    """Skip every item whose declared capability this box lacks.

    Each probe runs at most once per process, and only if an item in
    this collection carries its mark; an item already carrying a skip is
    left alone, so the delegated copy of this hook that tilestream/
    conftest.py runs over the same items adds nothing a second time.
    """

    for item in items:
        for mark in item.iter_markers(name=CAPABILITY_MARKER):
            if item.get_closest_marker("skip") is not None:
                break
            name = mark.args[0] if mark.args else mark.kwargs.get("name")
            if name not in CAPABILITY_PROBES:
                raise pytest.UsageError(
                    f"{item.nodeid} requires an unknown capability "
                    f"{name!r}; known: {sorted(CAPABILITY_PROBES)}")
            gap = capability_gap(name)
            if gap is not None:
                item.add_marker(pytest.mark.skip(reason=gap))
                break


# ---------------------------------------------------------------------------
# Case inputs: the files a shipped case configuration declares
# ---------------------------------------------------------------------------

CASE_INPUTS_MARKER = "requires_case_inputs"

#: The case loader's refusal for a declared input that is not on this disk
#: (woof/case_data.py, build_case_data): "<role> file <path> declared in
#: [case_data] of <config> does not exist." and the geog_root directory form.
_CASE_INPUT_ABSENT = re.compile(
    r"declared in \[case_data\] of .* does not exist")


@functools.lru_cache(maxsize=None)
def case_inputs_absent(config: str) -> str | None:
    """The case loader's own missing-input refusal for ``config``, or None.

    A test that loads a shipped case configuration with its inputs required
    (the default of ``woof.case_data.load_experiment_case``) runs only on a
    machine holding every file that configuration's ``[case_data]`` declares,
    and some of those are third-party reference data that ship in no wheel.
    Without the data such a test raised the loader's ValueError instead of
    skipping, so a release install reported it as a failure.  The loader
    names the absent file; that sentence becomes the skip reason.  Any other
    refusal returns None, so the test runs and reports it: only a missing
    declared input is a property of the machine rather than of the tree.
    """
    from woof.case_data import load_experiment_case

    try:
        load_experiment_case(config)
    except ValueError as error:
        message = str(error)
        if _CASE_INPUT_ABSENT.search(message):
            return message
    except Exception:                                   # noqa: BLE001
        return None
    return None


def requires_case_inputs(config) -> pytest.MarkDecorator:
    """Mark a test that loads ``config`` with its declared inputs required."""
    return pytest.mark.requires_case_inputs(str(config))


def _gate_on_case_inputs(items) -> None:
    """Skip every item whose case configuration's inputs are not on disk.

    Resolved here rather than in a skipif so the loader runs once per
    configuration and only for configurations a collected item names.
    """
    for item in items:
        for mark in item.iter_markers(name=CASE_INPUTS_MARKER):
            if item.get_closest_marker("skip") is not None:
                break
            absent = case_inputs_absent(str(mark.args[0]))
            if absent is not None:
                item.add_marker(pytest.mark.skip(
                    reason=f"this machine does not hold the case inputs: "
                           f"{absent}"))
                break
