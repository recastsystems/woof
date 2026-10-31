"""What this install can do, and the exact command that closes each gap.

ONE registry, consulted by every front door, so that "you need the GPU
runtime" is one sentence with one remedy wherever a reader meets it.

The rule this module exists to enforce, which this program has paid for
twice: **a command refuses BEFORE it does expensive work** -- before any
network fetch, large disk write, or GPU allocation -- **with a named
refusal that states what is missing and the exact command that fixes
it.**  The shape is :mod:`woof.obs.dealias`'s front-door refusal, which
names its install line before decode and gridding rather than an hour
into a run.

Three properties are essential, and each has a test:

* **The remedy is DERIVED from what is actually missing**, never keyed
  on the exception's class name.  ``woof run-plan`` used to answer
  every ``ModuleNotFoundError`` with the CuPy remedy, so a missing
  ``scipy`` inside a plan run told the caller to install a GPU wheel.
  Here the missing MODULE selects the remedy, and a module this registry
  does not know gets no remedy at all -- absence is accurate, a wrong
  remedy is not.
* **Every remedy names something that EXISTS.**  Every extra spelled in
  this file is checked against the distribution's own metadata, and
  every ``woof`` command spelled here is checked against the parser.
  ``woof render --pair`` used to send a reader to ``recast-woof[render]`` for
  Pillow, which that extra does not contain.
* **The probe says what it PROVED.**  Presence is answered with
  :func:`importlib.util.find_spec`, which resolves the module without
  importing it: no CuPy import, no CUDA context, nothing claimed about
  whether the runtime then works.  "Installed" and "working" are two
  claims and only ``woof doctor`` makes the second one; a front door
  that conflated them would either refuse a working box or bless a
  broken one.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from woof import data_assets
from woof.explain import layered


class CapabilityMissing(ValueError):
    """A front door refused because this install cannot reach the feature.

    A ``ValueError`` on purpose: :func:`woof.cli.main`'s refusal
    boundary already prints a ``ValueError`` as one sentence at exit 2
    with ``--explain`` layering, so every ``woof`` subcommand inherits
    the named refusal by raising this rather than by growing a fourth
    copy of the handler.  The ``python -m`` doors, which have no such
    boundary, catch it by name.
    """

    def __init__(self, message: str, *, requirement: "Requirement",
                 command: str) -> None:
        super().__init__(message)
        #: What was missing, so a caller can report it structurally.
        self.requirement = requirement
        #: The front door that refused, for the ``--explain`` pointer.
        self.command = command


@dataclass(frozen=True)
class Requirement:
    """One importable module, what ships it, and the command that fixes it.

    ``module`` is the top-level import name -- the thing
    ``ModuleNotFoundError.name`` carries -- because that is the only
    identifier both a failed import and this registry can agree on.
    ``distribution`` is what ``pip`` calls it, which is frequently a
    different word (``wrf`` / ``wrf-rust``, ``shapefile`` / ``pyshp``),
    and naming both is what lets a reader search either one.
    """

    module: str
    distribution: str
    #: woof extras that install it.  Empty for a base dependency (or
    #: for one that arrives transitively) -- and empty is why the remedy
    #: is stored separately rather than formatted from the extras.
    extras: tuple[str, ...]
    #: One clause: what this install cannot reach without it.
    unlocks: str
    #: The action block, one command per line, in the house form:
    #: ``  remedy: <command>`` with alternatives as ``  # ...`` comments.
    #: A single line with a parenthesised second option is a shell error
    #: when pasted whole, which is how readers actually consume these.
    remedy: str

    @property
    def label(self) -> str:
        """``cupy (cupy-cuda12x)`` -- module first, since that is what failed."""

        if self.distribution == self.module:
            return self.module
        return f"{self.module} ({self.distribution})"


#: The GPU runtime's remedy, verbatim from the refusal ``woof domain``
#: and ``woof verify`` have printed since 1.8.8, moved here so that the
#: commands that used to relay a raw traceback print the SAME words.
#:
#: Neither extra leads.  CuPy ships one wheel per CUDA major, this
#: message cannot see which major the box serves, and the version that
#: led with the cu12 extra was read by CUDA-13 owners as a
#: recommendation.
GPU_RUNTIME_REMEDY = (
    "  # CuPy ships one wheel per CUDA major; pick yours:\n"
    "  remedy: pip install 'recast-woof[gpu-cu12]'\n"
    "  #   ... on a box whose CUDA is 13-only, instead:\n"
    "  #   pip install 'recast-woof[gpu-cu13]'\n"
    "  # `woof doctor` reads the major off the driver, names\n"
    "  # the matching extra, and checks the runtime estate.")

GPU_RUNTIME = Requirement(
    module="cupy",
    distribution="cupy-cuda12x / cupy-cuda13x",
    extras=("gpu-cu12", "gpu-cu13"),
    unlocks="every path that integrates the model on a card",
    remedy=GPU_RUNTIME_REMEDY)

#: The oldest CuPy the engine integrates on: the floor the ``gpu-cu12``
#: and ``gpu-cu13`` extras declare.  Breakage it prevents: the dycore's
#: bookkeeping launches pass by-value tables as ``numpy.void`` (since
#: 2.8.1), which CuPy 13 rejects, so a box still holding CuPy 13 passed
#: this front door and stopped at its first model step with "TypeError:
#: Unsupported type <class 'numpy.void'>" (a rented 8x RTX 5090 box,
#: 2026-10-02).
GPU_RUNTIME_MINIMUM = (14, 0)

_CUPY_VERSION_ASSIGNMENT = re.compile(r"""__version__\s*=\s*['"](\d+)\.(\d+)""")


def installed_cupy_version() -> tuple[int, int] | None:
    """``(major, minor)`` of the CuPy an import would load, without importing it.

    Read from ``cupy/_version.py`` beside the module ``find_spec``
    resolves (or from the loaded module when CuPy is already imported).
    ``None`` when CuPy does not resolve or its version cannot be read: an
    unreadable version names no breakage, so it is not refused.
    """

    module = sys.modules.get("cupy")
    loaded = getattr(module, "__version__", None) if module is not None else None
    if loaded is not None:
        match = re.match(r"(\d+)\.(\d+)", str(loaded))
    else:
        try:
            spec = importlib.util.find_spec("cupy")
        except (ImportError, ValueError):
            return None
        if spec is None or not spec.origin:
            return None
        try:
            text = (Path(spec.origin).parent / "_version.py").read_text(encoding="utf-8")
        except OSError:
            return None
        match = _CUPY_VERSION_ASSIGNMENT.search(text)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def outdated_gpu_runtime() -> tuple[int, int] | None:
    """The installed CuPy's ``(major, minor)`` when it is below the floor, else ``None``."""

    found = installed_cupy_version()
    if found is not None and found < GPU_RUNTIME_MINIMUM:
        return found
    return None


def gpu_runtime_floor_refusal(door: str, found: tuple[int, int], *,
                              before: str | None = None) -> str:
    """The refusal for a CuPy that resolves but is older than the floor."""

    floor = ".".join(str(part) for part in GPU_RUNTIME_MINIMUM)
    action = (f"{door}: this command needs CuPy {floor} or newer, and this "
              f"install has CuPy {found[0]}.{found[1]}.\n"
              "  What needs it: every model step; its bookkeeping launches "
              "pass by-value tables that CuPy 13 rejects (\"Unsupported "
              "type <class 'numpy.void'>\").\n")
    if before:
        action += f"  {before}\n"
    action += GPU_RUNTIME_REMEDY
    return layered(action, (
        "Checked at the front door by reading the version file beside the "
        "CuPy module an import would load, without importing it.  The "
        f"recast-woof[gpu-cu12] and recast-woof[gpu-cu13] extras require CuPy {floor} "
        "or newer, so installing either upgrades this one; a CuPy installed "
        "by hand or carried in an image can predate that floor."))

SCIENCE_CORE = Requirement(
    module="wrf",
    distribution="wrf-rust",
    extras=("render",),
    unlocks="every derived field the matplotlib render engine draws",
    remedy=("  remedy: pip install 'recast-woof[render]'\n"
            "  # the extra is wrf-rust (the mandated science core) plus\n"
            "  # pyshp; `woof setup` stages the rust render engine, which\n"
            "  # draws the full catalog and needs neither."))

SHAPEFILE_READER = Requirement(
    module="shapefile",
    distribution="pyshp",
    extras=("render",),
    unlocks="the coastline/border basemap every DA and tile plot draws",
    remedy="  remedy: pip install 'recast-woof[render]'")

#: scipy is spelled by two extras with identical contents.  Both are
#: named, because which one a reader wants depends on which door they
#: came through and a remedy that guesses is the defect this module is
#: about.
#: scipy is a BASE dependency as of this release, so this remedy names
#: no extra.  It used to offer ``recast-woof[obs]`` and ``recast-woof[dealias]``,
#: which were the extras that carried scipy; the packaging lane moved
#: scipy into the runtime dependencies and left both extras declared and
#: EMPTY, so that they keep resolving for every install line already
#: written down.  Offering an empty extra is the failure mode this whole
#: registry exists to prevent: `pip install 'recast-woof[dealias]'` would
#: report success, install nothing, and the next attempt would fail
#: identically.  Reaching this message now means a declared dependency
#: went missing, which is a damaged environment and has the same remedy
#: as any other base dependency.  Same words as
#: :data:`woof.obs.dealias.SCIPY_REMEDY`, which reached this conclusion
#: first, in the module that owns the engine.
SCIPY = Requirement(
    module="scipy",
    distribution="scipy",
    extras=(),
    unlocks="obs-battery regridding and the vad-region dealiasing engine",
    remedy=("  remedy: pip install --upgrade --force-reinstall 'scipy>=1.11'\n"
            "  # scipy is a BASE dependency of woof, so this is a damaged\n"
            "  # environment rather than a missing extra; `pip install\n"
            "  # --force-reinstall woof` restores the whole set."))

#: Pillow is in NO woof extra.  It arrives transitively with
#: matplotlib, a base dependency, so on a healthy install it is already
#: there -- which is exactly why the old ``--pair`` message could send a
#: reader to ``recast-woof[render]`` for years without anyone noticing that
#: the extra does not contain it.
PILLOW = Requirement(
    module="PIL",
    distribution="Pillow",
    extras=(),
    unlocks="the side-by-side pair sheets `woof render --pair` composes",
    remedy=("  remedy: pip install Pillow\n"
            "  # Pillow is in no woof extra: it normally arrives with\n"
            "  # matplotlib, a base dependency, so a base install that\n"
            "  # lacks it is an incomplete install rather than a missing\n"
            "  # feature."))

def _companion_remedy() -> str:
    """The companion's remedy, from the module that owns it.

    NOT a second copy: :func:`woof.data_assets.companion_install_command`
    is the single spelling, because the pip line carries the version
    ``woof`` was built against and a restatement here would be a literal
    that goes stale at the next cut.

    One line and nothing after it, deliberately.  This is the remedy a
    reader meets at the END of the refusal, and a trailing ``#`` note
    would put commentary between them and the command.
    """

    return f"  remedy: {data_assets.companion_install_command()}"


#: The packaged reference tables, which live in their own distribution
#: since 2.5.0 -- the ``woof`` wheel had reached 103.62 MiB against
#: PyPI's 100 MiB per-file cap and these two directories were 64.21 MiB
#: of it (see :mod:`woof.data_assets`).
#:
#: It is registered here for one reason: ``woof.core.rrtmgp`` resolves
#: its data directory at MODULE level, so on an install whose companion
#: was uninstalled the refusal surfaces during an import, deep inside
#: the preflight, and reaches :func:`woof.cli.main`'s
#: ``ModuleNotFoundError`` branch.  That branch derives its remedy from
#: the missing MODULE, and a module this table does not know gets no
#: remedy and is re-raised as a stack -- measured at 15 frames from
#: `woof check` and 17 from `woof domain`.
#:
#: No extra, and the remedy says why rather than offering one: the
#: companion is a HARD ``==`` dependency, so a plain `pip install recast-woof`
#: brings it and this state means the install was edited afterwards.
COMPANION_DATA = Requirement(
    module=data_assets.COMPANION_PACKAGE,
    distribution=data_assets.COMPANION_DISTRIBUTION,
    extras=(),
    unlocks=("every radiation scheme and every mp_physics=8/28 run -- it "
             "carries the RRTMGP k-distribution and cloud-optics tables "
             "and the Thompson microphysics lookup tables"),
    remedy=_companion_remedy())


def _geog_remedy() -> str:
    """The geography stack's remedy, from the module that owns it.

    NOT a second copy.  :mod:`woof.static.geog_stack` is the single
    source of truth for this one, by its own docstring's rule -- "the
    remedy is one string, written once, so the refusal, the deep guards
    and ``woof doctor`` cannot drift into naming different install
    commands for the same missing library."  This registry consumes it
    rather than restating it.

    It matters here specifically, and the reason has now inverted twice.
    From 2.3.3 ``rasterio`` and ``pyproj`` were ordinary runtime
    dependencies and the ``geog`` extra was EMPTY, so a remedy of ``pip
    install 'recast-woof[geog]'`` would resolve, succeed and install nothing.
    Since the warp substrate flipped onto the Rust static-fields library
    the extra carries them again -- and a remedy of ``pip install
    --upgrade woof`` would now be the empty one, because the default
    high-resolution path does not read either library.  Consuming the
    owning module's string is what keeps this correct without anybody
    remembering which way round it currently is.
    """

    from woof.static.geog_stack import geog_unavailable_detail

    detail = geog_unavailable_detail()
    return detail[detail.index("  remedy:"):]


RASTERIO = Requirement(
    module="rasterio",
    distribution="rasterio",
    extras=("geog",),
    unlocks=("the mosaic step of the PURE-PYTHON high-resolution "
             "fallback (WOOF_STATIC_PYTHON=1); the default engine is "
             "the Rust static-fields library, which needs neither"),
    remedy=_geog_remedy())

PYPROJ = Requirement(
    module="pyproj",
    distribution="pyproj",
    extras=("geog",),
    unlocks=("the reprojection step of the PURE-PYTHON high-resolution "
             "fallback (WOOF_STATIC_PYTHON=1); the default engine is "
             "the Rust static-fields library, which needs neither"),
    remedy=_geog_remedy())


def _base_dependency(module: str, distribution: str, unlocks: str
                     ) -> Requirement:
    """A requirement for something the base install already declares.

    Its absence is not a missing feature, it is a damaged install, and
    the remedy says so rather than naming an extra that would not
    reinstall it.
    """

    return Requirement(
        module=module, distribution=distribution, extras=(), unlocks=unlocks,
        remedy=(f"  remedy: pip install --upgrade --force-reinstall "
                f"'{distribution}'\n"
                f"  # {distribution} is a BASE dependency of woof, so this\n"
                f"  # is a damaged environment rather than a missing extra;\n"
                f"  # `pip install --force-reinstall woof` restores the set."))


#: Every module any woof front door can be missing, and its remedy.
#: Keyed by the top-level import name, which is what
#: ``ModuleNotFoundError.name`` carries.
REQUIREMENTS: tuple[Requirement, ...] = (
    GPU_RUNTIME,
    SCIENCE_CORE,
    SHAPEFILE_READER,
    SCIPY,
    PILLOW,
    RASTERIO,
    PYPROJ,
    COMPANION_DATA,
    _base_dependency("numpy", "numpy", "every array in the product"),
    _base_dependency("netCDF4", "netCDF4", "reading and writing wrfout files"),
    _base_dependency("matplotlib", "matplotlib",
                     "the matplotlib render engine and every analysis chart"),
    _base_dependency("jsonschema", "jsonschema",
                     "run-plan document validation"),
)

_BY_MODULE = {requirement.module: requirement for requirement in REQUIREMENTS}

#: Import names that mean one of the registered requirements.  CuPy in
#: particular fails under three different top-level names depending on
#: which of its own submodules the importer reached first.
_ALIASES = {
    "cupy_backends": "cupy",
    "cupyx": "cupy",
    "Pillow": "PIL",
    "wrf_rust": "wrf",
}


def requirement_for_module(name: str | None) -> Requirement | None:
    """The registered requirement a module name means, or ``None``.

    ``None`` is a real answer and the caller must treat it as one: a
    module this registry has never heard of gets NO remedy, because the
    alternative -- the nearest remedy in the table -- is how a missing
    ``scipy`` came to be answered with a CuPy install line.
    """

    if not name:
        return None
    top = str(name).split(".")[0]
    top = _ALIASES.get(top, top)
    return _BY_MODULE.get(top)


def is_installed(module: str) -> bool:
    """Whether ``module`` RESOLVES here, without importing it.

    ``find_spec`` walks the same finders an ``import`` would and stops
    before executing the module body.  That is deliberate on both
    counts: importing CuPy at a front door costs seconds and, for the
    orchestrating processes, would leave the library loaded for the
    life of a run that has not started yet.

    It proves the module is INSTALLED.  It proves nothing about whether
    it then works -- a CuPy wheel built for the wrong CUDA major
    resolves perfectly and dies at its first cuBLAS load.  Verifying
    that is ``woof doctor``'s job, which runs a real import and a real
    matmul in a subprocess.  This is the cheap, always-safe half.
    """

    top = str(module).split(".")[0]
    if top in sys.modules:
        return True
    try:
        return importlib.util.find_spec(top) is not None
    except (ImportError, ValueError):
        # A parent package that is itself broken, or a module whose
        # spec cannot be built.  Not installed, for this question.
        return False


def missing(requirements) -> tuple[Requirement, ...]:
    """Which of ``requirements`` do not resolve here, in the given order."""

    return tuple(item for item in requirements if not is_installed(item.module))


def remedy_for_module(name: str | None) -> str | None:
    """The remedy block for a missing module, or ``None`` when unknown."""

    requirement = requirement_for_module(name)
    return None if requirement is None else requirement.remedy


#: ``No module named 'x'`` -- CPython's own wording, which survives
#: every relay this product does: a worker's last stderr line quoted
#: into a ``SupervisorError``, a stage's output quoted into a chain
#: message, a plan run's ``failed`` event.  The MODULE NAME is what is
#: extracted; the sentence around it is not interpreted.
_MODULE_IN_MESSAGE = re.compile(r"No module named ['\"]([A-Za-z0-9_.]+)['\"]")


def remedy_for_error(error: BaseException) -> str | None:
    """The remedy for a failure, DERIVED from what the failure names.

    NEVER from the exception's class.  ``woof run-plan`` used to look
    the class name up in a table, so every ``ModuleNotFoundError``
    inside a plan run -- ``wrf``, ``scipy``, ``shapefile`` -- was
    reported to the caller's event stream with the CuPy remedy.  The
    class says nothing about which package is absent; the module name
    says everything, and it is carried three ways:

    1. ``ModuleNotFoundError.name``, when the failure is raised here;
    2. ``No module named 'x'`` inside the message, when a child process
       reported it and a parent relayed the text;
    3. a registered module or distribution named in quotes anywhere in
       an ``ImportError``'s message, which is the shape of this
       product's own re-raises.

    ``None`` when the failure names nothing this registry knows.  That
    is the accurate answer and the caller must print no remedy at all --
    the nearest available remedy is exactly the defect this replaces.
    """

    if isinstance(error, CapabilityMissing):
        # The most direct carrier there is: this refusal was raised
        # BECAUSE of a specific requirement and holds it.
        return error.requirement.remedy
    requirement = requirement_for_module(getattr(error, "name", None))
    if requirement is not None:
        return requirement.remedy
    text = str(error)
    found = _MODULE_IN_MESSAGE.search(text)
    if found is not None:
        requirement = requirement_for_module(found.group(1))
        if requirement is not None:
            return requirement.remedy
    if not isinstance(error, ImportError):
        return None
    for candidate in REQUIREMENTS:
        if (f"'{candidate.module}'" in text
                or f"'{candidate.distribution}'" in text):
            return candidate.remedy
    return None


def refusal(door: str, requirement: Requirement, *,
            before: str | None = None) -> str:
    """The layered refusal text one front door prints for one gap.

    ``door`` is the front door AS A READER WOULD TYPE IT -- ``woof
    go``, ``woof render --pair``, ``woof-prepared-forecast``,
    ``python -m woof.prepared_domain_tree_forecast``.  Verbatim, not
    assembled from a subcommand name, because the ``python -m`` and
    console-script doors are not subcommands and a message that called
    them one would be naming a command that does not exist.

    ``before`` is the sentence that says WHAT WAS NOT DONE -- the fetch
    that did not run, the directory that was not created.  It is the
    whole point of refusing here rather than downstream, so it is a
    parameter rather than boilerplate: a door that cannot name the
    expensive work it just avoided has not earned the claim.
    """

    action = (f"{door}: this command needs "
              f"{requirement.label}, which this install does not have.\n"
              f"  What needs it: {requirement.unlocks}.\n")
    if requirement == GPU_RUNTIME:
        from woof.forecast_gpu import ForecastGPURequired

        action += f"  {ForecastGPURequired('CuPy is not installed')}\n"
    if before:
        action += f"  {before}\n"
    action += requirement.remedy
    return layered(action, _WHY.format(
        module=requirement.module,
        distribution=requirement.distribution,
        extras=(", ".join(f"woof[{extra}]" for extra in requirement.extras)
                or "no woof extra -- see the remedy")))


_WHY = (
    "Checked at the front door, by resolving {module} without importing "
    "it: `importlib.util.find_spec` walks the same finders an import "
    "would and stops before running the module body.  So this refusal "
    "is a statement that {distribution} is NOT INSTALLED, and nothing "
    "more -- it does not claim that an installed one would work.  "
    "`woof doctor` makes that second claim, by importing in a "
    "subprocess and exercising the runtime.\n\n"
    "The extras that ship it: {extras}.\n\n"
    "The refusal is here, ahead of the work, because the alternative "
    "was measured: the same gap used to surface as a relayed traceback "
    "after gigabytes of forcing data had been downloaded and a case "
    "prepared, with no extra named anywhere in the output.")


def require(door: str, *requirements: Requirement,
            before: str | None = None) -> None:
    """Refuse at ``door`` if any of ``requirements`` is not installed.

    Raises :class:`CapabilityMissing` for the FIRST gap, in the order
    given -- one refusal with one remedy, because a reader handed three
    install lines at once acts on none of them.  Silent when everything
    resolves, which is the direction that must also be tested.
    """

    for requirement in missing(requirements):
        raise CapabilityMissing(
            refusal(door, requirement, before=before),
            requirement=requirement, command=door)
    if GPU_RUNTIME in requirements:
        found = outdated_gpu_runtime()
        if found is not None:
            raise CapabilityMissing(
                gpu_runtime_floor_refusal(door, found, before=before),
                requirement=GPU_RUNTIME, command=door)


# ---------------------------------------------------------------------------
# Command -> capability table
# ---------------------------------------------------------------------------

#: What each front door cannot run without, checked before it does
#: anything expensive.  ONE table, so `go`, `run`, `resume`, `verify`
#: and the two prepared runners cannot drift apart again -- that drift
#: is exactly why `woof verify` printed a clean CuPy refusal while
#: `woof run`, on the identical install, relayed a raw traceback from a
#: worker it had already spawned.
#:
#: Deliberately NOT here:
#:
#: * ``stream``, which returns early rather than "ask for a GPU merely
#:   to enter a zero-iteration loop" -- a front-door gate would refuse
#:   the one path that correctly needs no card.
#: * ``multi-run``, whose ``--preflight off`` / ``--summary`` forms are
#:   not proven to need a card.
#: * ``check`` and ``domain``, which already refuse cleanly through the
#:   import boundary and whose estimator forms may legitimately run with
#:   no runtime installed.
#:
#: A command is added here only when EVERY path it can take needs the
#: capability.  A gate that refuses a working path is the same defect as
#: a traceback, pointed the other way.
COMMAND_REQUIREMENTS: dict[str, tuple[Requirement, ...]] = {
    "go": (GPU_RUNTIME,),
    # `woof ensemble` is `go` by another name, and argparse stores the
    # name the reader typed, so a table keyed on "go" alone misses it.
    # Without the row an install with no CuPy fetched and prepared every
    # member of a recipe ensemble before the forecast stage refused.
    "ensemble": (GPU_RUNTIME,),
    "run": (GPU_RUNTIME,),
    "resume": (GPU_RUNTIME,),
    # The unbundled forecast stage integrates the same model `go`'s
    # fifth stage does, so it owes the same preflight.  Without the row
    # here, `woof sim` on an install with no CuPy would have refused
    # deep inside the runner instead of at the front door -- the exact
    # failure this table was added to close, re-opened by a new door.
    "sim": (GPU_RUNTIME,),
    "fire-ideal": (GPU_RUNTIME,),
    "verify": (GPU_RUNTIME,),
}

#: What each command would have spent before the old failure point.
#: Printed in the refusal, so the reader can see what the check saved.
_BEFORE = {
    "go": ("Refusing here, before the fetch stage downloads the forcing "
           "data and the prepare stage writes the static fields, rather "
           "than at the forecast stage after both."),
    "ensemble": ("Refusing here, before any member's forcing data is "
                 "downloaded or prepared, rather than at the forecast "
                 "stage after every member's fetch and preparation."),
    "run": ("Refusing here, before a GPU is selected, a worker spawned "
            "and the case prepared."),
    "resume": ("Refusing here, before a GPU is selected and a worker "
               "spawned."),
    "sim": ("Refusing here, before the prepared tree is opened and its "
            "identity recomputed."),
    "fire-ideal": ("Refusing here, before the native sounding and ideal "
                   "fire state are initialized on the card."),
    "verify": "Refusing here, before the case allocates anything.",
}


def require_for_command(command: str) -> None:
    """The front-door preflight for one ``woof`` subcommand.

    Called once, from :func:`woof.cli.main`, after argparse and before
    dispatch -- so ``--help`` is never refused and no subcommand can
    forget to call it.
    """

    requirements = COMMAND_REQUIREMENTS.get(command)
    if not requirements:
        return
    require(f"woof {command}", *requirements, before=_BEFORE.get(command))
    # The opt-out preserves host orchestration and declared-runtime checks;
    # it is not evidence that an installed CuPy is absent or broken. Device
    # allocation and step still refuse before any CUDA contact under it.
    from woof.local_gpu import no_local_gpu

    if (not no_local_gpu()
            and command in ("run", "go", "sim", "resume", "ensemble", "fire-ideal")):
        from woof.forecast_gpu import require_forecast_gpu

        require_forecast_gpu()


def unmet_run_requirements() -> tuple[Requirement, ...]:
    """Every gap that stops this install from executing a model run.

    The readiness half of ``woof run-plan --probe`` reads this, so that
    document cannot report ``"ready": true`` on an install whose very
    next step would refuse.
    """

    unmet = missing((GPU_RUNTIME,))
    if not unmet and outdated_gpu_runtime() is not None:
        return (GPU_RUNTIME,)
    return unmet


__all__ = ["CapabilityMissing", "COMMAND_REQUIREMENTS", "COMPANION_DATA",
           "GPU_RUNTIME", "GPU_RUNTIME_MINIMUM",
           "GPU_RUNTIME_REMEDY", "gpu_runtime_floor_refusal",
           "installed_cupy_version", "outdated_gpu_runtime", "PILLOW", "PYPROJ", "RASTERIO",
           "REQUIREMENTS", "Requirement", "SCIENCE_CORE", "SCIPY",
           "SHAPEFILE_READER", "is_installed", "missing",
           "remedy_for_error", "remedy_for_module", "refusal", "require",
           "require_for_command", "requirement_for_module",
           "unmet_run_requirements"]
