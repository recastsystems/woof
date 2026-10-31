"""One resolution ladder for the Rust engines this package drives.

Nothing in this distribution draws a weather field, interpolates a
meteorological column, or writes an initial condition in Python.  Those
are Rust binaries -- ``rw_mpas_init``, ``rw_mpas_convert``,
``rw_wrfbatch`` -- built from the ``tools/rustwx`` workspace that lives
in the woof repository.  The wheel therefore CANNOT carry them, and a
door that needs one has exactly two states: it found the binary, or it
refuses by name and says what to run.

The reason this module exists rather than a ladder per door: the doors
were each resolving their own way, and one of them was wrong in a way
nobody could see from inside it.  ``pip install woof hex`` pulls
``woof``; ``woof fetch-bridges`` then stages a release's prebuilt
bundle -- and that bundle carries ``rw_mpas_mesh``, ``rw_mpas_static``,
``rw_mpas_init``, ``rw_mpas_convert`` and ``rw_wrfbatch`` -- into
``~/.woof/bridges``.  The init door read one environment variable and
PATH, so a user who had run the one command that installs the engine
still met a refusal telling them to build it with cargo.  A remedy that
ignores the estate the user already has is not a remedy.

THE LADDER, best first, identical for every engine:

1. the door's own command-line flag;
2. this distribution's environment variable, then any legacy spellings
   it has carried (a rename never silently stops reading a variable);
   then woof's own environment variable;
3. this distribution's own ``woof/hex/libexec``: the engines the woof hex
   wheel bundles because no released engine carries the version this
   release needs (``rw_mpas_geometry`` with hex-vertical-v1,
   ``rw_mpas_stream``, an ``rw_mpas_static`` whose generated-grid statics
   hold the binary64 coordinates they declare, and the ``rw_mpas_lbc`` whose
   boundary files equal native init_atmosphere case 9; ``data/engine-bridges.json``
   pins their bytes);
4. woof's bridge ladder -- a woof
   checkout's ``tools/rustwx/target/{release,debug}``, ``libexec/bridges``
   beside the installed package, the wheel-bundled directory inside it,
   and ``~/.woof/bridges`` where ``woof fetch-bridges`` stages;
5. ``PATH``.

Rungs 1, 2 and 4 fail LOUDLY when they name a file that is not there.  An
explicit configuration that falls through to a different binary is how a
box runs the wrong engine and reports success, so a named-but-missing
path is a refusal, never a skipped rung.

Rung 4 is read through :mod:`woof.mpas_mesh` when that module can be
imported, so the two ladders cannot drift; when it cannot -- an older
woof, or none at all -- the two directories that do not depend on
woof's internals (``~/.woof/bridges`` and ``libexec/bridges`` beside
the package) are still probed directly, and the refusal names the
version floor instead of pretending the rung does not exist.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil

from .errors import MpasPortError


class EngineRefusal(MpasPortError):
    """No engine could be resolved: what breaks, and the command that fixes it."""


#: The command that stages a prebuilt bundle, named in every refusal.
FETCH_COMMAND = "woof fetch-bridges"

#: Where that command stages, and the last rung of woof's ladder.
USER_BRIDGE_DIR = Path.home() / ".woof" / "bridges"


def executable_name(name: str) -> str:
    return f"{name}.exe" if os.name == "nt" else name


def bundled_program(name: str) -> str | None:
    """This wheel's own copy of a bundled program (``woof/hex/libexec``), or None.

    THE BREAKAGE THIS PREVENTS (RRFS-MPAS from a pip install, 2026-10-15): the
    RRFS fork dycore looked for rw_mpas_rrfs_connectivity, rw_mpas_rrfs_damping
    and rw_mpas_rrfs_deformation only in their environment variables and on
    PATH, so a clean install refused every RRFS forecast.  The wheel bundles them
    (setup.py, ``data/engine-bridges.json``); callers read an explicit setting
    first, then this, then PATH.
    """

    from .helpers import path as helper_path
    path = helper_path(executable_name(name))
    return str(path) if path.is_file() and os.access(path, os.X_OK) else None


@dataclass(frozen=True)
class EngineSpec:
    """One Rust binary: how it is named, what it owns, how it is built."""

    #: The artifact's basename, without any platform suffix.
    name: str
    #: The door's flag, printed in the resolution order.
    flag: str
    #: Environment variables, preferred first, legacy after.  Every one is
    #: read; none is ever dropped.
    env_names: tuple[str, ...]
    #: woof's own spelling for the same artifact, read after this
    #: distribution's own and before the directory rungs.
    gpuwm_env: str
    #: What the engine is, in one noun phrase.
    subject: str
    #: What is unreachable without it -- the concrete breakage, not "an error".
    what_breaks: str
    #: The cargo package that builds it inside ``workspace``.
    cargo_package: str
    #: The woof checkout directory holding that cargo workspace.  The WPS
    #: decode bridges (``met_intermediate``, ``gdt101_remap``) build in
    #: ``tools/grib1_bridge``, not in ``tools/rustwx``, and a remedy that
    #: names the wrong directory is a command that fails.
    workspace: str = "tools/rustwx"


#: Meteorological interpolation and the initial-condition write.
INIT = EngineSpec(
    name="rw_mpas_init",
    flag="--engine",
    env_names=("WOOF_HEX_RW_MPAS_INIT", "RW_MPAS_INIT"),
    gpuwm_env="WOOF_RW_MPAS_INIT",
    subject="the initial-condition builder",
    what_breaks=(
        "no meteorological interpolation and no initial-condition write can "
        "occur, so the init door has nothing to produce"
    ),
    cargo_package="rw-mpas",
)

#: Exact derived mesh geometry and RBF reconstruction.
GEOMETRY = EngineSpec(
    name="rw_mpas_geometry",
    flag="geometry bridge",
    env_names=("WOOF_HEX_RW_MPAS_GEOMETRY", "RW_MPAS_GEOMETRY"),
    gpuwm_env="WOOF_RW_MPAS_GEOMETRY",
    subject="the derived geometry builder",
    what_breaks="a native-free init has no edge normals, tangent planes or RBF reconstruction coefficients",
    cargo_package="rw-mpas",
)

#: Exact initialized terrain metrics and vertical momentum correction.
HOSTPREP = EngineSpec(
    name="rw_mpas_hostprep",
    flag="host preparation bridge",
    env_names=("WOOF_HEX_RW_MPAS_HOSTPREP", "RW_MPAS_HOSTPREP"),
    gpuwm_env="WOOF_RW_MPAS_HOSTPREP",
    subject="the terrain and momentum preparation bridge",
    what_breaks="an initialized forecast has no cell-oriented terrain metrics or coupled vertical momentum",
    cargo_package="rw-mpas",
)

#: History onto the renderer's tape.
CONVERT = EngineSpec(
    name="rw_mpas_convert",
    flag="--convert-exe",
    env_names=("WOOF_HEX_RW_MPAS_CONVERT", "MPAS_PORT_RW_MPAS_CONVERT"),
    gpuwm_env="WOOF_RW_MPAS_CONVERT",
    subject="the history converter",
    what_breaks=(
        "a native history has no wrfout-shaped frame for the renderer to "
        "import, so nothing can be drawn"
    ),
    cargo_package="rw-mpas",
)

#: The renderer itself.  There is no Python plotter behind this door.
RENDERER = EngineSpec(
    name="rw_wrfbatch",
    flag="--renderer-exe",
    env_names=(
        "WOOF_HEX_RW_WRFBATCH",
        "MPAS_PORT_RW_WRFBATCH",
    ),
    gpuwm_env="WOOF_RW_WRFBATCH",
    subject="the product renderer",
    what_breaks=(
        "weather-field products come from this binary only -- this door has "
        "no Python plotter behind it -- so there are no products"
    ),
    cargo_package="rw-wrfbatch",
)

#: Mesh generation, sizing and the limited-area cull.
#:
#: The swath door drives this one to PRICE a placement before anything is
#: built: ``--dry-run`` sizes a resolution spec on the CPU, writes nothing
#: and needs no card, so a plan can be refused for not fitting before a
#: cycle spends a minute on it.  The same binary takes the same shape row
#: as ``--region`` for the limited-area cull, which is why the swath layer
#: emits shapes the generator already understands rather than a shape kind
#: of its own.
MESH = EngineSpec(
    name="rw_mpas_mesh",
    flag="--mesh-exe",
    env_names=("WOOF_HEX_RW_MPAS_MESH", "RW_MPAS_MESH"),
    gpuwm_env="WOOF_RW_MPAS_MESH",
    subject="the mesh generator and region culler",
    what_breaks=(
        "a swath plan cannot be priced or culled: nothing can say how many "
        "cells a placement costs before it is built, so the cycle would "
        "discover a swath does not fit the card by running out of memory"
    ),
    cargo_package="rw-mpas",
)

#: The static builder: geography onto a generated grid.
#:
#: ``woof hex mesh-plan --point --generate`` drives it right after the
#: generator, because a grid without its static is not a deliverable: the
#: registry pins the two together by byte count and SHA-256 and refuses a
#: grid carrying one.  Staged by the same bundle as the four above.
STATIC = EngineSpec(
    name="rw_mpas_static",
    flag="--static-exe",
    env_names=("WOOF_HEX_RW_MPAS_STATIC", "RW_MPAS_STATIC"),
    gpuwm_env="WOOF_RW_MPAS_STATIC",
    subject="the static builder",
    what_breaks=(
        "a generated grid gets no static -- no terrain, land use, soil or "
        "deriv_two -- and the registry refuses a grid with no matching static, "
        "so a mesh generated at a point can never be run"
    ),
    cargo_package="rw-mpas",
)

#: The lateral-boundary producer: one boundary file per driving time.
#:
#: Every fine grid this distribution cuts is a limited-area cull whose
#: seven boundary rings hold no atmosphere of their own, so without this
#: binary a mesh can be generated, initialised and never integrated.  It
#: used to be resolved by ``woof.hex.cycle.chain`` alone, through one
#: environment variable and PATH, which is exactly the shape that made
#: the init door refuse on a box where ``woof fetch-bridges`` had already
#: staged the file: the 2.7.3 to 2.8.0 bundles carry ``rw_mpas_lbc`` beside the
#: other four, and the ladder reads it from there.
#:
#: The wheel bundles it now (hex 0.4.0 install audit, 2026-10-15): the case-9
#: port whose boundary files equal native init_atmosphere v8.4.2 case 9 is in no
#: released woof, and 2.9.0's copy refuses a regional grid without model_name,
#: so a plain install found 2.9.0's file on rung 4 and could not build
#: native-identical boundaries.  The bundled copy is read first (rung 3).
LBC = EngineSpec(
    name="rw_mpas_lbc",
    flag="--lbc-exe",
    env_names=("WOOF_HEX_RW_MPAS_LBC", "RW_MPAS_LBC"),
    gpuwm_env="WOOF_RW_MPAS_LBC",
    subject="the lateral-boundary producer",
    what_breaks=(
        "no boundary series can be built: every fine grid in a cascade is a "
        "limited-area cull whose seven boundary rings hold no atmosphere of "
        "their own, so a mesh can be cut and initialised that can never be "
        "integrated"
    ),
    cargo_package="rw-mpas",
)

#: Every engine a front door of this distribution drives, best-known name
#: first.  ``woof.hex.doctor`` reads this, so adding an engine is a row
#: here and is reported without a second edit.
#: Soundings and meteograms at points, straight off native history.  Listed
#: here so the one ladder finds it and `woof hex doctor` names it: the
#: product door's third binary was resolved by a search of its own that
#: doctor could not see, so a fresh install learned it was missing only at
#: the end of a product run.
POINTS = EngineSpec(
    name="rw_hex_points",
    flag="--points-exe",
    env_names=("WOOF_HEX_RW_HEX_POINTS",),
    gpuwm_env="WOOF_RW_HEX_POINTS",
    subject="the sounding and meteogram renderer",
    what_breaks=(
        "soundings and meteograms are drawn by this binary and nothing else, "
        "so the product door has no point pictures"
    ),
    cargo_package="rw-wrfbatch",
)

#: The WPS intermediate writer every init-source row decodes through.
#: THE BREAKAGE THIS ROW PREVENTS, measured on a clean pip install of the
#: 0.4.0 candidate beside woof 2.9.0 (g16 fresh-install QA, 2026-10-12):
#: ``init-source`` read only ``--writer``, ``$WOOF_MET_INTERMEDIATE`` and
#: PATH, so it refused "met_intermediate was not found" while the woof
#: wheel in the same venv carried the binary in ``libexec/bridges`` and
#: ``woof hex doctor`` never mentioned it.  As an ENGINES row it is
#: resolved by the same ladder as every other engine and doctor names it.
MET_INTERMEDIATE = EngineSpec(
    name="met_intermediate",
    flag="--writer",
    env_names=("WOOF_HEX_MET_INTERMEDIATE",),
    gpuwm_env="WOOF_MET_INTERMEDIATE",
    subject="the WPS intermediate writer",
    what_breaks=(
        "no source's fields can be written as the WPS intermediate the init "
        "door reads, so init-source cannot build an initial condition from any row"
    ),
    cargo_package="grib1_bridge",
    workspace="tools/grib1_bridge",
)

#: The regular-grid remapper for rows whose source grid is unstructured
#: (the table's ``remap`` rows).  Same ladder, same breakage as above; not an
#: ENGINES row because only those rows need it, so doctor does not count it
#: against a GFS-only install.
GDT101_REMAP = EngineSpec(
    name="gdt101_remap",
    flag="--remapper",
    env_names=("WOOF_HEX_GDT101_REMAP",),
    gpuwm_env="WOOF_GDT101_REMAP",
    subject="the unstructured-to-regular remapper",
    what_breaks=(
        "a source published on an unstructured grid cannot be put on the "
        "regular grid met_intermediate reads, so its init-source row cannot run"
    ),
    cargo_package="grib1_bridge",
    workspace="tools/grib1_bridge",
)

#: The RRFS-MPAS initializer: the fork's init_atmosphere (cases 7 and 9, its
#: static and vertical grid) as one Rust worker taking hex-init-job/v1 jobs on
#: stdin.  Not an ENGINES row: only the RRFS producer row of
#: data/profiles/init-producers.v1.json needs it.
INIT_RRFS = EngineSpec(
    name="rw_mpas_init_v831",
    flag="--rrfs-worker",
    env_names=("WOOF_HEX_RW_MPAS_INIT_V831",),
    gpuwm_env="WOOF_RW_MPAS_INIT_V831",
    subject="the RRFS-MPAS initializer",
    what_breaks=(
        "no RRFS-MPAS initial condition or boundary file can be made, so "
        "`woof hex init --physics-suite hrrrv5` (and every RRFS suite) has no producer"
    ),
    cargo_package="rw-mpas",
)

#: The GSL orographic statistics generator (ugwp_oro_data.nc for the RRFS
#: suites' bl_ugwp_gwdo drag).  Same reason as above for not being an ENGINES row.
GSL_ORO = EngineSpec(
    name="rw_mpas_gsl_oro",
    flag="--gsl-oro",
    env_names=("WOOF_HEX_RW_MPAS_GSL_ORO",),
    gpuwm_env="WOOF_RW_MPAS_GSL_ORO",
    subject="the GSL orographic statistics generator",
    what_breaks=(
        "ugwp_oro_data.nc cannot be made by the GSL generator RRFS uses, so the RRFS "
        "drag statistics fall back to hex's own method"
    ),
    cargo_package="rw-mpas",
)

ENGINES: tuple[EngineSpec, ...] = (
    INIT, GEOMETRY, HOSTPREP, CONVERT, MESH, STATIC, LBC, RENDERER, POINTS, MET_INTERMEDIATE)


# ---------------------------------------------------------------------------
# woof's ladder, read through woof when possible
# ---------------------------------------------------------------------------
def _gpuwm_module_candidates(spec: EngineSpec) -> tuple[Path, ...] | None:
    """woof's own candidate list for this artifact, or ``None``.

    ``None`` means woof could not answer -- not installed, too old to
    know this artifact, or an import that raised.  The caller falls back
    to the two directory rungs that do not need woof's internals, and
    the refusal names the floor.
    """

    try:
        from woof import mpas_mesh  # noqa: PLC0415 - lazy on purpose
    except Exception:  # pragma: no cover - depends on the installed estate
        return None
    bridge = getattr(mpas_mesh, "BRIDGES", {}).get(spec.name)
    if bridge is None:
        return None
    try:
        return tuple(Path(candidate) for candidate in bridge.candidates())
    except Exception:  # pragma: no cover - a woof whose ladder changed shape
        return None


def gpuwm_bundles(spec: EngineSpec) -> bool | None:
    """Does THIS installed woof's ``fetch-bridges`` carry this engine?

    ``True`` yes, ``False`` woof is here but its bundle has no such row,
    ``None`` no woof at all.

    Asked of the INSTALLED distribution, never assumed from a version
    number, because the answer has already differed from the obvious one:
    woof's published 2.5.2 wheel declares ``rw_wrfbatch`` among its
    bundled artifacts and does NOT declare the four MPAS binaries, while
    a woof source checkout of the same era declares all five.  A remedy
    written from the checkout tells a user on the released wheel to run a
    command that cannot give them the file, which is worse than saying
    nothing -- they run it, it succeeds, and the door still refuses.
    """

    try:
        from woof import bridge_assets  # noqa: PLC0415 - lazy on purpose
    except Exception:  # pragma: no cover - depends on the installed estate
        return None
    try:
        return any(
            getattr(artifact, "name", None) == spec.name
            for artifact in bridge_assets.BUNDLED_ARTIFACTS
        )
    except Exception:  # pragma: no cover - a woof whose table changed shape
        return None


def _gpuwm_package_dir() -> Path | None:
    try:
        import woof  # noqa: PLC0415 - lazy on purpose
    except Exception:  # pragma: no cover - depends on the installed estate
        return None
    location = getattr(woof, "__file__", None)
    return Path(location).resolve().parent if location else None


def _directory_candidates(spec: EngineSpec) -> tuple[Path, ...]:
    """The rungs that survive woof being absent or old."""

    filename = executable_name(spec.name)
    candidates: list[Path] = []
    package = _gpuwm_package_dir()
    if package is not None:
        candidates.append(package / "libexec" / "bridges" / filename)
        candidates.append(package.parent / "libexec" / "bridges" / filename)
    candidates.append(USER_BRIDGE_DIR / filename)
    return tuple(candidates)


def gpuwm_candidates(spec: EngineSpec) -> tuple[Path, ...]:
    """Rung 4, whichever way it can be answered."""

    through_gpuwm = _gpuwm_module_candidates(spec)
    if through_gpuwm is not None:
        return through_gpuwm
    return _directory_candidates(spec)


def bundled_candidate(spec: EngineSpec, *, fetch: bool = True) -> Path:
    """Rung 3: where this wheel bundles the engine, if it bundles it at all.

    THE BREAKAGE THIS RUNG PREVENTS (release readiness B5, 2026-10-14): the
    init door needs ``rw_mpas_geometry`` speaking hex-vertical-v1 and the
    stream I/O needs ``rw_mpas_stream``.  The frozen 2.9.0 engine bundle
    carries a geometry bridge WITHOUT hex-vertical-v1 and no stream bridge,
    so a clean install found woof's older file first and the init door
    refused ("lacks hex-vertical-v1").  The wheel now bundles both
    (setup.py, ``data/engine-bridges.json``), and they are read before
    woof's directories so the bytes this release was proven with are the
    bytes it runs.  A source checkout's ``src/hexcore/libexec`` holds no
    such file, so checkout runs keep resolving down the ladder as before.

    ``rw_mpas_static`` is bundled for the same reason
    (2026-10-15): the 2.9.0 static builder reads every
    grid through binary32, so a generated grid's static declares binary64
    coordinates it does not hold and the point door refused (its vertical
    check, then ``rw_mpas_mesh --cull-parent``).  Read first, the bundled
    builder is the fixed one.
    """

    from .helpers import path as helper_path
    # A PyPI wheel leaves the helpers to the release's hex helper archive
    # (woof.hex.helpers): the first resolve stages them, verified.
    return helper_path(executable_name(spec.name), fetch=fetch)


# ---------------------------------------------------------------------------
# the remedy
# ---------------------------------------------------------------------------
def resolution_order(spec: EngineSpec) -> str:
    """The ladder in one line, so a refusal says where it looked."""

    read = [spec.flag]
    read.extend(f"${name}" for name in spec.env_names)
    read.append(f"${spec.gpuwm_env}")
    read.append("this distribution's woof/hex/libexec")
    return (
        f"{', '.join(read)}, woof's bridge directories "
        f"(a woof checkout's tools/rustwx/target/release, libexec/bridges "
        f"beside the installed package, and {USER_BRIDGE_DIR}), then PATH"
    )


def _engine_requirement() -> str:
    """The derived pip specifier, or the bare name if none can be derived.

    Imported inside the function because this is a REMEDY path: it runs when
    something is already wrong, and a remedy that raises while explaining a
    failure leaves the user with a traceback instead of a command.  If
    :func:`engine_pin.gpuwm_requirement` refuses -- which it does when no
    published engine satisfies the seam manifest -- the fallback is the
    distribution name alone, which is still a command that runs.
    """

    from .engine_pin import DISTRIBUTION, EnginePinError, gpuwm_requirement

    try:
        return gpuwm_requirement()
    except EnginePinError:
        return DISTRIBUTION


def remedy(spec: EngineSpec) -> str:
    """What to run, most likely to work first, ON THIS BOX.

    The order is not fixed text: it is decided by asking the installed
    woof whether its bundle actually carries this engine
    (:func:`gpuwm_bundles`).  Where it does, the one-command staging is
    offered first because it needs no toolchain.  Where it does not, the
    build is offered first and the staging command is NOT offered at
    all -- an offer that cannot deliver the file wastes a user's time
    and costs the next refusal its credibility.

    Every line is a command as typed or a ``#`` comment, never prose
    fused onto a command.
    """

    build = [
        "  # build it from a woof source checkout:",
        "  cargo build --release --locked --offline "
        f"-p {spec.cargo_package} --bin {spec.name}",
        f"      # run at {spec.workspace} in the checkout, then:",
        f"  # set {spec.env_names[0]} to the built binary",
    ]
    staged = [
        f"  {FETCH_COMMAND}",
        f"      # stages the prebuilt bundle -- which carries {spec.name} --",
        f"      # into {USER_BRIDGE_DIR}, where the ladder above reads it.",
        "      # Requires a published bundle for this platform.",
    ]

    bundled = gpuwm_bundles(spec)
    if bundled is None:
        return "\n".join(
            [
                # The engine range is DERIVED, never typed: this line read
                # `woof>=2.5.5` for a whole release cycle, naming a version
                # that is not on PyPI at all and whose bytes the seam
                # manifest refuses.  A remedy that cannot be executed costs
                # the next refusal its credibility.
                f'  pip install "{_engine_requirement()}"',
                "      # no woof is installed in this interpreter, so its",
                f"      # {FETCH_COMMAND} route is unavailable until one is.",
                "",
                *staged,
                "",
                "  # or, on a platform with no published bundle:",
                *build[1:],
            ]
        )
    if bundled:
        return "\n".join([*staged, "", "  # or, to build it yourself:", *build[1:]])
    return "\n".join(
        [
            *build,
            "",
            f"      # NOTE: `{FETCH_COMMAND}` will not supply {spec.name} on the",
            "      # woof installed here -- its bundle carries no such artifact.",
            "      # A later woof release adds it; until then the build above is",
            "      # the route.  Check with:  pip install --upgrade recast-woof",
        ]
    )


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------
def _named_but_missing(spec: EngineSpec, source: str, path: Path) -> None:
    raise EngineRefusal(
        f"{source} names {path}, which is not a file.  An explicit setting is "
        f"never skipped in favour of a different binary -- that is how a box "
        f"runs the wrong engine and reports success.  Point it at a built "
        f"{spec.name}, or unset it to continue down the ladder.\n"
        f"{remedy(spec)}"
    )


def _usable(path: Path) -> bool:
    if not path.is_file():
        return False
    if os.name != "posix":
        return True
    return os.access(path, os.X_OK)


def _not_executable(path: Path) -> bool:
    return os.name == "posix" and not os.access(path, os.X_OK)


_CHMOD_MARK = "; run: chmod +x "


def _not_executable_text(where: str, path: Path) -> str:
    return (
        f"{where} {path}, which exists but is not executable (pip does not "
        f"preserve executable bits on package data){_CHMOD_MARK}{path}"
    )


def chmod_remedy(source: str) -> str | None:
    """The chmod a :func:`locate` gap names, or ``None`` for any other gap."""

    _, mark, path = source.partition(_CHMOD_MARK)
    return f"chmod +x {path}" if mark else None


def _refuse_not_executable(spec: EngineSpec, path: Path) -> None:
    raise EngineRefusal(
        f"{spec.subject} at {path} exists but is not executable.  pip does "
        f"not preserve executable bits on package data, so a staged copy can "
        f"arrive present and unrunnable.\n"
        f"  chmod +x {path}"
    )


def resolve(spec: EngineSpec, explicit: str | Path | None = None) -> Path:
    """The engine binary, or a refusal naming the command that supplies it."""

    if explicit:
        candidate = Path(explicit).expanduser()
        if not candidate.is_file():
            _named_but_missing(spec, spec.flag, candidate)
        if os.name == "posix" and not os.access(candidate, os.X_OK):
            _refuse_not_executable(spec, candidate)
        return candidate

    for name in (*spec.env_names, spec.gpuwm_env):
        value = os.environ.get(name)
        if not value:
            continue
        candidate = Path(value).expanduser()
        if not candidate.is_file():
            _named_but_missing(spec, f"${name}", candidate)
        if os.name == "posix" and not os.access(candidate, os.X_OK):
            _refuse_not_executable(spec, candidate)
        return candidate

    bundled = bundled_candidate(spec)
    if bundled.is_file():
        if os.name == "posix" and not os.access(bundled, os.X_OK):
            _refuse_not_executable(spec, bundled)
        return bundled

    for attempt in range(2):
        for candidate in gpuwm_candidates(spec):
            candidate = Path(candidate).expanduser()
            if candidate.is_file():
                if os.name == "posix" and not os.access(candidate, os.X_OK):
                    _refuse_not_executable(spec, candidate)
                return candidate.resolve()
        # woof 2.9.10 leaves rw_mpas_init_v831 and rw_mpas_gsl_oro to its release
        # bundle (PyPI's per-file limit); their first use stages them where woof's
        # ladder reads, so a plain install's RRFS init runs instead of refusing.
        if attempt or not _fetch_on_demand(spec):
            break

    found = shutil.which(spec.name)
    if found:
        return Path(found)

    # "<name> not found" is the exact phrase the packaging battery greps
    # for, and it is the phrase a user greps for too.  Keep it.
    raise EngineRefusal(
        f"{spec.name} not found, so {spec.what_breaks}.\n"
        f"Looked at: {resolution_order(spec)}.\n"
        f"This distribution bundles only the engines no released woof carries "
        f"(rw_mpas_geometry, rw_mpas_stream, rw_mpas_static, rw_mpas_lbc and the three RRFS "
        f"programs); the other Rust binaries are "
        f"built from the woof tools/rustwx workspace and are staged onto a "
        f"machine.  Supply it with one of:\n"
        f"{remedy(spec)}"
    )


def _fetch_on_demand(spec: EngineSpec) -> bool:
    """Stage an engine woof ships on demand (in its bundle, not its platform wheel).

    True when woof staged it now. Never raises: an engine woof does not ship on
    demand, an older woof without the fetch, or a failed download all leave the
    ladder's refusal (and its remedy) to speak. Read-only callers (:func:`locate`,
    doctor) never come here.
    """

    try:
        from woof import bridges  # noqa: PLC0415 - lazy on purpose
    except Exception:  # pragma: no cover - depends on the installed estate
        return False
    on_demand = getattr(bridges, "fetched_on_demand", None)
    fetch = getattr(bridges, "fetch_on_demand", None)
    if on_demand is None or fetch is None:
        return False
    try:
        return bool(on_demand(spec.name) and fetch(spec.name))
    except Exception:  # pragma: no cover - a download that failed is reported by the refusal
        return False


def locate(spec: EngineSpec) -> tuple[Path | None, str]:
    """Resolve without raising: ``(path, where it came from)``.

    The read-only form :mod:`woof.hex.doctor` reports through, so the
    report can name every gap at once instead of stopping at the first.
    It answers the way :func:`resolve` does, rung for rung: a file that is
    present and not executable is a gap, not a find.  THE BREAKAGE THIS
    PREVENTS, measured on a pip install of 0.3.1: the engine wheel's
    bridges arrived mode 664, ``doctor`` reported every one of them found
    while the door refused the same file as not executable, so the one
    command meant to name every gap at once named none.
    """

    for name in (*spec.env_names, spec.gpuwm_env):
        value = os.environ.get(name)
        if not value:
            continue
        candidate = Path(value).expanduser()
        if candidate.is_file():
            if _not_executable(candidate):
                return None, _not_executable_text(f"${name} names", candidate)
            return candidate, f"${name}"
        return None, f"${name} names a missing file: {candidate}"

    bundled = bundled_candidate(spec, fetch=False)
    from .helpers import on_demand
    if not bundled.is_file() and on_demand(executable_name(spec.name)):
        return bundled, ("this release's hex helper archive (on demand: the first run that needs it "
                         "fetches and verifies it, or `woof fetch-bridges` now)")
    if bundled.is_file():
        if _not_executable(bundled):
            return None, _not_executable_text("this distribution's woof/hex/libexec holds", bundled)
        return bundled, "this distribution's woof/hex/libexec"

    for candidate in gpuwm_candidates(spec):
        candidate = Path(candidate).expanduser()
        if candidate.is_file():
            if _not_executable(candidate):
                return None, _not_executable_text(
                    "woof's bridge directories hold", candidate
                )
            return candidate.resolve(), "woof's bridge directories"

    found = shutil.which(spec.name)
    if found:
        return Path(found), "PATH"
    return None, "not found on any rung"


__all__ = [
    "CONVERT",
    "ENGINES",
    "FETCH_COMMAND",
    "GDT101_REMAP",
    "GEOMETRY",
    "HOSTPREP",
    "INIT",
    "LBC",
    "MET_INTERMEDIATE",
    "RENDERER",
    "STATIC",
    "USER_BRIDGE_DIR",
    "EngineRefusal",
    "bundled_candidate",
    "EngineSpec",
    "chmod_remedy",
    "executable_name",
    "gpuwm_bundles",
    "gpuwm_candidates",
    "locate",
    "remedy",
    "resolution_order",
    "resolve",
]


def static_geog_argv(explicit: str | Path | None) -> list[str]:
    """``--geog ROOT`` for ``rw_mpas_static``: the explicit root, else woof's own ladder.

    The helper's built-in ladder is the one ``gpuwm fetch-geog`` stages into
    (``$GPUWM_WPS_GEOG``, ``$GPUWM_CASE_DATA_ROOT/WPS_GEOG``,
    ``~/.local/share/gpuwm/WPS_GEOG``).  ``woof fetch-geog`` stages into
    ``woof.geog_assets.default_geog_root()`` (``~/.local/share/woof/WPS_GEOG``
    unless the case-data root is set), so a door that left the root to the
    helper refused with "no geography root" beside the archive ``woof
    fetch-geog`` had just staged (WOOF 1.1.0 r13: ``woof hex global`` without
    ``--geog`` on a clean install).  The root comes from
    ``woof.rustwx_static.default_geog_root()``, the ladder ``woof mesh`` reads.
    When that finds nothing no flag is passed, and the helper's own refusal
    names where it looked.
    """

    if explicit:
        return ["--geog", str(Path(explicit).expanduser().absolute())]
    try:
        from woof.rustwx_static import default_geog_root
    except ImportError:
        return []
    found = default_geog_root()
    return ["--geog", str(Path(found).absolute())] if found is not None else []


def refusal_text(stderr: str | None, stdout: str | None = None, *, max_lines: int = 40) -> str:
    """An engine's refusal as it printed it, not just its last line.

    Defect (hex global run, 2026-10-16): the doors kept only the last line of
    ``rw_mpas_static``'s stderr.  Its geography refusal ends with the
    alternative remedy ("--geog DIR pointing at a complete archive"), so the
    door reported that line and dropped the one above it naming the missing
    datasets and ``woof fetch-geog --datasets mesh``.  The engine's refusals
    are a few lines and progress goes to stdout, so the whole of stderr (its
    last ``max_lines`` lines) is the message.
    """

    from .command_names import helper_remedies

    lines = [line.rstrip() for line in helper_remedies(stderr or stdout or "").strip().splitlines()]
    return "\n".join(lines[-max_lines:]) if lines else "no output"
