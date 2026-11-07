"""Reaching ``rw_mpas_static``: the static a generated grid needs to run.

A grid file alone reaches no dycore.  It carries a mesh on a unit sphere
with no terrain, no land use and no soil, and the mesh registry that
admits a mesh to a run pins BOTH files by byte count and sha256.  Until
this bridge existed, ``woof mesh`` produced half of a runnable pair and
said so in its own receipt.

The shape is the one every bridge in this package uses: an environment
override, a deterministic candidate list, a two-question probe
(``--version`` separates a runnable executable from a truncated or
wrong-platform file; ``--abi`` catches a build that launches but speaks a
different argument vector), and a runner that streams the engine's
progress grammar.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

from woof.bridges import (RUSTWX_CRATE_RELATIVE, artifact_remedy,
                           lazy_build_hints,
                           rustwx_build_hint,
                           accept_resolved, executable_name, launchable,
                           quiet_loader_errors)
from woof.rustwx import _candidates, crate_dir

#: Environment variable naming a prebuilt static builder.
STATIC_ENV = "WOOF_RW_MPAS_STATIC"

#: Executable base name of the vendored static builder.
STATIC_NAME = "rw_mpas_static"

#: ``CARGO_BUILD_HINT``: the one-liner that builds it, from a checkout
#: root, spelled for the shell rule when it is read.
__getattr__ = lazy_build_hints(
    __name__, CARGO_BUILD_HINT=RUSTWX_CRATE_RELATIVE)

#: The tools ``rw_mpas_static`` carries besides the static door, as
#: ``rw_mpas_static TOOL ARGS...``.  Each was its own binary until 2.9.0
#: (``rw_mpas_static_bounded``, ``rw_mpas_static_words``,
#: ``rw_mpas_gwd_replace``, ``rw_mpas_gwd_review``, ``rw_mpas_verify``); one
#: executable keeps one copy of the shared code in each platform wheel, and
#: the binary invoked under a former name still runs that tool.
STATIC_TOOLS = ("bounded", "words", "gwd-replace", "gwd-review", "verify")

#: ``rw_mpas_static --abi`` output: the argument vector this wrapper was
#: written against, the progress grammar :func:`run_static` parses, and the
#: tool line.  The tool line is what tells a build that carries the tools
#: from one that predates them: an older build still launches and answers
#: the door's contract, then treats ``verify`` as a static request and
#: refuses for want of ``--grid``.
STATIC_ABI_MARKER = (
    "rw_mpas_static --grid GRID.nc --out STATIC.nc [--geog DIR] "
    "[--legacy-static] [--terrain-only] [--nominal-dx-m M] "
    "[--valid-time YYYY-MM-DD_hh:mm:ss] "
    "[--receipt JSON] [--clobber] "
    "| --compare REFERENCE.nc CANDIDATE.nc [--report JSON]\n"
    "GRID\tNOMINALDX\tVECTORS\tOPERATORS\tGEOGGRID\tGWDBAND\tWROTE\tFINISHED\n"
    "TOOLS\t" + "\t".join(STATIC_TOOLS))

#: The schema tag a 2.9.0 default (native-stream) static carries in its
#: ``rw_mpas_static_schema`` attribute and its receipt.  A consumer that
#: caches statics keys on it: a file without it was written by an older
#: builder from the with-lakes land use, with every lake stored as land,
#: and is rebuilt rather than reused.
STATIC_SCHEMA_TAG = "rw-mpas-static/v4-native"

#: The tag ``--legacy-static`` (and every pre-2.9.0 builder) writes.
LEGACY_STATIC_SCHEMA_TAG = "rw-mpas-static/v3-unified"

#: Geography datasets the builder reads by default.  A root missing any of
#: them cannot produce a static, and saying which one is missing before the
#: mesh relaxation runs is the difference between a five-second refusal
#: and a wasted half hour.
#:
#: The default is MPAS-A v8.4.2's own static (2.9.0): land use from
#: ``modis_landuse_20class_30s``, where a lake is water (category 17), and
#: the gravity-wave-drag statistics over the USGS ``landuse_30s`` water
#: mask, as native ``init_atmosphere`` reads them.  The with-lakes land use
#: is read only by ``--legacy-static`` (:data:`LEGACY_GEOG_DATASETS`).
REQUIRED_GEOG_DATASETS = (
    "topo_gmted2010_30s",
    "modis_landuse_20class_30s",
    "landuse_30s",
    # Soil category and deep-soil temperature: the permissive defaults
    # (2026-10-21), SoilGrids250m's texture_top (inside the soilgrids
    # archive) and the ERA5 climatology fetch-geog builds.  MPAS-A's own
    # STATSGO/FAO tiles and ISLSCP-I soiltemp_1deg are read only when
    # named (--soilcat-data STATSGO, --soiltemp-data ISLSCP).
    "texture_top",
    "soiltemp_era5_1deg",
    "maxsnowalb_modis",
    "greenfrac_fpar_modis",
    "albedo_modis",
    # The Noah-MP soil group, which one static writer read and the other
    # did not.  While the schema was split between two writers these five
    # variables were ABSENT from the file the door produced; the single
    # writer emits them, so the archive that feeds them is required and
    # not optional.  A bare ``woof fetch-geog`` stages all five from one
    # upstream tarball: the pin table scopes that archive to this door
    # (``required_by=("mesh",)``) and the default fetch set is every pin,
    # so nothing here has to be asked for by name.
    "soilcomp",
    "texture_layer1",
    "texture_layer2",
    "texture_layer3",
    "texture_layer4",
)

#: What ``--legacy-static`` reads in place of the two native land-use
#: datasets: the with-lakes MODIS land use, and no USGS mask (its GWD
#: statistics are the legacy ones).
LEGACY_GEOG_DATASETS = tuple(
    "modis_landuse_20class_30s_with_lakes" if name == "modis_landuse_20class_30s"
    else name for name in REQUIRED_GEOG_DATASETS if name != "landuse_30s")

#: Fields the static carries as a bitwise +0 PLACEHOLDER rather than as
#: values.  A consumer overlays the real arrays from the init file; the
#: engine writes only the slot and the zeros, because the MPAS port pins
#: the digest of an exact +0 array at each of these shapes and refuses the
#: file with "placeholder bytes changed" if a derived value is put there.
#:
#: The SLOT is not optional.  ``coeffs_reconstruct`` was left out of the
#: file entirely until a real forecast was attempted, and the port stopped
#: at ``AttributeError: mesh has no 'coeffs_reconstruct'`` before a single
#: device byte was allocated.
PLACEHOLDER_FIELDS = (
    "coeffs_reconstruct",
    "edgeNormalVectors",
    "localVerticalUnitVectors",
    "cellTangentPlane",
)

#: Fields that USED to be absent from the file, and are not any more.
#:
#: The crate carried two static writers with different manifests, and this
#: list was the difference between them: ``cell_gradient_coef_x/y`` and
#: ``defc_a/b`` were computed by one and declared missing by the other,
#: and the soil-composition group was read by one and unknown to the
#: other.  One writer serves the door now and it emits all nine.  They are
#: named here, and in the engine's own ``--help``, so a reader who learnt
#: the old absent list does not go looking for a gap that is not there.
FORMERLY_UNWRITTEN_FIELDS = (
    "cell_gradient_coef_x", "cell_gradient_coef_y",
    "soilcomp", "soilcl1", "soilcl2", "soilcl3", "soilcl4",
)

#: Fields the native static stream does not carry, by MPAS's own registry:
#: the deformation weights and the two land-use aliases belong to the init
#: file, as they do for a published native static.  ``--legacy-static``
#: still writes them.  Not a gap: the port reads ``defc_a``/``defc_b`` from
#: the init file and the mesh binding drops them from a native static by
#: name (see ``woof.mpas_mesh.PORT_BLOCKING_STATIC_FIELDS``).
NATIVE_STREAM_OMITTED_FIELDS = ("defc_a", "defc_b", "lu_index", "soilcat_top")

_PROBE_TIMEOUT_S = 20


def static_candidates() -> tuple[Path, ...]:
    """Deterministic candidate paths for the builder, best first."""

    return _candidates(STATIC_ENV, STATIC_NAME)


def find_static_bin() -> Path | None:
    """First existing candidate, or None.

    An environment override naming a missing file is a hard error rather
    than a fall-through: explicit configuration that silently selects a
    different binary would build a static from geography the operator did
    not choose, and nothing downstream would say so.
    """

    override = os.environ.get(STATIC_ENV)
    for candidate in static_candidates():
        if candidate.is_file():
            return accept_resolved(candidate.resolve())
        if override and candidate == Path(override):
            raise FileNotFoundError(
                f"{STATIC_ENV} names a missing file: {candidate}")
    return None


def static_remedy() -> str:
    """The remedy for a missing builder, true for THIS install."""

    return artifact_remedy(
        env_var=STATIC_ENV, filename=executable_name(STATIC_NAME),
        subject="the mpas static builder",
        crate_relative=RUSTWX_CRATE_RELATIVE,
        one_liner=rustwx_build_hint(), artifact=STATIC_NAME)


def tool_command(tool: str, binary: Path | None = None) -> list[str]:
    """The argument-vector head that runs one of ``rw_mpas_static``'s tools.

    ``binary`` defaults to the resolved builder (:func:`find_static_bin`).
    The binary is probed first, so a build that predates the tools is
    refused with the rebuild remedy rather than run: such a build would
    take ``verify`` for a static request.  A caller appends the tool's own
    arguments, e.g. ``tool_command("verify") + ["--request", r, "--out", o]``.
    """

    if tool not in STATIC_TOOLS:
        raise ValueError(
            f"rw_mpas_static has no tool {tool!r}; it carries "
            f"{', '.join(STATIC_TOOLS)}")
    path = binary if binary is not None else find_static_bin()
    if path is None:
        raise FileNotFoundError(static_remedy())
    ok, evidence = probe_static_bin(Path(path))
    if not ok:
        raise RuntimeError(f"rw_mpas_static at {path}: {evidence}")
    return [str(path), tool]


def probe_static_bin(path: Path) -> tuple[bool, str]:
    """Launch ``path --version``, then ``--abi``; is this build usable?"""

    ok, evidence = launchable(path)
    if not ok:
        return False, f"{evidence} -- corrupt, stale, or built for " \
                      "another platform"
    try:
        with quiet_loader_errors():
            probe = subprocess.run(
                [str(path), "--version"], capture_output=True, text=True,
                errors="replace", timeout=_PROBE_TIMEOUT_S)
    except OSError as error:
        return False, f"exists but failed to execute: {error}"
    except subprocess.TimeoutExpired:
        return False, (f"probe invocation did not exit within "
                       f"{_PROBE_TIMEOUT_S} s")
    transcript = f"{probe.stdout or ''}{probe.stderr or ''}"
    if probe.returncode != 0 or not transcript.startswith("rw_mpas_static "):
        return False, (f"probe --version exited {probe.returncode} without "
                       "its version line -- corrupt, stale, or built for "
                       "another platform")
    try:
        with quiet_loader_errors():
            abi = subprocess.run(
                [str(path), "--abi"], capture_output=True, text=True,
                errors="replace", timeout=_PROBE_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired) as error:
        return False, f"executes but --abi failed: {error}"
    observed = (abi.stdout or "").strip().replace("\r\n", "\n")
    if abi.returncode != 0 or observed != STATIC_ABI_MARKER:
        return False, ("executes but reports a different static-builder ABI "
                       "than this woof expects -- rebuild it: "
                       f"{rustwx_build_hint()}")
    return True, (f"{transcript.strip()} -- --abi matches the static "
                  "request and progress contract")


def geog_root_candidates() -> tuple[Path, ...]:
    """Every place a geography archive is looked for, best first.

    THIS LADDER IS NOT FREE-CHOICE.  It is where ``woof fetch-geog``
    stages, so the door cannot refuse an archive the product itself
    downloaded.  It used to stop after ``$GPUWM_WPS_GEOG`` and
    ``~/.local/share/woof/WPS_GEOG``; that second path is only the
    non-Windows case-data default, so on Windows -- and on any box with
    ``$GPUWM_CASE_DATA_ROOT`` exported -- ``woof mesh`` printed
    "REFUSED: no geography archive" while a complete archive sat at
    ``fetch-geog``'s own default root.  A refusal that names data the
    product staged is a defect, not a refusal.

    ``geog_assets.default_geog_root`` is asked rather than re-spelled, so
    a change to where ``fetch-geog`` writes moves this ladder with it.
    """

    from woof.geog_assets import default_geog_root as staged_geog_root

    candidates: list[Path] = []

    def add(path: Path) -> None:
        if path not in candidates:
            candidates.append(path)

    override = (os.environ.get("GPUWM_WPS_GEOG") or "").strip()
    if override:
        add(Path(override))
    # fetch-geog's default --root, whether $GPUWM_CASE_DATA_ROOT is set or
    # falling back to this platform's case-data root.
    add(staged_geog_root())
    home = os.environ.get("HOME") or os.environ.get("USERPROFILE")
    if home:
        # The historical candidate, kept so a box that staged there before
        # this ladder grew does not stop resolving.
        add(Path(home) / ".local" / "share" / "woof" / "WPS_GEOG")
    return tuple(candidates)


def default_geog_root() -> Path | None:
    """Where a geography archive lives when the caller did not say.

    Same ladder the builder itself walks, so the door's refusal and the
    engine's behaviour cannot disagree about which archive was chosen.
    """

    for candidate in geog_root_candidates():
        if candidate.is_dir():
            return candidate
    return None


def geog_dataset_candidates(root: Path, name: str) -> tuple[Path, ...]:
    """Every directory that may serve dataset `name` under `root`, best first.

    Two layouts are real, and both come from ``woof fetch-geog``.  A
    single-dataset archive unpacks to ``root/<name>``.  A container
    archive unpacks to ``root/<archive>/<name>``: its pin declares the
    children in ``index_subdirs`` and the fetcher validates each child's
    index exactly there.  This lookup used to know the first layout
    only, so a tree the product had just staged, precisely as designed,
    read as five datasets short -- doctor's own remedy could not turn
    its row green and ``woof mesh`` refused the static half of the
    pair on every install that followed it.

    Read off the pin table.  No archive name is spelled here, so a
    future container is one row in
    :data:`woof.geog_assets.GEOG_ARCHIVES` and nothing else, and the
    fetcher, doctor and the door cannot disagree about where a child
    lives.  The flat path comes first so a hand-built archive keeps
    winning over the staged one when a box carries both.
    """

    from woof.geog_assets import GEOG_ARCHIVES

    candidates = [root / name]
    for archive in GEOG_ARCHIVES:
        if name in archive.index_subdirs:
            candidates.append(root / archive.dataset / name)
    return tuple(candidates)


def geog_dataset_dir(root: Path, name: str) -> Path | None:
    """The directory that serves `name` under `root`, or None.

    Served means carrying its WPS ``index`` file; an empty directory
    left by an interrupted extraction is not a dataset.
    """

    for candidate in geog_dataset_candidates(root, name):
        if (candidate / "index").is_file():
            return candidate
    return None


def missing_geog_datasets(root: Path) -> list[str]:
    """Which required datasets `root` does not carry, in request order.

    Both staged layouts count (:func:`geog_dataset_candidates`); the
    engine resolves each dataset the same way, so a tree this calls
    complete is one the builder reads.
    """

    return [name for name in REQUIRED_GEOG_DATASETS
            if geog_dataset_dir(root, name) is None]


def fetchable_for(datasets) -> list[str]:
    """The ``fetch-geog`` dataset names that stage `datasets`.

    Not the same list.  One upstream archive carries seven directories,
    each with its own WPS index, so "fetch soilcomp" is not a command
    anybody can run and "fetch soilgrids" is.  The mapping is read off
    the pin table rather than spelled here, so a future multi-dataset
    product stays table work.
    """

    from woof.geog_assets import GEOG_ARCHIVES

    wanted = set(datasets)
    out: list[str] = []
    for archive in GEOG_ARCHIVES:
        served = set(archive.index_subdirs) or {archive.dataset}
        if served & wanted and archive.dataset not in out:
            out.append(archive.dataset)
    return out or sorted(wanted)


def geog_refusal(root: Path | None) -> str | None:
    """The refusal for a geography archive that cannot build a static.

    None when the archive is complete.  Names the concrete breakage: a
    grid file alone is refused by the mesh registry, so a run that gets
    only a grid stops at the door rather than at the dycore.
    """

    if root is None:
        looked = "\n".join(f"    {path}" for path in geog_root_candidates())
        return (
            "no geography archive, so the static half of the pair cannot be "
            "built. A grid file on its own reaches no dycore: the mesh "
            "registry pins BOTH the grid and a matching static by byte count "
            "and sha256, and refuses a mesh that has only one of them.\n"
            "looked here, and none of them is a directory:\n"
            f"{looked}\n"
            "what does work:\n"
            "  woof fetch-geog, which stages into the default geography path\n"
            "  --geog DIR pointing at a WPS_GEOG archive\n"
            "  set GPUWM_WPS_GEOG to an archive that is already on this box\n"
            "  --no-static to write the grid alone and accept that it is not "
            "runnable yet")
    missing = missing_geog_datasets(root)
    if not missing:
        return None
    return (
        f"the geography archive at {root} is missing {len(missing)} dataset(s) "
        f"a static needs: {', '.join(missing)}. Terrain, land use, soil class "
        "and composition, deep-soil temperature, green-ness and albedo all "
        "come from these directories, and a static written without one would "
        "hand the model a field of zeros that the run would carry as real "
        "geography.\n"
        "what does work:\n"
        f"  woof fetch-geog, whose default set stages every one of them "
        f"into {root}\n"
        f"    (that includes {', '.join(fetchable_for(missing))}; "
        "`--datasets mesh` narrows the download to what this door reads)\n"
        "  --geog DIR pointing at a complete archive\n"
        "  --no-static to write the grid alone and accept that it is not "
        "runnable yet")


class StaticEngineError(RuntimeError):
    """The builder refused, or could not be run."""


def run_static(binary: Path, argv: list[str], *,
               on_progress=None) -> dict:
    """Run the builder, streaming its progress lines, and return a receipt.

    The receipt is read back off disk from ``--receipt`` when one was
    asked for, so the caller's document is the engine's own measurement
    rather than a re-derivation of it.
    """

    try:
        with quiet_loader_errors():
            process = subprocess.Popen(
                [str(binary), *argv], stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, errors="replace")
    except OSError as error:
        raise StaticEngineError(
            f"{binary} could not be launched: {error}") from error

    lines: list[str] = []
    assert process.stdout is not None
    for raw in process.stdout:
        line = raw.rstrip("\r\n")
        lines.append(line)
        if on_progress is not None and "\t" in line:
            on_progress(line)
    stderr = (process.stderr.read() if process.stderr else "") or ""
    code = process.wait()
    if code != 0:
        # the builder names the command of the distribution it was written for
        raise StaticEngineError(stderr.strip().replace("gpuwm fetch-geog", "woof fetch-geog") or
                                f"{binary} exited {code} without a message")

    document: dict = {"progress": lines}
    for line in lines:
        if line.startswith("WROTE\t"):
            parts = line.split("\t")
            if len(parts) >= 5:
                document["static"] = {
                    "path": parts[1], "bytes": int(parts[2]),
                    "variables": int(parts[3]), "sha256": parts[4]}
        elif line.startswith("NOMINALDX\t"):
            parts = line.split("\t")
            if len(parts) >= 4:
                document["nominal_dx"] = {
                    "metres": float(parts[1]), "f32": float(parts[2]),
                    "f32_bits": parts[3]}
        elif line.startswith("RECEIPT\t"):
            path = Path(line.split("\t", 1)[1])
            if path.is_file():
                try:
                    document["receipt"] = json.loads(
                        path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    pass
    return document


__all__ = [
    "CARGO_BUILD_HINT", "REQUIRED_GEOG_DATASETS", "STATIC_ABI_MARKER",  # noqa: F822 -- module __getattr__
    "LEGACY_GEOG_DATASETS", "LEGACY_STATIC_SCHEMA_TAG", "STATIC_SCHEMA_TAG",
    "NATIVE_STREAM_OMITTED_FIELDS",
    "STATIC_ENV", "STATIC_NAME", "StaticEngineError",
    "PLACEHOLDER_FIELDS", "FORMERLY_UNWRITTEN_FIELDS", "fetchable_for",
    "crate_dir", "default_geog_root", "find_static_bin",
    "geog_dataset_candidates", "geog_dataset_dir", "geog_refusal",
    "missing_geog_datasets", "probe_static_bin", "run_static",
    "static_candidates", "static_remedy",
]
