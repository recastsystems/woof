"""Locate and drive ``rw_ensbatch`` and ``rw_obsgrid``.

The render law says weather-field product plots come from the real Rust
renderer.  Two families of weather field had no Rust renderer to come
from, so three modules in this tree drew them with matplotlib and said so
in their own docstrings:

* **ensemble products** -- ``woof/da/enprod.py``: *"the vendored rust
  renderer's catalog has no ensemble entries ... there is no second
  engine to switch to"*;
* **gridded radar observations** -- ``tools/da_level2_render.py`` and
  ``tools/da_nowcast_render.py``, which already borrow the Rust
  renderer's assets and palette while re-implementing its fill in
  matplotlib.  Both now carry ``--engine`` and resolve it through the one
  :func:`resolve_obsgrid_engine`.  ``da_level2_render`` DRIVES
  ``rw_obsgrid`` for the five single-panel products and names the reason
  in one sentence when it falls back instead.  ``da_nowcast_render``
  composes its sheets itself and drives no engine yet, so it prints the
  DEPRECATED FALLBACK sentence every run and adds the resolution reason
  beside it when there is one.  What keeps their own figures is the
  SHAPE, not the science: three of the four are side-by-side or
  per-radar GRIDS of panels and the engine composes one, so the sheets
  stay until there is a compositor to tile the engine's PNGs with.

``rw_ensbatch`` and ``rw_obsgrid`` are those two engines.  This module is
the seam the three callers reach them through, built exactly like
:mod:`woof.rustwx` and :mod:`woof.rustwx_fetch`: the same five-rung
resolution ladder, the same ``--abi`` contract handshake (a binary that
launches is not a binary that speaks your grammar), the same
event-line parsing.

Neither binary is a replacement for ``rw_wrfbatch``: its
one-positional-wrfout contract is untouched, and a caller with one
deterministic run still drives it.  What these add is the two input
shapes that contract cannot express -- N members at ONE valid time, and
a file of observations that is not a forecast.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from woof import bridges
from woof.bridges import (RUSTWX_CRATE_RELATIVE, artifact_remedy,
                           lazy_build_hints,
                           rustwx_build_hint,
                           executable_name)
from woof.rustwx import _candidates, crate_dir

#: Environment variables naming prebuilt binaries.
ENSEMBLE_ENV = "WOOF_RW_ENSBATCH"
OBSGRID_ENV = "WOOF_RW_OBSGRID"
COMPARE_ENV = "WOOF_RW_COMPARE"

#: Executable base names.
ENSEMBLE_NAME = "rw_ensbatch"
OBSGRID_NAME = "rw_obsgrid"
#: The comparison engine: a run's products beside a reference model's own
#: fields for the same cycle and lead (``woof render --compare``).  A
#: third sibling for the same reason the other two exist: two inputs of
#: two formats on one grid is neither one run's frames nor N members.
COMPARE_NAME = "rw_compare"

#: The nowcast frames converter: a 2D reflectivity NetCDF nowcast written
#: out as the ``gpuwm-obs.nowcast-frames.v1`` contract
#: (``docs/nowcast-frames.md``), the only form the comparison engine's
#: ``frames:`` panel reads.  Same workspace and same ladder as the three
#: engines above; a converter rather than a renderer, so it is not a flag
#: on ``rw_compare``.
NOWCAST_FRAMES_ENV = "WOOF_RW_NOWCAST_FRAMES"
NOWCAST_FRAMES_NAME = "rw_nowcast_frames"

#: ``CARGO_BUILD_HINT``: the one-liner that builds them.  Same workspace
#: as the batch renderer, so one cargo invocation produces all three.
#: Spelled for the shell rule when it is read.
__getattr__ = lazy_build_hints(
    __name__, CARGO_BUILD_HINT=RUSTWX_CRATE_RELATIVE)

#: The exact ``--abi`` lines these wrappers were written against.
#:
#: Same discipline as :data:`woof.rustwx.RENDERER_ABI_MARKER`, and for
#: the same reason: "does the binary start" is not a contract check.  What
#: is listed is the vocabulary the PYTHON half parses -- the product names
#: it may pass and the event words it reads -- so changing either changes
#: the literal and every older build fails the handshake instead of
#: quietly answering the old grammar.
ENSEMBLE_ABI_MARKER = (
    "gpuwm-rw-ensbatch-products-v1\tmean\tspread\tprob\tpmm\tpaintball\tmax\t"
    "gpuwm-rw-ensbatch-fields-v1\tFIELD\tname\ttitle\tunits\t"
    "default_threshold\tunit_slug\t"
    "gpuwm-rw-ensbatch-events-v1\tRENDERED\tSKIPPED\tFAILED\t"
    "gpuwm-rw-ensbatch-vocabulary-v1\tMEMBERS\tCOVERAGE\tTIES")

OBSGRID_ABI_MARKER = (
    "gpuwm-rw-obsgrid-products-v1\tz-composite\tcoverage-depth\t"
    "radar-overlap\tvr-lowest\tradar-contribution\t"
    "gpuwm-rw-obsgrid-events-v1\tRENDERED\tSKIPPED\tFAILED\t"
    "gpuwm-rw-obsgrid-vocabulary-v1\tOBSERVED\tSITES")

#: What this wrapper parses from ``rw_compare``: the listing rows and the
#: event words.  The references and products themselves are NOT listed
#: here -- they are the engine's tables, asked with ``--list-products``.
COMPARE_ABI_MARKER = (
    "gpuwm-rw-compare-references-v1\tREFERENCE\tname\tlabel\t"
    "gpuwm-rw-compare-products-v1\tPRODUCT\tname\tslug\ttitle\t"
    "difference\t"
    "gpuwm-rw-compare-events-v1\tRENDERED\tSKIPPED\tFAILED\tSTATS\t"
    "MATCH\tSOURCE\tFINISHED\t"
    "gpuwm-rw-compare-presentation-v1\t--theme")

#: The converter's ``--abi`` line: its one subcommand and every flag this
#: wrapper may pass.  Bound to the Rust ``ABI_MARKER`` constant and to
#: :data:`woof.bridges.BRIDGE_ABI_MARKERS` by tests.
NOWCAST_FRAMES_ABI_MARKER = (
    "gpuwm-rw-nowcast-frames-v1\tfrom-netcdf\t--netcdf\t--out\t"
    "--source-id\t--lattice\t--variable\t--lat\t--lon\t--issue\t"
    "--causal\t--receipt\t--source-kind\t--package\t--revision")

#: Ensemble products, in the order ``all`` expands to.
ENSEMBLE_PRODUCTS = ("mean", "spread", "prob", "pmm", "paintball", "max")

#: Ensemble fields, as of the build this wrapper was written against.
#:
#: NOT the authority any more, and the difference matters: the engine
#: answers ``--list-fields`` with each field's name, title, units,
#: default threshold and unit slug, and
#: :func:`list_ensemble_fields` reads it.  What stays here is the
#: no-engine fallback -- the vocabulary a message can name when there is
#: no binary to ask -- so a caller states a list it can prove rather
#: than none at all.  A field added to ``ensbatch.rs`` is reachable
#: without touching this tuple.
ENSEMBLE_FIELDS = ("refl", "uh", "precip", "t2", "wspd10", "visibility", "fog_prob")

#: Observation-grid products.
OBSGRID_PRODUCTS = ("z-composite", "coverage-depth", "radar-overlap",
                    "vr-lowest", "radar-contribution")

_PROBE_TIMEOUT_S = 20


def _find(env_var: str, name: str) -> Path | None:
    """First existing candidate, or None.

    An environment override that names a missing file is a hard error
    naming the three ways out, exactly as :func:`woof.rustwx
    .find_renderer` does: explicit configuration must fail loudly rather
    than fall through to a matplotlib fallback the operator did not ask
    for.
    """

    override = os.environ.get(env_var)
    for candidate in _candidates(env_var, name):
        if candidate.is_file():
            return bridges.accept_resolved(candidate.resolve())
        if override and candidate == Path(override):
            raise FileNotFoundError(
                f"{env_var} names a missing file: {candidate}.  Point it "
                f"at a built {name} binary, unset {env_var} to use the "
                f"vendored resolution ladder, or build it with: "
                f"{rustwx_build_hint()}")
    return None


def find_ensemble_bin() -> Path | None:
    return _find(ENSEMBLE_ENV, ENSEMBLE_NAME)


def find_obsgrid_bin() -> Path | None:
    return _find(OBSGRID_ENV, OBSGRID_NAME)


def find_compare_bin() -> Path | None:
    return _find(COMPARE_ENV, COMPARE_NAME)


def compare_remedy() -> str:
    return artifact_remedy(
        env_var=COMPARE_ENV, filename=executable_name(COMPARE_NAME),
        subject="the rust comparison engine",
        crate_relative=RUSTWX_CRATE_RELATIVE, one_liner=rustwx_build_hint())


def probe_compare_bin(path: Path) -> tuple[bool, str]:
    return _probe(path, COMPARE_ABI_MARKER, COMPARE_NAME, COMPARE_ENV)


def require_compare_bin() -> Path:
    """The comparison engine, resolved and proven, or a refusal.

    There is no second engine on this lane and none is offered: a
    comparison sheet is weather fields end to end, so a missing or
    foreign ``rw_compare`` is refused with the line that builds it
    rather than drawn some other way.
    """

    try:
        path = find_compare_bin()
    except FileNotFoundError as error:
        raise RuntimeError(f"{error}  {compare_remedy()}") from error
    if path is None:
        raise RuntimeError(
            f"comparison sheets come from the rust engine {COMPARE_NAME}, "
            f"and it is not built.  Build it: {rustwx_build_hint()}  "
            f"{compare_remedy()}")
    usable, evidence = probe_compare_bin(path)
    if not usable:
        raise RuntimeError(f"{path}: {evidence}")
    return path


def probe_compare_reference_bin(path: Path) -> tuple[bool, str]:
    """Does ``path`` speak the reference-input contract?

    The SECOND door through ``rw_compare``: the reference panels beside a
    run's observation verification call ``--fetch-reference`` and parse
    a ``gpuwm.reference-input.v1`` receipt, which the ``--abi`` line
    :func:`probe_compare_bin` reads does not name.  The two contracts
    landed on parallel branches, so a build can carry either one alone:
    the ``--abi`` probe passes a build that has no ``--fetch-reference``,
    and this one passes a build whose ``--abi`` predates the theme line.
    Judged statically out of the bytes against
    :data:`woof.bridges.BRIDGE_ABI_MARKERS`, the check
    ``woof fetch-bridges`` applies before it stages one.
    """

    return bridges.bridge_abi_matches(COMPARE_NAME, path)


def require_compare_reference_bin() -> Path:
    """The comparison engine for the reference panels, or a refusal.

    Resolved through :func:`find_compare_bin`, the ladder ``render
    --compare`` uses, and judged by :func:`probe_compare_reference_bin`,
    the probe ``woof doctor`` reports for this door.  Breakage the
    refusal prevents: a build without ``--fetch-reference`` used to be
    launched anyway, rejected the flag, and surfaced as a bare
    ``reference input:`` usage error that did not say the build was
    stale or how to replace it.  A missing override still raises
    :class:`FileNotFoundError`, as the ladder does for every door.
    """

    path = find_compare_bin()
    if path is None:
        raise RuntimeError(artifact_remedy(
            env_var=COMPARE_ENV, filename=executable_name(COMPARE_NAME),
            subject="the reference renderer",
            crate_relative=RUSTWX_CRATE_RELATIVE,
            one_liner=rustwx_build_hint(), artifact=COMPARE_NAME))
    usable, evidence = probe_compare_reference_bin(path)
    if not usable:
        raise RuntimeError(
            f"{path}: {evidence}.  Rebuild it: {rustwx_build_hint()}")
    return path


def find_nowcast_frames_bin() -> Path | None:
    return _find(NOWCAST_FRAMES_ENV, NOWCAST_FRAMES_NAME)


def nowcast_frames_remedy() -> str:
    return artifact_remedy(
        env_var=NOWCAST_FRAMES_ENV,
        filename=executable_name(NOWCAST_FRAMES_NAME),
        subject="the nowcast frames converter",
        crate_relative=RUSTWX_CRATE_RELATIVE, one_liner=rustwx_build_hint(),
        artifact=NOWCAST_FRAMES_NAME)


def probe_nowcast_frames_bin(path: Path) -> tuple[bool, str]:
    return _probe(path, NOWCAST_FRAMES_ABI_MARKER, NOWCAST_FRAMES_NAME,
                  NOWCAST_FRAMES_ENV)


def require_nowcast_frames_bin() -> Path:
    """The nowcast frames converter, resolved and proven, or a refusal.

    Breakage the refusal prevents: a build from another checkout whose
    ``--abi`` line differs would accept or reject a flag this wrapper
    passes and write a root the comparison sheets read under another
    contract.  There is no Python converter to fall back to; the frames
    are written in Rust or not at all.
    """

    try:
        path = find_nowcast_frames_bin()
    except FileNotFoundError as error:
        raise RuntimeError(f"{error}  {nowcast_frames_remedy()}") from error
    if path is None:
        raise RuntimeError(
            f"NetCDF nowcasts become frames roots through the rust "
            f"converter {NOWCAST_FRAMES_NAME}, and it is not built or "
            f"staged.  {nowcast_frames_remedy()}")
    usable, evidence = probe_nowcast_frames_bin(path)
    if not usable:
        raise RuntimeError(f"{path}: {evidence}")
    return path


def ensemble_remedy() -> str:
    return artifact_remedy(
        env_var=ENSEMBLE_ENV, filename=executable_name(ENSEMBLE_NAME),
        subject="the rust ensemble product engine",
        crate_relative=RUSTWX_CRATE_RELATIVE, one_liner=rustwx_build_hint())


def obsgrid_remedy() -> str:
    return artifact_remedy(
        env_var=OBSGRID_ENV, filename=executable_name(OBSGRID_NAME),
        subject="the rust observation-grid engine",
        crate_relative=RUSTWX_CRATE_RELATIVE, one_liner=rustwx_build_hint())


def _probe(path: Path, marker: str, name: str,
           env_var: str) -> tuple[bool, str]:
    """``--abi``: is this binary runnable, and does it speak our grammar?

    The contract line is compared exactly.  A ``GPUWM_BRIDGE_SOURCE_REV``
    line after it (``rw_ensbatch`` prints one) is not part of the
    contract: it is judged by the shared stamp check, so a build from
    another checkout is still refused, naming both revisions.
    """

    launchable, why = bridges.launchable(path)
    if not launchable:
        return False, why
    try:
        probe = subprocess.run([str(path), "--abi"], capture_output=True,
                               text=True, errors="replace",
                               timeout=_PROBE_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError) as error:
        return False, f"{name} --abi did not run: {error}"
    from woof import provenance_gate

    answer = (probe.stdout or "").strip()
    contract, revision = provenance_gate.split_abi_answer(answer)
    if probe.returncode != 0 or contract != marker:
        detail = answer or f"exit {probe.returncode} with no --abi line"
        return False, (
            f"launches, but --abi does not match the contract this woof "
            f"expects ({detail}) -- it is a build from another checkout.  "
            f"REBUILD it from this one: {rustwx_build_hint()}")
    if revision is not None:
        refusal = provenance_gate.abi_revision_refusal(
            path, revision, env_var=env_var)
        if refusal is not None:
            return False, (
                f"launches and --abi matches the contract, but {refusal}.  "
                f"REBUILD it from this one: {rustwx_build_hint()}")
    return True, "--abi matches the contract"


def resolve_lane_engine(request: str, *, find, probe, name: str,
                        remedy: str, workaround: str) -> tuple[str, str]:
    """``(engine, why)`` for one lane's ``--engine``; THE contract.

    Three answers and one law, spelled once so two lanes cannot spell it
    differently.  ``matplotlib`` is selectable BY NAME only and is never
    probed: a caller who names it is choosing the documented workaround.
    ``rust`` that cannot be honoured RAISES, because naming an engine is
    a statement about which one must draw.  And ``auto`` has exactly two
    answers, rust or a refusal -- it does not degrade.

    That last clause is the render law, not a preference: weather-field
    plots come from the real engine, the law permits one fallback, and
    this is not it.  A degradation here drew the same fields with the
    other engine and exited 0, so nothing in the run said the pictures
    were not the engine's.

    The refusal names the breakage and the way out in one sentence each,
    which is what makes it a refusal rather than an absence.
    """

    if request == "matplotlib":
        return "matplotlib", "requested"
    try:
        path = find()
    except FileNotFoundError as error:
        raise RuntimeError(_lane_refusal(name, str(error), remedy, workaround)) from error
    if path is None:
        raise RuntimeError(_lane_refusal(
            name, f"{name} is not built", remedy, workaround))
    usable, evidence = probe(path)
    if not usable:
        raise RuntimeError(_lane_refusal(
            name, f"{path}: {evidence}", remedy, workaround))
    return "rust", str(path)


def _lane_refusal(name: str, failure: str, remedy: str,
                  workaround: str) -> str:
    """One refusal sentence, plus the mechanism behind it."""

    from woof.explain import layered

    return layered(
        f"weather-field panels on this lane come from the rust engine "
        f"{name}, and it could not be resolved: {failure}.  Build it "
        f"({rustwx_build_hint()}), or draw them with {workaround}, which is "
        f"a named workaround and reports itself as one.",
        f"The render law permits ONE fallback and this is not it: an "
        f"--engine auto that quietly drew the same weather fields with "
        f"matplotlib exited 0, so nothing in the run said which engine "
        f"made the pictures.  {remedy}")


def probe_ensemble_bin(path: Path) -> tuple[bool, str]:
    return _probe(path, ENSEMBLE_ABI_MARKER, ENSEMBLE_NAME, ENSEMBLE_ENV)


def probe_obsgrid_bin(path: Path) -> tuple[bool, str]:
    return _probe(path, OBSGRID_ABI_MARKER, OBSGRID_NAME, OBSGRID_ENV)


def list_ensemble_fields(engine: Path) -> dict[str, dict]:
    """The ENGINE's field vocabulary: name -> title/units/threshold/slug.

    Asked, never transcribed.  A hand-kept copy of this table had ``t2``
    at 30 in degrees Celsius while the engine drew it at 303 in kelvin,
    so one panel's threshold meant two different temperatures depending
    on which half of the pair you read.

    Empty when the engine cannot be asked, which a caller states rather
    than treating as an empty vocabulary.
    """

    try:
        result = subprocess.run(
            [str(engine), "--list-fields"], capture_output=True, text=True,
            errors="replace", timeout=_PROBE_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError):
        return {}
    if result.returncode != 0:
        return {}
    fields: dict[str, dict] = {}
    for line in (result.stdout or "").splitlines():
        parts = line.split("\t")
        if parts[0] != "FIELD" or len(parts) < 4:
            continue
        row = {"title": parts[2], "units": parts[3]}
        for extra in parts[4:]:
            key, _, value = extra.partition("=")
            if key == "default_threshold":
                try:
                    row["default_threshold"] = float(value)
                except ValueError:
                    continue
            elif key == "default_thresholds":
                try:
                    row["default_thresholds"] = tuple(float(v) for v in value.split(","))
                except ValueError:
                    continue
            elif key == "unit_slug":
                row["unit_slug"] = value
        fields[parts[1]] = row
    return fields


def resolve_ensemble_engine(request: str) -> tuple[str, str]:
    """The ensemble lane's ``--engine``, through the shared contract."""

    return resolve_lane_engine(
        request, find=find_ensemble_bin, probe=probe_ensemble_bin,
        name=ENSEMBLE_NAME, remedy=ensemble_remedy(),
        workaround="--engine matplotlib")


def resolve_obsgrid_engine(request: str) -> tuple[str, str]:
    """The observation-grid lane's ``--engine``, through the same one.

    Its matplotlib route is not merely a second drawing of the same
    panel: two of the tools that own it compose SHEETS (side-by-side and
    per-radar grids) the single-panel engine cannot lay out, so the
    fallback is reached by name, with the reason printed, and the sheets
    are otherwise composed by tiling the engine's own PNGs.
    """

    return resolve_lane_engine(
        request, find=find_obsgrid_bin, probe=probe_obsgrid_bin,
        name=OBSGRID_NAME, remedy=obsgrid_remedy(),
        workaround="--engine matplotlib")


def ensemble_workaround_notice() -> str:
    """What the matplotlib ensemble route must say, every run.

    Fixed means default-on; an opt-in route is a workaround and reports
    itself as one.  The numbers are this lane's own fields and
    products, not the wrfout catalog's.
    """

    return (f"WORKAROUND: these panels are drawn by matplotlib, not by "
            f"{ENSEMBLE_NAME}.  The rust ensemble lane covers the same "
            f"{len(ENSEMBLE_FIELDS)} fields "
            f"({', '.join(ENSEMBLE_FIELDS)}) by "
            f"{len(ENSEMBLE_PRODUCTS)} products "
            f"({', '.join(ENSEMBLE_PRODUCTS)}), so nothing is gained by "
            f"drawing them here; build it with {rustwx_build_hint()} and "
            f"drop --engine matplotlib.")


def _renderer_env() -> dict[str, str]:
    """Subprocess environment: the basemap assets, as the renderer needs."""

    from woof import rustwx

    return rustwx.renderer_env()


def _read_events(stdout: str, stderr: str, subject: str
                 ) -> tuple[list[Path], list[str], list[tuple[str, str]]]:
    """``RENDERED`` / ``SKIPPED`` / ``FAILED``, all three read.

    All three, because reading only two is exactly the defect
    :func:`woof.rustwx.run_renderer`'s docstring records: a product the
    engine skipped for a stated reason became invisible to every caller, and a
    reader saw one image fewer than the catalog with nothing saying why.
    """

    written: list[Path] = []
    failures: list[str] = []
    skipped: list[tuple[str, str]] = []
    for line in (stdout or "").splitlines():
        if line.startswith("RENDERED "):
            _, _, rest = line.partition(" ")
            _, _, path = rest.partition(" ")
            if path:
                written.append(Path(path))
        elif line.startswith("SKIPPED "):
            slug, _, reason = line[len("SKIPPED "):].partition(" ")
            skipped.append((slug, f"{subject}: {reason or 'no reason given'}"))
    from woof import rustwx

    rustwx.relay_native_warnings(stderr)
    for line in (stderr or "").splitlines():
        if line.startswith("FAILED "):
            failures.append(f"{subject}: {line[len('FAILED '):]}")
    return written, failures, skipped


def run_ensemble_renderer(
        engine: Path, manifest: Path, *, store_root: Path, out_dir: Path,
        field: str = "refl", products: str = "mean,spread,prob,pmm,paintball,max",
        threshold: float | None = None, neighborhood_km: float = 0.0,
        frames: int | None = None, nan_policy: str = "mask",
        pmm_tie_rule: str = "flat-index",
        accept_status: str | None = None,
        width: int = 1200, height: int = 900,
        source_label: str | None = None,
        overlays: Path | None = None, annotate: Path | None = None,
        members: dict[int, Path] | None = None,
        domain: str | None = None,
        ) -> tuple[list[Path], list[str], list[tuple[str, str]], dict]:
    """One ensemble batch; ``(written, failures, skipped, report)``.

    ``report`` carries what the panels stamp and a caller may want in a
    provenance record: the member roster, the missingness/coverage
    counts, the PMM tie statistics, and the paintball member->colour
    map.  It is read off the engine's own stdout rather than recomputed,
    so a caption and a record cannot disagree.

    ``members`` values may be one wrfout or the member DIRECTORY holding
    its frames; a member written with ``frames_per_outfile = 1`` is a
    series of single-frame files and all of them are that member, which
    is what ``--frames`` indexes into.
    """

    command = [
        str(engine),
        "--store-root", str(store_root),
        "--out-dir", str(out_dir),
        "--manifest", str(manifest),
        "--field", field,
        "--products", products,
        "--nan-policy", nan_policy,
        "--pmm-tie-rule", pmm_tie_rule,
        "--width", str(width),
        "--height", str(height),
    ]
    if threshold is not None:
        command.extend(("--threshold", repr(float(threshold))))
    if neighborhood_km:
        command.extend(("--neighborhood-km", repr(float(neighborhood_km))))
    if frames is not None:
        command.extend(("--frames", str(frames)))
    if accept_status:
        command.extend(("--accept-status", accept_status))
    if source_label:
        command.extend(("--source-label", source_label))
    if overlays is not None:
        command.extend(("--overlays", str(overlays)))
    if annotate is not None:
        command.extend(("--annotate", str(annotate)))
    # Which nest to reduce, when a member directory holds more than one.
    # Every member is selected the same way by the same token, which is
    # the disambiguation the engine's own ambiguity refusal asks for.
    if domain:
        command.extend(("--domain", str(domain)))
    for number, path in sorted((members or {}).items()):
        command.extend(("--member", f"{number}={path}"))

    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                errors="replace", env=_renderer_env())
    except OSError as error:
        return [], [f"{manifest}: {ENSEMBLE_NAME} failed to launch: {error}"], \
            [], {}
    written, failures, skipped = _read_events(
        result.stdout, result.stderr, str(manifest))
    report: dict = {}
    # The RENDERED event's SLUG as well as its path.  The shared event
    # reader answers "what was written"; a caller that has to FILE those
    # pictures needs to know which product each one is, and the engine
    # already says so on the same line.
    report["rendered"] = [
        (line[len("RENDERED "):].partition(" ")[0],
         Path(line[len("RENDERED "):].partition(" ")[2]))
        for line in (result.stdout or "").splitlines()
        if line.startswith("RENDERED ") and line[len("RENDERED "):].partition(" ")[2]]
    for line in (result.stdout or "").splitlines():
        if line.startswith("MEMBERS "):
            report["members"] = line[len("MEMBERS "):]
        elif line.startswith("COVERAGE "):
            report["coverage"] = line[len("COVERAGE "):]
        elif line.startswith("TIES "):
            report["pmm_ties"] = line[len("TIES "):]
        elif line.startswith("PAINTBALL_LEGEND\t"):
            report["paintball_legend"] = line.split("\t", 1)[1]
    if result.returncode != 0 and not failures:
        failures.append(
            f"{manifest}: {engine_failure_reason(result.stderr, result.returncode)}")
    return written, failures, skipped, report


def engine_failure_reason(stderr: str, returncode: int) -> str:
    """The line that says why an engine exited non-zero.

    ``rw_ensbatch`` answers an argument or roster problem with the reason
    and then its two-line usage text, so the last line of stderr is the
    usage line.  Reported as the failure, it told an operator nothing (a
    v2 ensemble's refused roster read as "rw_ensbatch --list-fields |
    --help | --abi").  The usage text is cut off and the last line before
    it is the reason.
    """

    lines = [line.strip() for line in (stderr or "").splitlines()
             if line.strip()]
    for index, line in enumerate(lines):
        if line.startswith("usage:"):
            lines = lines[:index]
            break
    return lines[-1] if lines else f"exit {returncode} with no message"


def run_obsgrid_renderer(
        engine: Path, obs: Path, *, out_dir: Path,
        products: str = "all", radar: int | None = None,
        rings_km: str | None = None, sites: bool = False,
        width: int = 1200, height: int = 900,
        source_label: str | None = None,
        overlays: Path | None = None, annotate: Path | None = None,
        ) -> tuple[list[Path], list[str], list[tuple[str, str]], list[dict]]:
    """One observation-grid batch; ``(written, failures, skipped, sites)``.

    The fourth element is the site roster the engine read out of the
    file -- ``[{"index", "id", "lat", "lon"}, ...]`` -- so a caller
    composing a gallery can label panels without re-opening the NetCDF.
    """

    command = [
        str(engine),
        "--obs", str(obs),
        "--out-dir", str(out_dir),
        "--products", products,
        "--width", str(width),
        "--height", str(height),
    ]
    if radar is not None:
        command.extend(("--radar", str(radar)))
    if rings_km:
        command.extend(("--rings-km", rings_km))
    if sites:
        command.append("--sites")
    if source_label:
        command.extend(("--source-label", source_label))
    if overlays is not None:
        command.extend(("--overlays", str(overlays)))
    if annotate is not None:
        command.extend(("--annotate", str(annotate)))

    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                errors="replace", env=_renderer_env())
    except OSError as error:
        return [], [f"{obs}: {OBSGRID_NAME} failed to launch: {error}"], [], []
    written, failures, skipped = _read_events(
        result.stdout, result.stderr, str(obs))
    roster: list[dict] = []
    for line in (result.stdout or "").splitlines():
        if line.startswith("SITES\t"):
            parts = line.split("\t")
            if len(parts) == 5:
                roster.append({"index": int(parts[1]), "id": parts[2],
                               "lat": float(parts[3]),
                               "lon": float(parts[4])})
    if result.returncode != 0 and not failures:
        # rw_obsgrid prints its usage after an argument error too.
        failures.append(
            f"{obs}: {engine_failure_reason(result.stderr, result.returncode)}")
    return written, failures, skipped, roster


__all__ = [
    "CARGO_BUILD_HINT", "COMPARE_ABI_MARKER", "COMPARE_ENV", "COMPARE_NAME",  # noqa: F822 -- module __getattr__
    "compare_remedy", "find_compare_bin", "probe_compare_bin",
    "probe_compare_reference_bin", "require_compare_bin",
    "require_compare_reference_bin",
    "ENSEMBLE_ABI_MARKER", "ENSEMBLE_ENV",
    "NOWCAST_FRAMES_ABI_MARKER", "NOWCAST_FRAMES_ENV", "NOWCAST_FRAMES_NAME",
    "find_nowcast_frames_bin", "nowcast_frames_remedy",
    "probe_nowcast_frames_bin", "require_nowcast_frames_bin",
    "ENSEMBLE_FIELDS", "ENSEMBLE_NAME", "ENSEMBLE_PRODUCTS",
    "OBSGRID_ABI_MARKER", "OBSGRID_ENV", "OBSGRID_NAME", "OBSGRID_PRODUCTS",
    "crate_dir", "ensemble_remedy", "ensemble_workaround_notice",
    "find_ensemble_bin", "find_obsgrid_bin", "list_ensemble_fields",
    "obsgrid_remedy", "probe_ensemble_bin", "probe_obsgrid_bin",
    "resolve_ensemble_engine", "resolve_lane_engine", "resolve_obsgrid_engine",
    "run_ensemble_renderer", "run_obsgrid_renderer",
]
