"""Locate and drive the vendored Rusty Weather renderer (``rw_wrfbatch``).

``woof render --engine rust`` renders wrfout files through the vendored
``tools/rustwx`` workspace -- the production Rusty Weather batch
renderer, the same engine (and plot quality) as the campaign's paired
CPU-vs-GPU product sheets.  Like the GRIB bridges, the pip wheel ships
no compiled Rust: the binary is built once from the vendored workspace
(``cargo build --release --locked --offline``) and then *pointed at*,
with the same resolution ladder as :mod:`woof.bridges`:

1. the ``WOOF_RW_WRFBATCH`` environment variable naming the built file
   (a missing file it names is a hard error, never silently skipped);
2. a source checkout's ``tools/rustwx/target/{release,debug}``;
3. ``<root>/libexec/bridges`` beside the package;
4. ``~/.woof/bridges``.

The renderer draws coastlines/state/county basemaps from the vendored
Natural Earth + US Census assets in ``tools/rustwx/assets/basemap``.
When the binary runs from a checkout it finds them by walking its own
ancestors.  An installed binary has nothing above it to find: the
platform wheel stages the renderer into ``woof/libexec/bridges`` and
has no room for the shapefiles, so they ship in the ``recast-woof-data``
companion every install already pulls
(:func:`woof.data_assets.companion_basemap_dir`) and
:func:`renderer_env` pins ``RUSTWX_BASEMAP_DIR`` to the first of the
checkout assets, the companion's, and the ones ``woof fetch-bridges``
staged.  An explicit ``RUSTWX_BASEMAP_DIR``/``RUSTWX_ASSETS_DIR`` in the
caller's environment always wins.

Nothing here runs cargo.  Resolution has one side effect and one
only: an artifact found in ``~/.woof/bridges`` that is not the one this
release pinned is re-fetched before it is handed to a door
(:func:`woof.bridges.require_release_pin`).  ``woof doctor`` resolves
inside :func:`woof.bridges.inspection_only`, where there is no side
effect at all, so the report still says what the estate IS.
"""

from __future__ import annotations

import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

from woof import bridges
from woof.bridges import (RUSTWX_CRATE_RELATIVE, artifact_remedy,
                           default_bridge_dir,
                           legacy_bridge_candidates, lazy_build_hints,
                           rustwx_build_hint,
                           executable_name, packaged_bridge_dir)

#: Environment variable naming a prebuilt renderer executable.
RENDERER_ENV = "WOOF_RW_WRFBATCH"

#: Optional product metadata, matching rw_wrfbatch's CHEM_CORE_FIELD_CATALOG.
#: The renderer inspects source presence; this table never plans a product.
#: (slug, wrfout source, mass level or None, display units).
CHEM_PRODUCT_ROWS = (
    ("smoke_near_surface", "SMOKE_SFC", None, "ug/m^3"),
    ("smoke_column", "SMOKE_COLUMN", None, "mg/m^2"),
    ("pm25_near_surface", "PM2_5_DRY", 1, "ug/m^3"),
    ("aod_550", "AOD5502D", None, "1"),
    ("dust_near_surface", "DUST_SFC", None, "ug/m^3"),
    ("ozone_near_surface", "o3", 1, "ppb"),
)

#: Executable base name of the vendored batch renderer.
RENDERER_NAME = "rw_wrfbatch"

#: The built-in render themes the shipped renderer resolves by name
#: (rustwx-render theme.rs ``RenderTheme::builtin_names``), for a caller that
#: picks a theme per run before anything renders.  ``woof-light-tc`` is the
#: WOOF look with the tropical-cyclone wind ladder (bands on 34, 50, 64, 83,
#: 96, 113 and 137 kt, to 180 kt) and isobars every 4 hPa from 880 hPa.  A
#: caller names a theme only when it is listed here: an unknown name is a
#: THEME_ERROR, and every map of the run falls back to the plain default look.
RENDER_THEMES = ("default", "light", "dark", "woof-light", "woof-dark", "tc", "woof-light-tc")

#: How a native renderer line opens when it tells the reader something
#: about the pictures themselves -- a subtitle it had to cut, a record it
#: had to start over -- rather than reporting progress.
NATIVE_WARNING_PREFIXES = ("warning:", "WARNING ")

SIMULATED_RADAR_ABI = (
    "rw_simradar --request REQUEST.json schema=simulated-radar.request/v1 "
    "manifest=simulated-radar.manifest/v1 volume_paths=v1 scene_shapes=v1 bands=v1"
)
CANONICAL_RADAR_ABI = (
    "native-atmosphere.columns/v1 temperature=temperature_k "
    "winds=earth-relative-mass-grid/v1"
)

VERIFICATION_ENV = "WOOF_RW_VERIFY"


def find_verification_binary() -> Path | None:
    """Use the native checkout rung and shared override/packaged resolver."""
    filename = executable_name("rw_verify")
    override = os.environ.get(VERIFICATION_ENV)
    if override:
        return bridges.find_artifact(VERIFICATION_ENV, filename)
    for path in (crate_dir() / "target" / "release" / filename,
                 crate_dir() / "target" / "debug" / filename):
        if path.is_file():
            return bridges.accept_resolved(path.resolve())
    return bridges.find_artifact(VERIFICATION_ENV, filename)


def probe_verification_binary(path: Path) -> tuple[bool, str]:
    """Does ``path`` speak the request contract this woof writes?

    THE probe for ``rw_verify``: :func:`verification_binary` refuses by
    it and ``woof doctor`` reports by it, so the report and the door
    cannot disagree about one build.  Judged statically out of the bytes
    against :data:`woof.bridges.BRIDGE_ABI_MARKERS`, the check
    ``woof fetch-bridges`` applies before it stages one.
    """
    return bridges.bridge_abi_matches("rw_verify", path)


def verification_binary() -> Path:
    """Require the native artifact selected by the shared resolution ladder.

    Breakage the stale-build refusal prevents: a build predating the
    request contract launched, answered ``--prepare`` with a request this
    woof does not read, and surfaced as "returned an incompatible
    request" with no word that the build was stale or how to replace it,
    while ``woof doctor`` reported that same build stale.
    """
    filename = executable_name("rw_verify")
    found = find_verification_binary()
    if found is None:
        raise RuntimeError(artifact_remedy(
            env_var=VERIFICATION_ENV, filename=filename,
            subject="the observation verification engine", crate_relative=RUSTWX_CRATE_RELATIVE,
            one_liner=rustwx_build_hint(), artifact="rw_verify"))
    usable, evidence = probe_verification_binary(found)
    if not usable:
        raise RuntimeError(f"{found}: {evidence}.  Rebuild it: {rustwx_build_hint()}")
    return found


def _verification_run(arguments: list[str], *, timeout: float) -> dict:
    import json

    env = renderer_env()
    env["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run([str(verification_binary()), *arguments], capture_output=True,
                            text=True, errors="replace", env=env, timeout=timeout)
    if result.returncode:
        raise RuntimeError("native observation verification: " +
                           (result.stderr.strip() or f"exit {result.returncode}"))
    try:
        record = json.loads(result.stdout)
    except ValueError as error:
        raise RuntimeError("native observation verification returned no JSON receipt") from error
    return record


def verify_lightning_counts(request: dict, *, receipt_path: Path, timeout=900) -> dict:
    """Verify stored LTG1/2/3 and PR92 against native GLM total-flash counts.

    The request declares contiguous observed coverage, its geographic footprint,
    and the native invariant grid used for cell areas. Rust reads and scores it.
    """
    from woof.verification_visuals import _atomic_json

    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    request_path = receipt_path.with_name(receipt_path.stem + ".request.json")
    _atomic_json(request_path, request)
    record = _verification_run(["--request", str(request_path)], timeout=timeout)
    if record.get("schema") != "gpuwm.lightning-verification.receipt.v1":
        raise RuntimeError("native lightning verification returned an incompatible receipt")
    _atomic_json(receipt_path, record)
    return record


def verification_inventory(request: dict, *, workdir: Path, timeout=120) -> dict:
    """Ask Rust for grid and time metadata without Python field decoding."""
    from woof.verification_visuals import _atomic_json

    workdir.mkdir(parents=True, exist_ok=True)
    path = workdir / "inventory.request.json"
    _atomic_json(path, request)
    record = _verification_run(["--inventory", str(path)], timeout=timeout)
    if record.get("schema") != "gpuwm.verify-visuals.inventory.v1":
        raise RuntimeError("native verification inventory returned an incompatible schema")
    return record


def prepare_verification(request: dict, *, workdir: Path, timeout=120) -> dict:
    """Retain native verification planes so later arrivals survive raw cleanup."""
    import json
    from woof.verification_visuals import _atomic_json

    workdir.mkdir(parents=True, exist_ok=True)
    path = workdir / "prepare.request.json"
    prepared_path = workdir / "prepared.request.json"
    staged = workdir / ".prepared.request.json.partial"
    _atomic_json(path, request)
    try:
        _verification_run(["--prepare", str(path), "--json", str(staged)], timeout=timeout)
        if not staged.is_file():
            raise RuntimeError("native verification did not commit its reusable input request")
        prepared = json.loads(staged.read_text(encoding="utf-8"))
        if prepared.get("schema") != "gpuwm.verify-visuals.request.v1":
            raise RuntimeError("native verification preparation returned an incompatible request")
        staged.replace(prepared_path)
        return {**request, **prepared}
    finally:
        staged.unlink(missing_ok=True)


def verify_observations(request: dict, *, receipt_path: Path, image_path: Path,
                        timeout=900) -> dict:
    """Submit paths and metadata; Rust samples, scores and renders the outputs."""
    import hashlib
    import json
    from woof.verification_visuals import _atomic_json, _outputs_present

    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    path = receipt_path.with_name(receipt_path.stem + ".request.json")
    staged = receipt_path.with_name("." + receipt_path.name + ".partial")
    _atomic_json(path, request)
    try:
        _verification_run(["--request", str(path), "--json", str(staged),
                           "--image", str(image_path)], timeout=timeout)
        if not staged.is_file() or not image_path.is_file():
            raise RuntimeError("native verification did not commit its score receipt and image")
        record = json.loads(staged.read_text(encoding="utf-8"))
        if record.get("schema") != "gpuwm.verify-visuals.receipt.v1":
            raise RuntimeError("native verification returned an incompatible score receipt")
        if (record.get("request_sha256") != hashlib.sha256(path.read_bytes()).hexdigest()
                or record.get("domain") != request["domain"]
                or record.get("valid_time") != request["valid_time"]):
            raise RuntimeError("native verification receipt does not bind the submitted hour and request")
        if not _outputs_present(record):
            raise RuntimeError("native verification images are missing or do not match their receipt hashes")
        staged.replace(receipt_path)
    finally:
        staged.unlink(missing_ok=True)
    artifacts = record.setdefault("artifacts", [])
    if not any(row.get("path") == str(image_path) for row in artifacts):
        artifacts.append({"product": "verification", "path": str(image_path)})
    return record


def verification_reference(*, cache: Path, reference: str, cycle, hour: int,
                           timeout=120, reference_dir=None, reference_file=None, offline=False) -> dict:
    """Resolve a reference through the renderer's native metadata table."""
    import json

    from woof import rustwx_lanes

    # The comparison engine's own ladder and its reference-input probe,
    # the pair `woof doctor` reports for this door; never a second copy.
    binary = rustwx_lanes.require_compare_reference_bin()
    cache.mkdir(parents=True, exist_ok=True)
    arguments = [str(binary),
        "--fetch-reference", "--store-root", str(cache), "--reference", reference,
        "--cycle", cycle.strftime("%Y%m%d%H"), "--forecast-hour", str(hour)]
    if reference_dir:
        arguments.extend(["--reference-dir", str(reference_dir)])
    if reference_file:
        arguments.extend(["--reference-file", str(reference_file)])
    if offline:
        arguments.append("--offline")
    result = subprocess.run(arguments,
        capture_output=True, text=True, errors="replace", env=renderer_env(), timeout=timeout)
    if result.returncode:
        raise RuntimeError("reference input: " + result.stderr.strip())
    record = json.loads(result.stdout)
    if record.get("schema") != "gpuwm.reference-input.v1" or not Path(record["path"]).is_file():
        raise RuntimeError("reference input returned an incompatible or incomplete receipt")
    return record

#: The environment variable naming an explicit ``rw_simradar`` build.
SIMULATED_RADAR_ENV = "WOOF_RW_SIMRADAR"

#: What the replay door's native refusals say to do next.
SIMULATED_RADAR_NEXT = (
    "Next: correct the named option or input and run the command again; "
    "`woof simulated-radar --estimate --config CONFIG.toml` checks the scan "
    "geometry and memory without writing radar.")


class SimulatedRadarRefusal(RuntimeError):
    """The simulated radar engine cannot serve a request.

    The message names what breaks and the next step: a missing or stale
    ``rw_simradar`` (no radar volume could be written, or one would be
    written to a contract this release does not read), or the native
    command's own refusal of an option or input. ``woof simulated-radar``
    prints it as one refusal at exit 2; a forecast door raises it before
    the fetch.
    """


def _simulated_radar_remedy() -> str:
    return artifact_remedy(
        env_var=SIMULATED_RADAR_ENV, filename=executable_name("rw_simradar"),
        subject="the simulated radar engine", crate_relative=RUSTWX_CRATE_RELATIVE,
        one_liner=rustwx_build_hint(), artifact="rw_simradar")


def simulated_radar_binary() -> Path | None:
    """Resolve the native radar simulator through the standard artifact paths."""
    override = os.environ.get(SIMULATED_RADAR_ENV)
    for candidate in _candidates(SIMULATED_RADAR_ENV, "rw_simradar"):
        if candidate.is_file():
            return bridges.accept_resolved(candidate.resolve())
        if override and candidate == Path(override):
            raise SimulatedRadarRefusal(
                f"{SIMULATED_RADAR_ENV} names a missing file: {candidate}, so no "
                f"radar volume can be written. Next: unset {SIMULATED_RADAR_ENV} "
                "to use the installed rw_simradar, or point it at a built one.")
    return None


def require_simulated_radar_binary() -> Path:
    """The ``rw_simradar`` this release's request contract can drive.

    Resolves the binary and checks ``--abi`` against
    :data:`SIMULATED_RADAR_ABI`. A forecast asking for radar calls this at
    its door, before the fetch, preparation or GPU allocation: a missing or
    stale binary otherwise surfaced only when the first history landed.
    """
    binary = simulated_radar_binary()
    if (binary is None and not os.environ.get(SIMULATED_RADAR_ENV)
            and bridges.fetch_on_demand("rw_simradar")):
        # On demand since 2.9.11 (PyPI per-file limit): pinned in this
        # release's bundle, staged by the first door that needs it.
        binary = simulated_radar_binary()
    if binary is None:
        failure = bridges.on_demand_failure("rw_simradar")
        raise SimulatedRadarRefusal(
            "rw_simradar is not installed, so [simulated_radar] cannot write "
            "a radar volume" + (f" ({failure})" if failure else "") + ".\n"
            + _simulated_radar_remedy())
    try:
        probe = subprocess.run([str(binary), "--abi"], capture_output=True, text=True,
                               timeout=60, env=renderer_env())
    except (OSError, subprocess.SubprocessError) as error:
        raise SimulatedRadarRefusal(
            f"{binary} did not run ({error}), so no radar volume can be "
            "written.\n" + _simulated_radar_remedy()) from error
    if probe.returncode or probe.stdout.strip() != SIMULATED_RADAR_ABI:
        raise SimulatedRadarRefusal(
            f"{binary} answers a different simulated radar request contract "
            f"than this release writes (expected {SIMULATED_RADAR_ABI!r}, got "
            f"{probe.stdout.strip()!r}), so its requests would be refused or "
            "misread. Next: rebuild or re-stage it.\n" + _simulated_radar_remedy())
    return binary


def canonical_radar_binary() -> Path:
    """Require native column and temperature support before a conversion."""
    binary = require_simulated_radar_binary()
    probe = subprocess.run([str(binary), "--canonical-abi"], capture_output=True,
                           text=True, timeout=60, env=renderer_env())
    if probe.returncode or probe.stdout.strip() != CANONICAL_RADAR_ABI:
        raise SimulatedRadarRefusal(
            "the simulated radar binary lacks the native atmosphere and "
            "temperature contract, so it would read native columns without "
            "their actual temperature. Next: rebuild or re-stage rw_simradar.\n"
            + _simulated_radar_remedy())
    return binary


def canonical_radar_scene(source_path: Path, *, outdir: Path) -> Path:
    """Convert native physical columns to a durable radar scene in Rust."""
    import json

    binary = canonical_radar_binary()
    source_path = Path(source_path).resolve()
    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    # Source paths are one durable atmosphere per model time. The native reader
    # validates its timestamp; orchestration never decodes the field transport.
    target = outdir / ("wrfout_d01_" + source_path.stem + ".nc")
    result = subprocess.run([str(binary), "--canonical-atmosphere", str(source_path),
                             "--out", str(target)], capture_output=True, text=True,
                            env=renderer_env())
    if result.returncode:
        raise SimulatedRadarRefusal(
            "native radar atmosphere: " + (result.stderr.strip() or
            f"native command exited {result.returncode}") + "\n" + SIMULATED_RADAR_NEXT)
    event = json.loads(result.stdout.strip().splitlines()[-1])
    if event.get("event") != "canonical_atmosphere_committed" or not target.is_file():
        raise RuntimeError("native radar atmosphere did not publish its scene")
    return target


def simulate_radar(history_paths, *, outdir: Path, config: dict, volume_paths=None,
                   binary: Path | None = None, started=None, events=None) -> dict:
    """Pass durable histories to Rust and return its committed run inventory.

    All field reads, beam sampling, file encoding, hashing and PPI rendering
    happen in the native command. Python carries request metadata only.

    ``volume_paths`` names the histories that publish volumes (default all);
    the rest are scan-timing neighbours. ``binary`` is an already admitted
    executable (:func:`require_simulated_radar_binary`), so a live forecast
    probes it once rather than per history. ``started`` is called with the
    native ``Popen`` so a forecast stopping early can terminate it.
    ``events``, when supplied, collects the native JSON execution events,
    including reused and newly committed domain-times for this call.
    """
    import json
    import tempfile

    if binary is None:
        binary = require_simulated_radar_binary()
    paths = [str(Path(path).resolve()) for path in history_paths]
    if not paths:
        raise ValueError("simulated radar needs at least one durable history file")
    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    request = {"schema": "simulated-radar.request/v1", "history_paths": paths,
               "outdir": str(outdir), "config": config}
    if volume_paths is not None:
        request["volume_paths"] = [str(Path(path).resolve()) for path in volume_paths]
    with tempfile.TemporaryDirectory(prefix=".simulated-radar-", dir=outdir) as tmp:
        request_path = Path(tmp) / "request.json"
        request_path.write_text(json.dumps(request, allow_nan=False), encoding="utf-8")
        process = subprocess.Popen([str(binary), "--request", str(request_path)],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, env=renderer_env())
        try:
            if started is not None:
                started(process)
            _stdout, stderr = process.communicate()
        except BaseException:
            # Never leave a child writing radar after its caller stopped.
            if process.poll() is None:
                process.kill()
                process.wait()
            raise
    relay_native_warnings(stderr)
    if process.returncode:
        raise SimulatedRadarRefusal(
            "simulated radar: " + (stderr.strip() or
            f"native command exited {process.returncode}") + "\n" + SIMULATED_RADAR_NEXT)
    manifest_path = outdir / "radar" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "simulated-radar.manifest/v1" or manifest.get("simulated") is not True:
        raise RuntimeError("native radar command did not commit a simulated volume manifest")
    if events is not None:
        for line in _stdout.splitlines():
            event = json.loads(line)
            if isinstance(event, dict):
                events.append(event)
    return manifest


def estimate_simulated_radar(history_paths=(), *, outdir: Path, config: dict,
                             scene_shapes=(), binary: Path | None = None) -> dict:
    """Ask Rust for geometry, memory admission and output bounds.

    An empty history list estimates polar work only. Full canonical scenes
    add their header dimensions without reading weather arrays;
    ``scene_shapes`` (``(nx, ny, nz)`` mass grids) price a forecast's grids
    before any history exists. The native resource schema is checked
    independently of the volume request ABI.
    """
    import json
    import tempfile

    if binary is None:
        binary = require_simulated_radar_binary()
    request = {"schema": "simulated-radar.request/v1",
               "history_paths": [str(Path(path).resolve()) for path in history_paths],
               "outdir": str(Path(outdir).resolve()), "config": config}
    if scene_shapes:
        request["scene_shapes"] = [[int(n) for n in shape] for shape in scene_shapes]
    with tempfile.TemporaryDirectory(prefix="simulated-radar-estimate-") as tmp:
        path = Path(tmp) / "request.json"
        path.write_text(json.dumps(request, allow_nan=False), encoding="utf-8")
        result = subprocess.run([str(binary), "--estimate", str(path)],
                                capture_output=True, text=True, timeout=60,
                                env=renderer_env())
    if result.returncode:
        raise SimulatedRadarRefusal(
            "simulated radar estimate: " + (result.stderr.strip() or
            f"native command exited {result.returncode}") + "\n" + SIMULATED_RADAR_NEXT)
    try:
        estimate = json.loads(result.stdout)
    except ValueError as error:
        raise RuntimeError("the simulated radar binary did not return its resource contract") from error
    if estimate.get("schema") != "simulated-radar.resources/v1":
        raise RuntimeError("the simulated radar binary lacks the resource estimate contract")
    return estimate


def relay_native_warnings(stderr: str | None) -> list[str]:
    """Print the native renderer's warning lines on this process's stderr.

    A bridge reads the renderer's stderr instead of passing it through,
    because most of it is progress and machine rows.  Reading only the
    ``FAILED`` rows dropped every warning with them: a subtitle the engine
    cut, and said it had cut, reached nobody who rendered through
    ``woof render``.  Returns the lines relayed, in the engine's order.
    """

    lines = [line for line in (stderr or "").splitlines()
             if line.startswith(NATIVE_WARNING_PREFIXES)]
    for line in lines:
        print(line, file=sys.stderr)
    return lines

#: ``CARGO_BUILD_HINT``: the one-liner that builds the renderer, from a
#: checkout root, spelled for the shell rule when it is read (Windows
#: PowerShell 5.1 cannot parse ``&&``).
__getattr__ = lazy_build_hints(
    __name__, CARGO_BUILD_HINT=RUSTWX_CRATE_RELATIVE)

#: The exact ``rw_wrfbatch --abi`` line this wrapper was written against.
#:
#: The renderer was the only bundled binary with no contract check.  Its
#: two workspace siblings pin one (:data:`woof.rustwx_fetch
#: .FETCH_ABI_MARKER`, :data:`woof.obs.nexrad.NEXRAD_ABI_MARKER`) and
#: the five GRIB decoders pin a static one
#: (:data:`woof.bridges.BRIDGE_ABI_MARKERS`); ``rw_wrfbatch`` was asked
#: only whether it started.  Two builds four megabytes and two days
#: apart -- md5 72c739e8... and 894cc90f... -- both printed the usage
#: line, both were reported ``verified``, and neither was from this
#: tree.  A renderer that verifies on "it launches" is a renderer that
#: can draw a whole product set from a foreign engine (task #106).
#:
#: The literal spells the contract out rather than carrying a version
#: number, exactly as ``BRIDGE_ABI_MARKERS`` requires: the ``PRODUCT``
#: /``CATALOG`` row grammar :func:`list_products` parses, the
#: ``RENDERED``/``SKIPPED``/``FAILED`` events and the ``SECTIONFILL``
#: line :func:`run_renderer` parses, and the generic ``var:`` and
#: vertical-section ``xsec:``
#: vocabularies whose absence from a stale build is what #106 was
#: reported as.  Changing any of those changes this string, and every
#: binary predating the change answers ``unknown option --abi`` instead
#: of the old grammar.
RENDERER_ABI_MARKER = (
    "gpuwm-rw-wrfbatch-catalog-v1\tPRODUCT\tslug\tkind\tstatus\tdetail\t"
    "code\tCATALOG\t"
    "gpuwm-rw-wrfbatch-requirements-v1\tNEEDS\tslug\tselector\tPLANNED\t"
    "store_field\t"
    "gpuwm-rw-wrfbatch-wrfout-lane-v1\tWRFOUT\tslug\tkind\tverdict\t"
    "minimum_hour\tdetail\t"
    "gpuwm-rw-wrfbatch-events-v2\tRENDERED\tSKIPPED\tFAILED\t"
    "frame-attributed\t"
    "gpuwm-rw-wrfbatch-sections-v1\tSECTIONFILL\tslug\tlo\thi\tabsence\t"
    "rule\t"
    "gpuwm-rw-wrfbatch-inputs-json-v1\t"
    "gpuwm-rw-wrfbatch-vocabulary-v2\tgeneric\tvar:\tvariables\txsec:\t"
    "mesh:\tmeshdiff:\tselectable_slugs\t"
    "gpuwm-rw-wrfbatch-layout-v1\t--layout\tauto\tfixed\t"
    "--size-class\t--scale\t--pair-sheet\t"
    "gpuwm-rw-wrfbatch-difference-v2\t--diff-against\t--diff-inputs-json\t"
    "--diff-label-a\t--diff-label-b\t--diff-labels\t"
    "--diff-sheet\tDIFFERENCE\tdiffering_cells\tRUSTWX_DIFFERENCE_TABLE")

#: The generic product families, read OUT of the pinned marker rather
#: than listed again beside it.  The marker is the contract a built
#: renderer answers with, so a family added there is advertised here in
#: the same commit and a door cannot end up offering a vocabulary the
#: engine does not have.
GENERIC_FAMILIES: tuple[str, ...] = tuple(
    field for field in RENDERER_ABI_MARKER.split("\t")
    if field.endswith(":"))

#: The vertical-section family's prefix, spelled once.
SECTION_PREFIX = "xsec:"

#: The event word carrying the range one vertical cut's fill was drawn
#: over.  A section's colour bar is fitted per frame at BOTH ends -- to
#: the rung its own signal reaches and, since a fill that would use less
#: than half a zero-anchored bar takes its own minimum instead, to the
#: air the cut holds -- so two pictures of one line an hour apart can be
#: drawn on two different bars and neither picture says so.  This is
#: what puts it in the receipt.
SECTION_FILL_EVENT = "SECTIONFILL"

#: The one generic family a STORE can be enumerated for.
#:
#: The generic catalog is, by construction, every stored 2-D variable
#: that no named product already draws -- so for this prefix, and only
#: this prefix, "there is no row" is PROOF that the store carries no
#: such variable, and the renderer answers a request for one with
#: ``stored 2-D variable "X" does not exist`` and fails the whole
#: invocation.  ``xsec:`` is cut from the wrfout files by the section
#: lane and ``mesh:``/``meshdiff:`` read a mesh, so none of those is a
#: store product and none can be decided from a store listing.
GENERIC_VAR_PREFIX = "var:"

#: The group keywords that name no product and carry no section term.
#:
#: Public because a door has to tell "the whole catalog, whatever it
#: holds" apart from "these named slugs".  The engine expands a group
#: itself and leaves out what it cannot draw, so a group token carries
#: no promise a caller can check and must pass through untouched; a
#: NAMED slug is a promise, and a door that forwards one the catalog
#: has just refused gets the renderer's whole-invocation failure.
GROUP_KEYWORDS = frozenset(
    {"all", "direct", "derived", "heavy", "windowed", "variables"})

#: The keyword that draws every stored 2-D variable no named product
#: draws.  ``all`` is the NAMED products the frames can draw and nothing
#: else: an 18 h run drawn with ``all`` used to publish 137 of its 204
#: product folders as raw variables, most of them grids a named product
#: beside them already drew.
VARIABLES_KEYWORD = "variables"

#: The spelling this module used before the set was public.
_GROUP_KEYWORDS = GROUP_KEYWORDS

#: The codes a windowed row carries when the store's TIME AXIS is what
#: excluded it: a store with a single frame, or (from renderers built
#: before 2.8.0) one whose frames sit on an exact-time ordinal axis.  The
#: current engine serves windows on that axis from each frame's lead and
#: no longer emits ``windowed-ordinal-axis``; the code stays so a listing
#: from an older bridge still reads.  No door branches on them --
#: :func:`catalog_verdict` drops every non-renderable row alike -- and
#: they are here as the engine's own vocabulary, named in
#: ``docs/render-output-layout.md`` as the family a forwarded product
#: FAILS a whole invocation on, and pinned by the catalog contract
#: tests.  A reason may be reworded at any time; these may not.
WINDOW_AXIS_CODES = ("windowed-needs-whole-hour-frames",
                     "windowed-ordinal-axis")

#: How the generic emitter opens the row it prints for a stored
#: variable whose name no request can carry.  It is the one
#: ``GENERIC_EXCLUDED`` form that is NOT a renderable variable, so
#: the listing skips it rather than folding it in.
_GENERIC_NAME_REFUSED = "name is not request-safe"

_PROBE_TIMEOUT_S = 20


def crate_dir() -> Path:
    """The vendored Rusty Weather workspace of a source checkout."""

    return Path(__file__).resolve().parent.parent / "tools" / "rustwx"


def basemap_dir() -> Path:
    """The vendored basemap assets (Natural Earth + US counties)."""

    return crate_dir() / "assets" / "basemap"


#: What to tell a caller whose install cannot read the basemap
#: shapefiles.  Spelled out here, once, beside the resolver for the
#: assets it reads, so every entry point says the same thing.
#:
#: One half now.  ``pyshp`` reads the geometry and the shapefiles ARE
#: the geometry; those used to arrive only in the bundle ``woof
#: fetch-bridges`` stages, so this remedy named that command too.  They
#: ship in the ``recast-woof-data`` companion every install pulls since 2.8.0,
#: so the pip line is the whole fix and the sentence says where the
#: geometry comes from instead of sending the reader to a second step.
PYSHP_REMEDY = (
    "the map frame needs pyshp (it reads the Natural Earth and US Census "
    "shapefiles the basemap is drawn from, which arrive with the "
    "recast-woof-data package every install pulls); install it with "
    "`pip install pyshp>=2.3` or `pip install recast-woof[render]`")


def pyshp_available() -> bool:
    """Whether the shapefile reader every basemap needs can be imported.

    Asked at a front door rather than left to the function-local ``import
    shapefile`` inside the renderers, for exactly the reason
    :func:`woof.obs.dealias.scipy_available` exists: the two failures are
    not the same failure.

    The DA nowcast's render stage runs DEAD LAST -- after the survey, the
    fetch, the preparation, the free forecast and every DA cycle -- and
    reaching that import meant a bare ``ModuleNotFoundError: No module
    named 'shapefile'`` with no message at all, having destroyed the most
    work of any failure in the product.  Answering here costs a
    ``find_spec`` before the run starts.

    ``find_spec`` rather than a real import: this is asked on the hot path
    of a front door that may then not draw anything, and importing a
    module to learn whether it exists is a side effect a capability check
    should not have.
    """

    from importlib.util import find_spec

    try:
        return find_spec("shapefile") is not None
    except (ImportError, ValueError):     # pragma: no cover - broken install
        return False


def require_pyshp() -> None:
    """Import-time gate for a module whose whole job is drawing a map.

    The named refusal the three bare ``import shapefile`` call sites
    lacked.  ``ImportError`` keeps the class a caller would already be
    catching around an import, and the message carries the remedy.
    """

    if not pyshp_available():
        raise ImportError(PYSHP_REMEDY)


#: How many ancestors of the renderer executable's own directory the
#: renderer walks looking for ``assets/basemap``.  Mirrors
#: ``rustwx-render``'s ``basemap_root_candidates``; a build at
#: ``tools/rustwx/target/release/`` reaches the crate's assets at the
#: second ancestor, which is why a renderer built from a clone finds its
#: basemaps whatever directory it is launched from.
_RENDERER_EXE_ANCESTORS = 8

#: And how many ancestors of the working directory it walks.
_RENDERER_CWD_ANCESTORS = 6


def basemap_candidates(renderer: Path | None = None) -> tuple[Path, ...]:
    """Where the RENDERER looks for basemaps, in its own order.

    Not where woof keeps them.  ``woof doctor`` used to probe the
    single checkout path :func:`basemap_dir` and announce "NO basemap
    assets found" whenever it was absent -- which is every pip install,
    including ones where ``rw_wrfbatch`` was resolving the assets
    perfectly well from its own build directory.  A report that
    contradicts the artifact is worse than no report, so this mirrors
    ``rustwx-render``'s ``basemap_root_candidates`` instead:

    1. ``RUSTWX_BASEMAP_DIR``;
    2. ``RUSTWX_ASSETS_DIR/basemap``;
    3. ``assets/basemap`` and ``Resources/assets/basemap`` under each of
       the first eight ancestors of the executable's own directory;
    4. ``assets/basemap`` and ``basemap`` under each of the first six
       ancestors of the working directory;
    5. the crate's own ``assets/basemap`` -- the compile-time workspace
       root, which for this vendored crate is :func:`basemap_dir`.

    Duplicates are dropped, first occurrence winning, exactly as the
    renderer does it.
    """

    candidates: list[Path] = []

    def push(path: Path) -> None:
        if path not in candidates:
            candidates.append(path)

    override = os.environ.get("RUSTWX_BASEMAP_DIR")
    if override:
        push(Path(override))
    assets = os.environ.get("RUSTWX_ASSETS_DIR")
    if assets:
        push(Path(assets) / "basemap")

    if renderer is not None:
        parent = Path(renderer).resolve().parent
        for ancestor in (parent, *parent.parents)[:_RENDERER_EXE_ANCESTORS]:
            push(ancestor / "assets" / "basemap")
            push(ancestor / "Resources" / "assets" / "basemap")

    working = Path.cwd()
    for ancestor in (working, *working.parents)[:_RENDERER_CWD_ANCESTORS]:
        push(ancestor / "assets" / "basemap")
        push(ancestor / "basemap")

    push(basemap_dir())
    return tuple(candidates)


def cartopy_natural_earth_root() -> Path | None:
    """The cartopy shapefile cache, if this machine has one.

    Not part of :func:`basemap_candidates` -- the renderer consults this
    *after* those, per layer, inside its own loaders -- but it is real
    geography and it is why this bug survived a release.  A workstation
    that has ever run cartopy has this directory, so ``rw_wrfbatch``
    draws perfectly good coastlines there while the same binary on a
    clean machine draws none.  Anything that reports on basemap
    availability has to know about it or it reports a state the
    artifact does not have.

    Mirrors ``rustwx-render``'s ``cartopy_natural_earth_root``,
    including its ``USERPROFILE``-before-``HOME`` order.
    """

    home = os.environ.get("USERPROFILE") or os.environ.get("HOME")
    if not home:
        return None
    root = (Path(home) / ".local" / "share" / "cartopy" / "shapefiles"
            / "natural_earth")
    return root if root.is_dir() else None


def companion_basemap_dir() -> Path | None:
    """The map assets the ``recast-woof-data`` companion carries, or None.

    Named here, beside the resolver that uses it, so a test can take the
    companion away without uninstalling it.
    """

    from woof.data_assets import companion_basemap_dir as companion

    return companion()


def staged_basemap_dir() -> Path:
    """Where ``woof fetch-bridges`` stages the bundle's map assets."""

    return default_bridge_dir() / "assets" / "basemap"


def wrapper_basemap_candidates() -> tuple[Path, ...]:
    """The map roots the Python half hands the renderer, best first.

    Not the renderer's own search (:func:`basemap_candidates`): a wheel's
    ``libexec`` renderer has no ``assets/basemap`` above it, so
    :func:`renderer_env` pins one of these as ``RUSTWX_BASEMAP_DIR``.

    1. the checkout's own assets (:func:`basemap_dir`);
    2. the ``recast-woof-data`` companion's, which arrived with this very
       ``woof`` at its exact version, so a bare ``pip install`` draws
       its maps with no further step;
    3. the ones ``woof fetch-bridges`` staged, which outlive an upgrade
       and so come last;
    4. for a pinned install, the ones an older release staged into the
       flat ``~/.woof/bridges`` -- read only, never written.
    """

    candidates = [basemap_dir()]
    companion = companion_basemap_dir()
    if companion is not None:
        candidates.append(companion)
    candidates.append(staged_basemap_dir())
    candidates.extend(path / "basemap"
                      for path in legacy_bridge_candidates("assets"))
    return tuple(candidates)


def _basemap_overridden(env) -> bool:
    return "RUSTWX_BASEMAP_DIR" in env or "RUSTWX_ASSETS_DIR" in env


def resolve_basemap_dir(renderer: Path | None = None) -> Path | None:
    """The first candidate that exists, or None if the renderer has none.

    The renderer resolves each asset SUBDIRECTORY independently, so a
    root that exists is evidence rather than proof; a root that exists
    nowhere is proof, and that is the only case worth warning about.

    After the renderer's own search come the roots :func:`renderer_env`
    hands it (:func:`wrapper_basemap_candidates`): a platform wheel's
    ``libexec`` renderer finds nothing above itself, and a report that
    ignored what the wrapper passes would call a drawn map missing.
    """

    for candidate in basemap_candidates(renderer):
        if candidate.is_dir():
            return candidate
    if not _basemap_overridden(os.environ):
        for candidate in wrapper_basemap_candidates():
            if candidate.is_dir():
                return candidate
    return None


def basemap_remedy() -> str:
    """The one command that gives a renderer its map assets back.

    The companion carries them and is a hard dependency, so a render
    that finds none is running beside a ``recast-woof-data`` that was removed,
    edited, or skewed from this ``woof``; reinstalling it at this
    version restores the files whichever it was.
    """

    from woof.data_assets import companion_reinstall_command

    return companion_reinstall_command()


def _candidates(env_var: str, name: str) -> tuple[Path, ...]:
    filename = executable_name(name)
    candidates: list[Path] = []
    override = os.environ.get(env_var)
    if override:
        candidates.append(Path(override))
    root = Path(__file__).resolve().parent.parent
    candidates.extend((
        crate_dir() / "target" / "release" / filename,
        crate_dir() / "target" / "debug" / filename,
        root / "libexec" / "bridges" / filename,
        packaged_bridge_dir() / filename,
        default_bridge_dir() / filename,
        *legacy_bridge_candidates(filename),
    ))
    return tuple(candidates)


def renderer_candidates() -> tuple[Path, ...]:
    """Deterministic candidate paths for the renderer, best first."""

    return _candidates(RENDERER_ENV, RENDERER_NAME)


_ANNOUNCED_RENDERERS: set[Path] = set()


def find_renderer() -> Path | None:
    """First existing candidate, or None.

    An environment override that names a missing file is a hard error:
    explicit configuration must fail loudly, not fall through.

    Loudly AND with the exit named (1.8.8 refusal sweep).  The message
    used to end at the path, which leaves a reader who set the variable
    weeks ago -- or inherited it from a shell profile -- with a
    diagnosis and no instruction.  Three ways out, in the order they are
    likely to be wanted: point the variable somewhere real, drop it and
    take the vendored ladder, or ask for the fallback engine outright.
    """

    override = os.environ.get(RENDERER_ENV)
    for candidate in renderer_candidates():
        if candidate.is_file():
            selected = bridges.accept_resolved(candidate.resolve())
            if selected not in _ANNOUNCED_RENDERERS:
                print(f"renderer: selected {selected}", file=sys.stderr)
                _ANNOUNCED_RENDERERS.add(selected)
            return selected
        if override and candidate == Path(override):
            raise FileNotFoundError(
                f"{RENDERER_ENV} names a missing file: {candidate}.  "
                f"Point it at a built rw_wrfbatch binary, unset "
                f"{RENDERER_ENV} to use the vendored resolution ladder "
                f"(build it with: {rustwx_build_hint()}), or pass "
                f"--engine matplotlib to draw with the fallback engine.")
    return None


def renderer_remedy() -> str:
    """The remedy for a missing renderer, true for THIS install.

    Delegates rather than repeating the shape a third time: this copy is
    what kept a ``<clone>`` placeholder and an "exact copy-pasteable"
    claim after the bridge copy stopped making either.
    """

    return artifact_remedy(
        env_var=RENDERER_ENV, filename=executable_name(RENDERER_NAME),
        subject="the rust render engine",
        crate_relative=RUSTWX_CRATE_RELATIVE,
        one_liner=rustwx_build_hint())


def renderer_env() -> dict[str, str]:
    """Subprocess environment for the renderer.

    An explicit ``RUSTWX_BASEMAP_DIR``/``RUSTWX_ASSETS_DIR`` is the
    user's to keep; otherwise the first of the checkout assets, the
    ``recast-woof-data`` companion's and the ones fetch-bridges staged
    (:func:`wrapper_basemap_candidates`) is pinned, so the platform
    wheel's ``libexec`` binary draws its basemaps on a bare install.
    """

    env = dict(os.environ)
    if not _basemap_overridden(env):
        for assets in wrapper_basemap_candidates():
            if assets.is_dir():
                env["RUSTWX_BASEMAP_DIR"] = str(assets)
                break
    return env


def raster_device_env(env: dict[str, str]) -> tuple[dict[str, str], str]:
    """Hand the renderer's CUDA raster fill an explicit card; say which.

    The Rust renderer fills map rasters on a CUDA card
    (``rustwx-render/src/rasterize_cuda.rs``) and on the CPU only when no
    card opens or ``RUSTWX_RASTER_DEVICE=cpu`` is set. Left alone it took
    CUDA's device 0 with nothing saying so. Here the card is the first one
    ``CUDA_VISIBLE_DEVICES`` exposes (``RUSTWX_RASTER_DEVICE=<n>`` picks
    another visible one); a card lease that names its cards only in
    ``GPU_MUTEX_CARDS`` is turned into ``CUDA_VISIBLE_DEVICES``; with no
    card visible the fill is pinned to the CPU. Returns the environment
    and one line naming the choice.
    """

    env = dict(env)
    explicit = env.get("RUSTWX_RASTER_DEVICE", "").strip()
    if explicit.lower() == "cpu":
        return env, "raster fills on the CPU (RUSTWX_RASTER_DEVICE=cpu)"
    visible = env.get("CUDA_VISIBLE_DEVICES")
    leased = env.get("GPU_MUTEX_CARDS", "").strip()
    if visible is None and leased:
        env["CUDA_VISIBLE_DEVICES"] = visible = leased
        env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    if visible is not None and visible.strip() in ("", "-1"):
        env["RUSTWX_RASTER_DEVICE"] = "cpu"
        return env, ("raster fills on the CPU (CUDA_VISIBLE_DEVICES exposes "
                     "no card)")
    ordinal = int(explicit) if explicit.isdigit() else 0
    tail = ("; the CPU draws them if that card does not open, and "
            "RUSTWX_RASTER_DEVICE=cpu keeps them there")
    if visible is None:
        return env, (f"raster fills on CUDA device {ordinal}: no card is "
                     "assigned (CUDA_VISIBLE_DEVICES unset), so set "
                     "CUDA_VISIBLE_DEVICES to choose one" + tail)
    cards = [card.strip() for card in visible.split(",") if card.strip()]
    card = cards[ordinal] if ordinal < len(cards) else f"#{ordinal}"
    return env, (f"raster fills on CUDA card {card} (visible device "
                 f"{ordinal} of CUDA_VISIBLE_DEVICES={visible})" + tail)


def probe_renderer(path: Path) -> tuple[bool, str]:
    """``--help`` then ``--abi``: is this binary runnable, and is it ours?

    ``rw_wrfbatch --help`` prints its usage line and exits 0.  That
    observable separates a runnable executable from an empty, truncated,
    or wrong-platform file, which refuses to launch (OSError) or dies
    with an abnormal status and no usage text.

    Launching is necessary and was never sufficient.  ``--abi`` is the
    stale-build half, and it is the same handshake ``rw_fetch`` and
    ``rw_nexrad`` have always answered -- not a new mechanism, the
    existing one finally applied to the third bundled binary in this
    workspace.  A build whose catalog grammar, event words or generic
    ``var:`` vocabulary differ from :data:`RENDERER_ABI_MARKER` fails
    here, where a report says so, instead of at the product sheet where
    a reader counts plots and wonders.

    Two builds of ``rw_wrfbatch`` with different md5s both passed the
    ``--help``-only probe and both were reported ``verified``; that is
    the defect this closes, and the remedy is REBUILD, never re-point:
    the contract moved, so every binary older than it fails the same
    way.

    The header is read before the launch, as in every probe in this
    package: on Windows a corrupt image header can hang
    ``CreateProcess`` where no timeout reaches.  See
    :func:`woof.bridges.native_executable_format`.
    """

    ok, evidence = bridges.launchable(path)
    if not ok:
        return False, f"{evidence} -- corrupt, stale, or built for " \
                      "another platform"
    try:
        with bridges.quiet_loader_errors():
            probe = subprocess.run(
                [str(path), "--help"], capture_output=True, text=True,
                errors="replace", timeout=_PROBE_TIMEOUT_S)
    except OSError as error:
        return False, f"exists but failed to execute: {error}"
    except subprocess.TimeoutExpired:
        return False, (f"probe invocation did not exit within "
                       f"{_PROBE_TIMEOUT_S} s")
    transcript = f"{probe.stdout or ''}{probe.stderr or ''}"
    if probe.returncode != 0 or "usage: rw_wrfbatch" not in transcript:
        return False, (f"probe --help exited {probe.returncode} without the "
                       "expected usage line -- corrupt, stale, or built for "
                       "another platform")
    try:
        with bridges.quiet_loader_errors():
            abi = subprocess.run(
                [str(path), "--abi"], capture_output=True, text=True,
                errors="replace", timeout=_PROBE_TIMEOUT_S)
    except OSError as error:
        return False, f"--abi did not run: {error}"
    except subprocess.TimeoutExpired:
        return False, (f"--abi did not exit within {_PROBE_TIMEOUT_S} s")
    from woof import provenance_gate

    observed = (abi.stdout or "").strip()
    # A GPUWM_BRIDGE_SOURCE_REV line after the contract is not part of it
    # (rw_ensbatch's whole-answer compare refused its own checkout's
    # build); it is judged by the shared stamp check below.
    contract, revision = provenance_gate.split_abi_answer(observed)
    if abi.returncode != 0 or contract != RENDERER_ABI_MARKER:
        # A build predating the handshake answers `unknown option --abi`
        # on exit 2, so name that case for what it is rather than
        # reporting an empty string against a long expected line.
        seen = (f"exit {abi.returncode} with no --abi line" if not observed
                else f"exit {abi.returncode}: {observed[:120]!r}")
        return False, (
            "launches, but --abi does not match the render contract this "
            f"woof expects ({seen}) -- it is a build from another "
            "checkout, so its product catalog is not this tree's; REBUILD "
            f"it, do not re-point {RENDERER_ENV} at another copy: "
            f"{rustwx_build_hint()}")
    if revision is not None:
        refusal = provenance_gate.abi_revision_refusal(
            path, revision, env_var=RENDERER_ENV)
        if refusal is not None:
            return False, (
                f"launches and --abi matches the render contract, but "
                f"{refusal}; REBUILD it: {rustwx_build_hint()}")
    return True, ("probe --help exited 0 with its usage line; --abi matches "
                  "the render contract")


def list_products(renderer: Path, wrfout: Path, *, store_root: Path,
                  heavy: bool = False
                  ) -> tuple[list[tuple[str, str, str, str]], str]:
    """One catalog listing for ``wrfout``: (rows, summary).

    Rows are ``(slug, kind, status, detail)`` exactly as the renderer's
    ``--list-products`` mode emits them (statuses: renderable,
    missing-fields, blocked, excluded -- there is no identity-gated
    status; every non-renderable row names fields or a stated lane
    reason); ``summary`` is its ``CATALOG ...`` tally line.  The import
    into ``store_root`` is the real one -- availability is proven
    against the stored fields, never guessed from filenames.

    Four fields, deliberately: this is what every consumer in the tree
    already unpacks.  :func:`catalog_rows` is the same listing with the
    machine code beside each row, for a consumer that has to DECIDE
    something rather than print it.
    """

    return list_products_series(renderer, (wrfout,), store_root=store_root, heavy=heavy)


def catalog_rows(renderer: Path, wrfouts, *, store_root: Path,
                 heavy: bool = False, products: str | None = None,
                 save_import: Path | None = None,
                 ) -> tuple[list[tuple[str, str, str, str, str]], str]:
    """The same listing, with each row's machine code: (rows, summary).

    ``(slug, kind, status, detail, code)``.  The detail is prose and is
    what a reader sees; the code is a stable spelling of WHY the row has
    the status it has, and it is what a consumer matches on.  Matching
    on the prose is a copy of the engine's sentences kept in Python,
    which stops matching the moment the engine rewords one -- at which
    point the excluded slug is forwarded and the whole render fails.

    A renderer built before the code column answers rows of five fields
    and the code comes back empty, which every helper below treats as
    "fall back to the prose".
    """

    return _catalog_listing(renderer, wrfouts, store_root=store_root,
                            heavy=heavy, products=products, save_import=save_import)


#: The longest command line a renderer is started with.  Windows starts
#: no process whose command line passes 32,767 UTF-16 units, and a series
#: names every frame of one grid: 48 hours of a 15 minute nest under an
#: ordinary Documents folder is about 29,000 of them, and a few more
#: hours or a longer folder name passes the limit.  Kept a little under
#: it on every platform, so the long-series launch is the same code
#: wherever it is tested.
COMMAND_LINE_BUDGET = 32_000

#: Options whose value is a path the renderer opens.
_PATH_OPTIONS = ("--store-root", "--out-dir", "--overlays", "--annotate",
                 "--save-import", "--from-import")
#: Environment names whose value is a path the renderer opens.
_PATH_ENV = ("RUSTWX_BASEMAP_DIR", "RUSTWX_ASSETS_DIR", "RUSTWX_THEME",
             "RUSTWX_DIFFERENCE_TABLE")

#: The difference bars' table, by environment
#: (``rustwx_render::difference::DIFFERENCE_TABLE_ENV``): ``table`` (the
#: built-in table, the default), ``auto`` (every bar fitted to its own
#: difference) or the path of a JSON file whose top-level keys replace the
#: built-in table's.  ``woof render --diff-table`` sets it for the
#: differences it draws.  Under every setting a table bar that would draw
#: every cell in its neutral colour is fitted to the difference instead.
DIFFERENCE_TABLE_ENV = "RUSTWX_DIFFERENCE_TABLE"
DIFFERENCE_TABLE_CHOICES = ("table", "auto")

#: The renderer's radar colour set, by environment: ``standard`` (the radar
#: tables) or ``classic`` (the reflectivity ladder and blue-red velocity
#: scale before 2.8.5). Every Rust door that draws a radar-table product
#: reads it (``rustwx_render::RADAR_COLORS_ENV``); ``woof render
#: --radar-colors`` sets it for the renders it starts.
RADAR_COLORS_ENV = "RUSTWX_RADAR_COLORS"


def _names_a_file(value: str) -> bool:
    """A theme spelled as a file rather than one of the built-in names."""

    return os.path.isfile(value) and (
        "/" in value or os.sep in value or value.lower().endswith(".json"))


def _command_units(command) -> int:
    """Windows command-line length, including UTF-16 surrogate pairs."""

    return len(subprocess.list2cmdline(command).encode("utf-16-le")) // 2


class _InputInventoryRequired(ValueError):
    """Path shortening did not fit the complete native input series."""


def fit_series_command(command: list[str], inputs: int,
                       env: dict[str, str], *, inventory: Path | None = None
                       ) -> tuple[list[str], str | None, dict[str, str]]:
    """``(command, cwd, env)`` for a renderer launch over a series.

    The last ``inputs`` entries of ``command`` are the frames.  A command
    that fits :data:`COMMAND_LINE_BUDGET` is returned exactly as given,
    with no working folder.  One that does not is started in the folder
    the frames share, each frame named relative to it (a bare file name
    when they share one folder, about a fifth of the length); every other
    path the renderer is given, on the command line or in its
    environment, is made absolute first, so it still names the same file.
    The reduced command is checked in UTF-16 units too. If it still does
    not fit, or the frames share no drive, ``inventory`` receives the
    complete ordered path list. The launch owns and removes that file.
    A caller without an inventory must use :func:`_series_command` to
    manage its lifetime. The renderer reads no other path relative to
    where it starts.
    """

    if _command_units(command) <= COMMAND_LINE_BUDGET:
        return command, None, env
    frames = [os.path.abspath(item) for item in command[len(command) - inputs:]]
    try:
        base = os.path.commonpath([os.path.dirname(frame) for frame in frames])
    except ValueError:
        base = None
    head = list(command[:len(command) - inputs])
    head[0] = os.path.abspath(head[0])
    for index in range(1, len(head) - 1):
        option = head[index]
        value = head[index + 1]
        if option in _PATH_OPTIONS or (
                option == "--theme" and _names_a_file(value)):
            head[index + 1] = os.path.abspath(value)
    moved = dict(env)
    for name in _PATH_ENV:
        value = moved.get(name)
        if value and (os.path.isdir(value) or _names_a_file(value)):
            moved[name] = os.path.abspath(value)
    if base is not None:
        shortened = [*head, *(os.path.relpath(frame, base) for frame in frames)]
        if _command_units(shortened) <= COMMAND_LINE_BUDGET:
            return shortened, base, moved
    if inventory is None:
        raise _InputInventoryRequired(
            "the complete renderer input series exceeds the Windows "
            "command-line budget after path shortening; an input inventory "
            "is required")
    import json

    inventory = inventory.resolve()
    reduced = [*command[:-inputs], "--inputs-json", str(inventory)]
    if _command_units(reduced) > COMMAND_LINE_BUDGET:
        raise ValueError(
            "renderer options and the input inventory path exceed the "
            "Windows command-line budget")
    inventory.write_text(json.dumps(frames, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    return reduced, None, env


@contextmanager
def _series_command(command: list[str], inputs: int, env: dict[str, str]):
    """Keep an ordered native input file only while its process is running."""

    try:
        fitted = fit_series_command(command, inputs, env)
    except _InputInventoryRequired:
        import tempfile

        with tempfile.TemporaryDirectory(prefix="gpuwm-native-inputs-") as folder:
            yield fit_series_command(command, inputs, env,
                                     inventory=Path(folder) / "inputs.json")
    else:
        yield fitted


def list_products_series(renderer: Path, wrfouts, *, store_root: Path,
                         heavy: bool = False
                         ) -> tuple[list[tuple[str, str, str, str]], str]:
    """Ask native availability after importing the complete selected timeline."""
    rows, summary = _catalog_listing(renderer, wrfouts, store_root=store_root,
                                     heavy=heavy)
    return [row[:4] for row in rows], summary


def _catalog_listing(renderer: Path, wrfouts, *, store_root: Path,
                     heavy: bool = False, products: str | None = None,
                     save_import: Path | None = None,
                     ) -> tuple[list[tuple[str, str, str, str, str]], str]:
    """One parse of the renderer's catalog mode, for both accessors."""
    inputs = [Path(path) for path in wrfouts]
    if not inputs:
        raise ValueError("a product availability series needs history files")
    wrfout = inputs[-1]
    command = [
        str(renderer),
        "--store-root", str(store_root),
        # Required by the CLI contract but never written in list mode.
        "--out-dir", str(store_root),
        "--list-products",
    ]
    if heavy:
        command.append("--heavy")
    if products is not None:
        command.extend(["--products", products])
    if save_import is not None:
        command.extend(["--save-import", str(save_import)])
    command.extend(str(path) for path in inputs)
    try:
        with _series_command(command, len(inputs), renderer_env()) as fitted:
            command, cwd, env = fitted
            result = subprocess.run(
                command, capture_output=True, text=True, errors="replace",
                env=env, cwd=cwd)
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"{wrfout}: renderer failed to launch: {error}") from error
    if result.returncode != 0:
        tail = [line for line in (result.stderr or "").splitlines()
                if line.strip()]
        raise RuntimeError(
            f"{wrfout}: {tail[-1] if tail else f'exit {result.returncode}'}")
    rows: list[tuple[str, str, str, str, str]] = []
    summary = ""
    for line in (result.stdout or "").splitlines():
        if line.startswith("PRODUCT\t"):
            # FIVE OR MORE, never exactly five: the row grew a machine
            # code, and a parser that demanded the old width dropped
            # every row of a current build and then reported that the
            # renderer produced no catalog at all.
            parts = line.split("\t")
            if len(parts) >= 5:
                rows.append((parts[1], parts[2], parts[3], parts[4],
                             parts[5] if len(parts) > 5 else ""))
        elif line.startswith("CATALOG "):
            summary = line[len("CATALOG "):]
    if not rows:
        raise RuntimeError(
            f"{wrfout}: renderer produced no catalog rows")
    # The store's DEDUPED generic variables, which the listing reports on
    # stderr: a variable the store carries whose grid a named product
    # already draws, so the catalog drops the duplicate `var:` row.  They
    # are folded in as renderable rows because the question a consumer
    # asks of a `var:` token is "does this store carry the variable?" --
    # and for these the answer is yes.  Without them, a listing's silence
    # about a present-but-deduped variable reads as absence, and a door
    # that drops on absence would refuse a spelling this build accepts.
    #
    # ONLY the deduped form.  The emitter prints this row for two
    # OPPOSITE outcomes (`rusty-weather/src/batch_render.rs`): the
    # deduped one, which is drawable, and a stored variable whose NAME
    # no request can carry, printed escape_debug'd because such a name
    # holds commas, control characters or edge whitespace.  Folding the
    # second in would make `woof render --list-products` print it as
    # renderable, which is the opposite of what the row says.  No real
    # request can match that name, so the skip refuses nothing; the
    # listing simply stops claiming a spelling it cannot accept.
    for line in (result.stderr or "").splitlines():
        if not line.startswith("GENERIC_EXCLUDED\t"):
            continue
        parts = line.split("\t")
        if len(parts) < 2 or not parts[1]:
            continue
        reason = parts[2] if len(parts) > 2 else ""
        if reason.startswith(_GENERIC_NAME_REFUSED):
            continue
        rows.append((f"{GENERIC_VAR_PREFIX}{parts[1]}", "generic",
                     "renderable",
                     reason or "stored variable already drawn by a named "
                     "product",
                     "generic-deduped"))
    return rows, summary


def catalog_code(row) -> str:
    """One row's machine code, or ``""`` for a build that emits none."""

    return row[4] if len(row) >= 5 else ""


def catalog_verdict(rows, requested) -> tuple[str, list[tuple[str, str]]]:
    """``(the spec that can be drawn, the named skips)`` for one request.

    The availability listing has already been paid for by the time a
    door has rows in hand; this reads them.  Every requested slug whose
    row is renderable stays in the spec, in the order it was asked for;
    every requested slug whose row is anything else comes back paired
    with the engine's OWN detail, verbatim, so the reader gets the
    engine's reason rather than a Python paraphrase of it.

    A token with no row at all passes through untouched: the group
    keywords (``all``, ``direct`` ...), the section and mesh families
    and any slug this build knows and this listing did not mention.  The
    engine is the authority on an unknown slug, and guessing here would
    refuse a spelling this build accepts.

    :data:`GENERIC_VAR_PREFIX` is the ONE exception, and it is not a
    guess.  The generic catalog enumerates the store: every stored 2-D
    variable appears, as its own row or as a deduped one folded in by
    the listing.  So a ``var:`` token with no row is PROVEN absent from
    this store, and forwarding it gets ``stored 2-D variable "X" does
    not exist`` and a failed invocation -- which is how a shipped
    preset naming two variables in wrfout spelling rather than store
    spelling discarded a whole 13-frame series.

    That proof needs the enumeration to have RUN, so it is taken from
    the listing rather than assumed: the exception applies only when
    this listing carried a generic row at all.  A listing with none has
    said nothing about the store's variables -- a build without the
    generic enumeration is one way to get one -- and reading its silence
    as absence would drop every ``var:`` request a user made, by guess,
    with the engine never asked.  Those are forwarded unchanged and the
    renderer decides, the same way :func:`catalog_rows` treats a build
    whose rows carry no machine code.

    An empty spec is the caller's signal to refuse before launching
    rather than to run an empty render.
    """

    wanted = (product_spec_terms(requested) if isinstance(requested, str)
              else [str(token).strip() for token in requested])
    wanted = [token for token in wanted if token]
    status = {row[0]: (row[2], row[3], catalog_code(row)) for row in rows}
    # Whether this listing enumerated the store's 2-D variables at all,
    # which is what makes an absent row a reading rather than a silence.
    enumerated = any(row[1] == "generic"
                     or row[0].startswith(GENERIC_VAR_PREFIX)
                     for row in rows)
    available, excluded = [], []
    for token in dict.fromkeys(wanted):
        row = status.get(token)
        if row is None:
            if enumerated and token.startswith(GENERIC_VAR_PREFIX):
                excluded.append((
                    token,
                    "no stored 2-D variable "
                    f"{token[len(GENERIC_VAR_PREFIX):]!r} in this store; "
                    "woof render --list-products names the ones there are"))
                continue
            available.append(token)
            continue
        if row[0] == "renderable" or row[2] == "invalid-active-fire-domain":
            available.append(token)
            continue
        excluded.append((token, row[1]))
    return ",".join(available), excluded


def split_section_spec(products: str) -> tuple[str, list[str]]:
    """``(the store spec, the xsec: terms)`` of one --products spelling.

    The engine's own split, mirrored: ``rw-wrfbatch/src/section.rs``
    ``split_product_spec``, over the terms :func:`product_spec_terms`
    reads.  A group keyword names no section at all.

    It exists so a door can answer "does this request need a section
    line?" without a renderer and without a wrfout, which is what a plan
    review has.
    """

    trimmed = (products or "").strip()
    if trimmed.lower() in _GROUP_KEYWORDS:
        return trimmed, []
    tokens = product_spec_terms(trimmed)
    store = [token for token in tokens if not token.startswith(SECTION_PREFIX)]
    sections = [token for token in tokens if token.startswith(SECTION_PREFIX)]
    return ",".join(store), sections


def product_spec_terms(products: str) -> list[str]:
    """The products of one --products spelling, each section term whole.

    THE tokenizer for a product list, mirroring the engine's
    (``rw-wrfbatch/src/section.rs`` ``split_product_spec``).  A level list
    inside a section term is comma-separated too (``xsec:wa=1,2,5@5``),
    so a token that follows a section term whose last term opened a
    level list, and that starts with a level, is that list's
    continuation and not a product.  The continuation may carry the term
    that closes the list (``0.1/wa`` in ``xsec:QCLOUD=0.01,0.1/wa``).

    Splitting on every comma instead is what broke three requests: a
    level list's numbers read as products, so they were deduplicated
    across two sections (the second section's colour range changed
    without a word), and a closing ``0.1/wa`` became a store product the
    renderer refused, taking every other product of the invocation with
    it.
    """

    tokens: list[str] = []
    for token in (part.strip() for part in (products or "").split(",")):
        if not token:
            continue
        if (tokens and tokens[-1].startswith(SECTION_PREFIX)
                and _level_list_open(tokens[-1])
                and _continues_level_list(token)):
            tokens[-1] = f"{tokens[-1]},{token}"
            continue
        tokens.append(token)
    return tokens


def _level_list_open(token: str) -> bool:
    """True when the token's last term opened an ``=`` level list."""

    last = token.rsplit("/", 1)[-1]
    return "=" in last and "@" not in last.rsplit("=", 1)[-1]


def _continues_level_list(token: str) -> bool:
    """A level (``-10``, ``0.5``, ``10@5``), or the list's last level
    followed by the term that closes it (``0.1/wa``, ``10@5/tk=-20``)."""

    level, separator, rest = token.partition("/")
    if separator:
        return _is_level_token(level) and bool(rest.strip())
    return _is_level_token(token)


def _is_level_token(token: str) -> bool:
    """True for a level with an optional highlight (``5``, ``-2.5``,
    ``1e-2``, ``10@5``), never a slug: what the engine parses as a number."""

    level, separator, highlight = token.partition("@")
    return _engine_number(level) and (not separator or _engine_number(highlight))


def _engine_number(text: str) -> bool:
    # The engine reads a level with Rust's ``str::parse::<f32>``, which
    # takes no digit separators; Python's float() does.
    if not text or "_" in text:
        return False
    try:
        float(text.strip())
    except ValueError:
        return False
    return True


#: The height range ``rw_wrfbatch`` accepts for ``--section-top-km``
#: (``tools/rustwx/crates/rw-wrfbatch/src/main.rs``), and the ceiling it
#: uses when the caller names none.
SECTION_TOP_KM_RANGE = (1.0, 40.0)
SECTION_TOP_KM_DEFAULT = 14.0


def section_top_problem(top_km) -> str | None:
    """The refusal for a section ceiling outside 1-40 km, or ``None``.

    The engine's own sentence, because the engine is the authority on
    its own grammar, and it fires at the door so a caller reads it
    before a render launches rather than after.

    This is a refusal about a VALUE, not about a request: a spec that
    names a cut with no line is not refused anywhere any more, it is
    dropped per product by :func:`drop_storeless_terms` and the rest of
    the request is drawn.
    """

    if top_km is None:
        return None
    low, high = SECTION_TOP_KM_RANGE
    try:
        value = float(top_km)
    except (TypeError, ValueError):
        return (f"--section-top-km '{top_km}' is not within "
                f"{low:g}-{high:g} km")
    if not (value == value and abs(value) != float("inf")
            and low <= value <= high):
        return (f"--section-top-km '{top_km}' is not within "
                f"{low:g}-{high:g} km")
    return None


#: The shortest line ``rw_wrfbatch`` cuts (``section.rs``
#: ``SectionLine::from_endpoints``), measured on its sphere
#: (``rustwx-cross-section/src/geo.rs``, 6371.0 km).
SECTION_MIN_LENGTH_KM = 1.0
_SECTION_EARTH_RADIUS_KM = 6371.0
_SECTION_FILE_KEYS = frozenset({"start", "end", "points", "extend_km", "label"})


def is_section_line(section) -> bool:
    """True when ``section`` is spelled ``lat,lon,lat,lon`` rather than a file.

    The engine's own test (``section.rs`` ``SectionLine::parse``): four
    comma-separated numbers are a line, anything else names a JSON file.
    """

    parts = [part.strip() for part in str(section or "").strip().split(",")]
    return len(parts) == 4 and all(_engine_number(part) for part in parts)


def section_line_problem(section) -> str | None:
    """The engine's refusal of a ``--section`` value, or ``None``.

    Mirrors ``rw-wrfbatch/src/section.rs`` ``SectionLine::parse``:
    ``lat,lon,lat,lon`` with each latitude within 90 degrees and the ends
    at least 1 km apart, or a readable JSON file holding ``start`` and
    ``end`` or a ``points`` polyline of at least two points (plus an
    optional ``extend_km`` and ``label``, and nothing else).  ``None``
    for no section at all.

    WHAT BREAKAGE THIS PREVENTS (gate law).  The renderer reads the line
    as its arguments are validated, so a line it cannot read fails the
    WHOLE invocation, every product of it, and on ``woof go`` that
    invocation runs only after the forecast has integrated.  Asked here,
    a door says so before anything is fetched.
    """

    if section is None:
        return None
    spec = str(section).strip()
    if not spec:
        return ("--section is empty; give lat,lon,lat,lon or a JSON file "
                "with {start, end} or a {points, extend_km} polyline")
    if is_section_line(spec):
        lat0, lon0, lat1, lon1 = (float(part) for part in spec.split(","))
        return (_section_point_problem(lat0, lon0)
                or _section_point_problem(lat1, lon1)
                or _section_length_problem((lat0, lon0), (lat1, lon1)))
    path = Path(spec)
    if not path.is_file():
        return (f"--section '{spec}' is neither 'lat,lon,lat,lon' nor a "
                "readable JSON file")
    import json

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as error:
        return f"section file {path}: {error}"
    if not isinstance(document, dict):
        return f"section file {path}: the file is not a JSON object"
    unknown = sorted(set(document) - _SECTION_FILE_KEYS)
    if unknown:
        return (f"section file {path}: unknown field(s) "
                + ", ".join(repr(key) for key in unknown)
                + "; a section file names start, end, points, extend_km "
                  "and label")

    def pair(value):
        if (isinstance(value, list) and len(value) == 2
                and all(isinstance(item, (int, float))
                        and not isinstance(item, bool) for item in value)):
            return float(value[0]), float(value[1])
        return None

    start, end = document.get("start"), document.get("end")
    points = document.get("points") or []
    if not isinstance(points, list):
        return f"section file {path}: points is not a list of [lat, lon]"
    polyline = [pair(point) for point in points]
    if any(point is None for point in polyline):
        return f"section file {path}: points is not a list of [lat, lon]"
    if start is not None and end is not None:
        ends = (pair(start), pair(end))
        if None in ends:
            return f"section file {path}: start and end are each [lat, lon]"
    elif start is None and end is None and len(polyline) >= 2:
        ends = (polyline[0], polyline[-1])
    else:
        return (f"section file {path}: give both start and end, or a points "
                "polyline with at least two points")
    extend = document.get("extend_km")
    if extend is not None and (isinstance(extend, bool)
                               or not isinstance(extend, (int, float))):
        return f"section file {path}: extend_km is a number of km"
    label = document.get("label")
    if label is not None and not isinstance(label, str):
        return f"section file {path}: label is text"
    problem = (_section_point_problem(*ends[0])
               or _section_point_problem(*ends[1]))
    if problem is not None:
        return problem
    if extend is not None and float(extend) > 0:
        # Extended past both ends, the line is at least twice that long.
        return None
    return _section_length_problem(*ends)


def _section_point_problem(lat: float, lon: float) -> str | None:
    import math

    if (not math.isfinite(lat) or not math.isfinite(lon)
            or not -90.0 <= lat <= 90.0):
        return (f"section point {lat:g},{lon:g}: invalid geographic "
                "coordinate (latitude within 90 degrees, both finite)")
    return None


def _section_length_problem(start, end) -> str | None:
    import math

    lat0, lon0 = (math.radians(value) for value in start)
    lat1, lon1 = (math.radians(value) for value in end)
    hav = (math.sin((lat1 - lat0) / 2) ** 2
           + math.cos(lat0) * math.cos(lat1)
           * math.sin((lon1 - lon0) / 2) ** 2)
    length = 2 * _SECTION_EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(hav)))
    if length < SECTION_MIN_LENGTH_KM:
        return "--section endpoints are less than 1 km apart"
    return None


#: The shortest across-line frame ``rw_wrfbatch`` accepts for
#: ``--section-across`` (``tools/rustwx/crates/rw-wrfbatch/src/main.rs``).
SECTION_ACROSS_KM_MIN = 2.0


def section_across_problem(across_km) -> str | None:
    """The refusal for an across-line length under 2 km, or ``None``.

    The engine's own sentence, fired at the door for the reason
    :func:`section_top_problem` is: a NaN, an infinity or a negative
    length used to be forwarded, and the engine refused it only after
    the render had launched and opened its frames.
    """

    if across_km is None:
        return None
    try:
        value = float(across_km)
    except (TypeError, ValueError):
        value = float("nan")
    if not (value == value and abs(value) != float("inf")
            and value >= SECTION_ACROSS_KM_MIN):
        return (f"--section-across '{across_km}' is not a length of at "
                f"least {SECTION_ACROSS_KM_MIN:g} km")
    return None


#: The generic families drawn from a MESH file rather than from a
#: history frame, read out of the pinned vocabulary marker.
MESH_PREFIXES: tuple[str, ...] = tuple(
    family for family in GENERIC_FAMILIES if family.startswith("mesh"))


#: Why a mesh term cannot be drawn from history frames, per term.
_MESH_TERM_REASON = (
    "mesh products are drawn from a mesh file's cell boundaries "
    "(rw_wrfbatch --mesh-grid FILE.nc); a history frame carries cell "
    "centres and no polygons, and this render door passes no mesh file. "
    "Ask for the instantaneous or windowed product of the same field, "
    "which is drawn from the frames themselves, or draw the mesh panel "
    "with rw_wrfbatch --mesh-grid FILE.nc directly")

#: Why a section term cannot be drawn without a line, per term.
_SECTION_TERM_REASON = (
    "a vertical section is cut along a LINE and this invocation composes "
    "none. Add --section lat,lon,lat,lon or --section FILE.json to draw "
    "it, or drop the term")


def drop_storeless_terms(products: str, *, section=None
                         ) -> tuple[str, list[tuple[str, str]]]:
    """``(the spec a wrfout render can carry, the terms dropped)``.

    Three families are not store products and cannot be decided from a
    store listing, so :func:`catalog_verdict` never sees them: ``mesh:``
    and ``meshdiff:`` read a mesh file's cell boundaries, and ``xsec:``
    is cut along a section line.  A request that names one of them on a
    door with no mesh file and no line is answered here, with nothing
    opened and nothing launched.

    WHAT BREAKAGE THIS PREVENTS (gate law).  The renderer's own answer
    to such a term is PER INVOCATION, taken before a single picture is
    drawn (``rw-wrfbatch/src/main.rs``): a ``mesh:`` term with no
    ``--mesh-grid`` is a usage error at argument validation, a ``mesh:``
    term beside store or ``xsec:`` products is refused at the entry to
    the batch render because the two families read different inputs, and
    an ``xsec:`` term with no ``--section`` is refused before the store
    render starts.  So ONE such term forwarded from a door costs every
    other requested product its pictures: measured on the shipped wheel,
    a two-product series naming ``mesh:cell_area`` exits 1 with no
    pictures at all.

    The drop is therefore per PRODUCT, which the renderer will not do
    for itself: the term goes, the rest of the request is drawn, and the
    caller reports each dropped term with the sentence returned beside
    it.  A caller left with an empty spec has nothing to draw and
    refuses rather than launching an empty render, the same signal
    :func:`catalog_verdict` gives.

    ``section`` is the line the caller will pass to the renderer.  With
    one, an ``xsec:`` term is drawable here and is kept; the mesh
    families have no such door, because no door in this package passes
    ``--mesh-grid``.

    A request that drops nothing comes back byte for byte.  One that
    drops something is rebuilt from the engine's own tokenization
    (:func:`split_section_spec`, so a section term's comma-separated
    level list stays one term), which groups the surviving section
    terms after the store terms; the renderer splits the spec into those
    same two lists before it draws, so the order between families is not
    a fact anything downstream reads.
    """

    store_spec, sections = split_section_spec(products)
    kept: list[str] = []
    dropped: list[tuple[str, str]] = []
    for term in (token.strip() for token in store_spec.split(",")):
        if not term:
            continue
        if term.lower().startswith(MESH_PREFIXES):
            dropped.append((term, _MESH_TERM_REASON))
        else:
            kept.append(term)
    drawable_section = section is not None and str(section).strip() != ""
    for term in sections:
        if drawable_section:
            kept.append(term)
        else:
            dropped.append((term, _SECTION_TERM_REASON))
    if not dropped:
        return products, []
    return ",".join(kept), dropped


def parse_section_fill(line: str) -> dict | None:
    """One ``SECTIONFILL`` line as a receipt row, or ``None``.

    ``SECTIONFILL <slug> lo=<v> hi=<v> absence=<0|1> rule=<token>``.  A
    line whose numbers do not parse is dropped rather than raised on: a
    receipt is metadata beside a picture that was drawn, and a render
    that succeeded is not failed over a field a later engine spells
    differently.
    """

    if not line.startswith(SECTION_FILL_EVENT + " "):
        return None
    slug, _, rest = line[len(SECTION_FILL_EVENT) + 1:].partition(" ")
    fields = {}
    for token in rest.split():
        key, sep, value = token.partition("=")
        if sep:
            fields[key] = value
    try:
        low = float(fields["lo"])
        high = float(fields["hi"])
    except (KeyError, ValueError):
        return None
    if not slug or low != low or high != high:
        return None
    return {"family": slug, "lo": low, "hi": high,
            "absence": fields.get("absence") == "1",
            "rule": fields.get("rule", "")}


def _frame_of(reason: str, inputs) -> tuple[str, str]:
    """``(the input file this reason is about, the reason without it)``.

    The engine opens a per-frame reason with the path of the input its
    frame was read from, exactly as that path was put on its command line
    (events v2, ``frame-attributed``).  A reason it did not attribute is
    about the invocation, and is filed against the last input: the frame
    whose valid time the panels carry.
    """

    for path in inputs:
        for spelling in dict.fromkeys((str(path), os.path.abspath(path))):
            prefix = f"{spelling}: "
            if reason.startswith(prefix):
                return str(path), reason[len(prefix):]
    return str(inputs[-1]), reason


def frame_attributed(reason: str, inputs) -> str:
    """One skip reason, named against the input file of its own frame.

    WHAT BREAKAGE THIS PREVENTS (gate law): a 19-frame series render put
    the LAST input in front of every skip line, so
    ``wrfout_d01_2026-09-27_00_00_00: F000: 12-h QPF requires forecast
    hour >= 12`` named the F018 file against an F000 reason.  The engine
    names the frame's own file now; this keeps that name, and names the
    last input only for a reason no frame owns.
    """

    path, detail = _frame_of(reason, inputs)
    return f"{path}: {detail}"


#: How the engine prints one import note on stderr (``rw-wrfbatch``
#: ``import_note_line``).  Most notes are progress a render keeps quiet.
IMPORT_NOTE_PREFIX = "IMPORT_NOTE\t"

#: The opening of the import note the engine writes when it does not build a
#: frame's isobaric volumes (``volume_omission_note`` in
#: ``rw-wrfbatch/src/wrf_process.rs``), the clause of the ceiling cause and
#: the one reason that note carries when the frame has nothing to build them
#: from, and the family the lost products are reported under.  Every product
#: drawn off a pressure level (the 200 to 850 hPa charts) then leaves an
#: ``all`` request without a ``SKIPPED`` line, whatever stopped the volumes:
#: the working set passing the memory the host has available (never less
#: than 4 GiB; ``preflight_iso_volume_shape`` in ``wrf_volumes.rs``), or a
#: panic ``build_iso_volumes`` raised, which the engine isolates into the
#: same note.  A frame that stores no 3-D pressure has no such products to
#: draw: its note says ``variable not found`` (wrf-core's ``VarNotFound``)
#: and is not a skip.
PRESSURE_VOLUME_OMITTED_NOTE = "WRF 3-D pressure-volume products omitted"
PRESSURE_VOLUME_CEILING_CLAUSE = "host memory ceiling"
PRESSURE_VOLUME_ABSENT_FIELD_CLAUSE = "variable not found"
PRESSURE_LEVEL_FAMILY = "pressure-level products"

#: The snowfall windows the engine accumulates over a frame's history
#: (``PRODUCTS`` in ``rw-wrfbatch/src/snowfall.rs``), and how its note says
#: one was not built.  Such a product leaves an ``all`` request without a
#: ``SKIPPED`` line, like the pressure-level family.  THE BREAKAGE: a host
#: removed earlier history frames to save disk, and the windows that need
#: them vanished from the pictures with nothing in the render summary
#: naming the frame they needed; the engine's note names it.
SNOWFALL_WINDOW_SLUGS = frozenset((
    "snowfall_window", "snow_10to1_window", "snow_kuchera_window",
    "snowfall_liquid_equivalent", "snow_depth_total", "snow_depth_change"))
SNOWFALL_UNAVAILABLE_CLAUSE = " unavailable: "


def import_note_skip(note: str, inputs) -> tuple[str, str] | None:
    """An import note that took products out of the render, as a skip row.

    WHAT BREAKAGE THIS PREVENTS (gate law): a frame whose volumes the
    engine left out drew fewer pictures (43 where an 880x704x55 frame of
    the same forecast shape drew 67, when the ceiling was a fixed 4 GiB),
    and the render summary named no skipped or undrawn family, because the
    engine said it had omitted the pressure-level volumes only in an import
    note this bridge keeps quiet.  The row carries the engine's own note as
    its reason, so the summary says which family and why.  Any omission
    is a skip except one whose frame does not store the 3-D fields the
    volumes are built from: matching the ceiling's words alone left a
    panic in the volume builder as silent as the ceiling had been.
    A snowfall window the engine could not build (its history lacks a
    frame it needs) is a skip of that product, with the engine's note,
    which names the missing frame, as its reason.
    ``None`` for every other note.
    """

    slug, clause, _ = note.partition(SNOWFALL_UNAVAILABLE_CLAUSE)
    if clause and slug in SNOWFALL_WINDOW_SLUGS:
        return slug, frame_attributed(note, inputs)
    if (not note.startswith(PRESSURE_VOLUME_OMITTED_NOTE)
            or PRESSURE_VOLUME_ABSENT_FIELD_CLAUSE in note):
        return None
    return PRESSURE_LEVEL_FAMILY, frame_attributed(note, inputs)


def run_cf_renderer(renderer: Path, source: Path, *, variable: str,
                    out_dir: Path, valid: str | None = None,
                    bounds: str | None = None, table: Path | None = None,
                    contours: str | None = None, contour_file: Path | None = None,
                    observation: Path | None = None,
                    observation_variable: str | None = None,
                    overlays: Path | None = None, size: tuple[int, int] | None = None,
                    receipt: Path | None = None, theme: str | None = None) -> int:
    """Native CF planes, overlays and observation sheets. All data work is Rust."""
    command = [str(renderer), "--cf", str(source), "--variable", variable,
               "--out-dir", str(out_dir)]
    for flag, value in (("--valid", valid), ("--bounds", bounds),
                        ("--cf-table", table), ("--contours", contours),
                        ("--contour-file", contour_file),
                        ("--observation", observation),
                        ("--observation-variable", observation_variable),
                        ("--overlays", overlays), ("--receipt", receipt)):
        if value is not None:
            command.extend((flag, str(value)))
    if size is not None:
        command.extend(("--width", str(size[0]), "--height", str(size[1])))
    env = renderer_env()
    if theme is not None:
        env["RUSTWX_THEME"] = theme
    return subprocess.run(command, env=env, check=False).returncode


def run_renderer(renderer: Path, wrfout: Path, *, store_root: Path,
                 out_dir: Path, products: str, frames: str,
                 width: int | None = None, height: int | None = None,
                 heavy: bool = False,
                 source_label: str | None = None,
                 overlays: Path | None = None,
                 annotate: Path | None = None,
                 streamlines: bool | None = None,
                 theme: str | None = None,
                 section: str | None = None,
                 isotherms: str | None = None,
                 section_across_km: float | None = None,
                 section_size: tuple[int, int] | None = None,
                 section_top_km: float | None = None,
                 fills: list | None = None,
                 from_import: Path | None = None,
                 ) -> tuple[list[Path], list[str],
                            list[tuple[str, str]]]:
    """Render one wrfout file into ``out_dir``; (written, failures, skipped).

    One invocation per file with its own store root, exactly like the
    campaign flow: a shared store would merge two wrfouts into one run.
    The renderer's event lines are relayed to stdout as they arrive is
    not attempted -- the run is short-lived and its transcript is small,
    so it is captured and parsed for RENDERED/SKIPPED/FAILED events
    instead.

    The engine has always drawn the third distinction itself: a product
    whose stored fields are not there is ``SKIPPED <slug> <reason>`` on
    stdout and is *not* counted in ``summary.failed``, so the process
    still exits 0.  This function used to read only two of the three
    lines, which made an accurate skip invisible to every caller -- the
    reason a reader saw 53 images and no word about the 54th.  All three
    are read now, and only ``FAILED`` is a failure.

    ``fills``, when a list is given, collects one
    :func:`parse_section_fill` row per vertical cut drawn: the range its
    colour bar spans and the rule that set it.  It is an out-parameter
    rather than a fourth return value because every caller of these two
    functions already unpacks three.
    """

    return run_renderer_series(
        renderer, (wrfout,), store_root=store_root, out_dir=out_dir,
        products=products, frames=frames, width=width, height=height,
        heavy=heavy, source_label=source_label, overlays=overlays,
        annotate=annotate, streamlines=streamlines, theme=theme,
        section=section, isotherms=isotherms,
        section_across_km=section_across_km, section_size=section_size,
        section_top_km=section_top_km, fills=fills, from_import=from_import)


RESIDENT_ENSEMBLE_ABI = (
    "gpuwm-rw-wrfbatch-resident-ensemble-v2\tCDF5\tmean\tspread\tmin\tmax\tprobability\tpaintball\tpostage\tfraction\ttyped-diagnostic\tRENDERED\tFAILED")

CPU_ENSEMBLE_REDUCTION_ABI = (
    "gpuwm-ensemble-diagnostic-reduce.v1\tf32-words\tf64-two-pass\tmember-order\tsha256\tmember-weights-v1")


def find_ensemble_product_renderer(renderer: Path | None = None) -> Path | None:
    """Resolve the shared native product CLI through the ensemble executable.

    Member maps keep the ordinary renderer resolver. Both binaries dispatch
    this aggregate contract into the same Rust implementation, preserving
    typed diagnostic words and rendered bytes.
    """
    override_name = ("WOOF_RW_ENSBATCH" if os.environ.get("WOOF_RW_ENSBATCH")
                     else "WOOF_ENSEMBLE_RENDERER")
    override = os.environ.get(override_name)
    if override:
        selected = Path(override)
        if not selected.is_file():
            raise FileNotFoundError(f"{override_name} names a missing file: {selected}; "
                "point it at the built rw_ensbatch binary or unset the override")
        return bridges.accept_resolved(selected.resolve())
    filename = executable_name("rw_ensbatch")
    if renderer is not None:
        renderer = Path(renderer)
        if renderer.stem == "rw_ensbatch":
            return renderer
        sibling = renderer.with_name(filename)
        if sibling.is_file():
            return bridges.accept_resolved(sibling.resolve())
    for candidate in _candidates(override_name, "rw_ensbatch"):
        if candidate.is_file():
            return bridges.accept_resolved(candidate.resolve())
    return renderer


def require_ensemble_diagnostic_reducer(renderer: Path | None = None) -> Path:
    """Refuse an older native artifact before any member forecasts begin."""
    renderer = find_ensemble_product_renderer(renderer)
    if renderer is None:
        raise RuntimeError("the Rust CPU ensemble reducer is missing; build rw_ensbatch from this source tree")
    environment = renderer_env()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run([str(renderer), "--ensemble-diagnostic-reduce-abi"],
        capture_output=True, text=True, timeout=_PROBE_TIMEOUT_S, env=environment)
    if result.returncode or result.stdout.strip() != CPU_ENSEMBLE_REDUCTION_ABI:
        raise RuntimeError("the native artifact lacks the Rust CPU ensemble reduction contract; "
            "build rw_ensbatch from this source tree before starting the forecast")
    return renderer


def product_source_label() -> str:
    """``WOOF <executing version>``, the stamp woof.render.default_source_label
    puts on a plot, built here without importing woof.render.

    Breakage it prevents: this module ships in the RW-WPS preparation wheel,
    which leaves woof.render out, so an import of it here made
    tools/build_rw_wps_release.py refuse the wheel for an unresolved internal
    import.  tests/test_provenance_gate.py holds the two labels equal.
    """
    from woof.provenance import UNKNOWN_VERSION
    from woof.provenance_gate import executing_version

    try:
        version = executing_version()
    except Exception:                                   # noqa: BLE001
        return "WOOF"
    if not version or version == UNKNOWN_VERSION:
        return "WOOF"
    return f"WOOF {version}"


def run_cyclone_product_renderer(renderer: Path, manifest: Path):
    """Native moving-grid tracks, stamps, animation and weighted wind swaths."""
    import json
    schema = json.loads(Path(manifest).read_text()).get("schema")
    mode = "--tc-accumulator" if schema == "gpuwm.tc-accumulator.v1" else "--cyclone-products"
    result = subprocess.run([str(renderer), mode, str(manifest)],
                            capture_output=True, text=True, env=renderer_env())
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or 'Rust cyclone products failed')
    written = [line.split('\t', 1)[1] for line in result.stdout.splitlines() if line.startswith('RENDERED\t')]
    if not written or any(not Path(path).is_file() for path in written):
        raise RuntimeError('Rust cyclone products returned no durable pictures')
    return {'maps': written, 'manifest': str(manifest), 'renderer': str(renderer)}


def run_ensemble_product_renderer(renderer: Path, product_frame: Path, *,
                                  out_dir: Path, fields=(), domain="d01",
                                  width=1200, height=900, source_label=None,
                                  products=("mean", "spread", "min", "max", "prob", "paintball", "postage")):
    """Draw pre-reduced resident probability planes without member histories.

    The native mode reads the ensemble CDF5 contract directly. It never
    imports a fabricated WRF frame or recalculates ensemble reductions.
    """
    renderer = find_ensemble_product_renderer(renderer)
    if source_label is None:
        source_label = product_source_label()
    probe = subprocess.run([str(renderer), "--ensemble-products-abi"],
                           capture_output=True, text=True, timeout=_PROBE_TIMEOUT_S,
                           env=renderer_env())
    advertised = probe.stdout.strip().split("\t")
    required = RESIDENT_ENSEMBLE_ABI.split("\t")
    # S2S renderers add window/projection capabilities to this same contract.
    # Keep the version and every base capability mandatory, allowing extras.
    if (probe.returncode or advertised[0] != required[0]
            or not set(required[1:]).issubset(advertised[1:])):
        raise RuntimeError("the Rust renderer lacks the resident ensemble product contract; build rw_ensbatch from this source tree")
    command = [str(renderer), "--ensemble-products", str(product_frame),
               "--out-dir", str(out_dir), "--domain", str(domain),
               "--width", str(width), "--height", str(height),
               "--source-label", str(source_label), "--products", ",".join(products)]
    if fields:
        command.extend(("--fields", ",".join(fields)))
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, env=renderer_env())
    try:
        stdout, stderr = child.communicate()
        result = subprocess.CompletedProcess(command, child.returncode, stdout, stderr)
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
    relay_native_warnings(result.stderr)
    written = [Path(line.split("\t", 1)[1]) for line in result.stdout.splitlines()
               if line.startswith("RENDERED\t")]
    failures = [line.split("\t", 1)[1] for line in result.stderr.splitlines()
                if line.startswith("FAILED\t")]
    if result.returncode and not failures:
        failures.append((result.stderr.strip().splitlines() or
                         [f"renderer exited {result.returncode}"])[-1])
    if failures:
        raise RuntimeError("; ".join(failures))
    if not written or any(not path.is_file() for path in written):
        raise RuntimeError("resident probability render completed without all reported map files")
    _fold_ensemble_georefs(Path(out_dir), result.stdout)
    return written, result.stdout


def _fold_ensemble_georefs(out_dir: Path, stdout: str) -> None:
    """Each drawn map's transform (the renderer's ``GEOREF<TAB>path<TAB>json`` lines) into ``out_dir``'s
    ``render-georef.json``, keyed by the map's path under ``out_dir``, as every other render door records it.
    THE BREAKAGE THIS CLOSES: ensemble product maps carried no transform, so a click on one could not become a
    latitude and longitude (no soundings on 2,016 maps of a 20-member branched ensemble). Best effort: a record
    not written never fails the maps it describes."""
    import json
    from woof import render_georef
    panels = {}
    for line in stdout.splitlines():
        if not line.startswith("GEOREF\t"):
            continue
        try:
            _, path, text = line.split("\t", 2)
            key = Path(os.path.abspath(path)).relative_to(Path(os.path.abspath(out_dir))).as_posix()
            panels[key] = json.loads(text)
        except (ValueError, OSError):
            continue
    if panels:
        render_georef.fold(out_dir, {"schema": render_georef.GEOREF_SCHEMA, "panels": panels,
                                     "without_georeference": []})


def compose_ensemble_sheet(renderer: Path, rows, output: Path, *, title, valid_time):
    """Compose already rendered panels with the Rust renderer's sheet layout."""
    import json
    import tempfile
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = {"rows": [[str(Path(p).resolve()) for p in row] for row in rows],
                  "output": str(output.resolve()), "title": str(title),
                  "valid_time": str(valid_time)}
    with tempfile.TemporaryDirectory(prefix="ensemble-sheet-", dir=output.parent) as scratch:
        request = Path(scratch) / "request.json"
        request.write_text(json.dumps(descriptor))
        result = subprocess.run([str(renderer), "--ensemble-sheet", str(request)],
                                capture_output=True, text=True, env=renderer_env())
    if result.returncode or not output.is_file():
        raise RuntimeError(result.stderr.strip() or "Rust ensemble sheet produced no image")
    return output


def read_ensemble_diagnostic_rows(renderer: Path, path: Path, field: str, *, row: int, rows: int):
    """Read one bounded diagnostic slab through the Rust typed decoder.

    The native path transports float32 words directly. Numeric promotion
    would change NaN payloads and cannot serve member diagnostic replay.
    """
    import json
    import tempfile
    import numpy as np
    renderer = find_ensemble_product_renderer(renderer)
    with tempfile.TemporaryDirectory(prefix="gpuwm-ensemble-read-") as directory:
        output = Path(directory) / "words.bin"
        command = [str(renderer), "--ensemble-diagnostic-dump", str(path),
                   "--field", str(field), "--row", str(row), "--rows", str(rows),
                   "--output", str(output)]
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=900, env=renderer_env())
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or "Rust ensemble diagnostic read failed")
        record = json.loads(result.stdout)
        if (record.get("schema") != "gpuwm-ensemble-diagnostic-dump.v1"
                or record.get("dtype") != "<f4" or record.get("row") != row
                or record.get("rows") != rows or len(record.get("shape", ())) != 3):
            raise RuntimeError("Rust ensemble diagnostic read returned an incompatible slab contract")
        shape = tuple(record["shape"])
        if shape[1] != rows or any(not isinstance(n, int) or n < 1 for n in shape):
            raise RuntimeError("Rust ensemble diagnostic slab has invalid dimensions")
        if output.stat().st_size != int(np.prod(shape)) * 4:
            raise RuntimeError("Rust ensemble diagnostic read returned an incomplete stored-word slab")
        return np.fromfile(output, dtype="<f4").reshape(shape)


def reduce_ensemble_diagnostics(renderer: Path, *, packs, member_order, shape,
                                valid_time, requests, inventory, scratch_directory, member_weights=None):
    """Transport exact native CPU product words without Python field math."""
    import json
    import hashlib
    import tempfile
    import numpy as np
    renderer = find_ensemble_product_renderer(renderer)
    scratch = Path(scratch_directory).resolve()
    expected = {row["name"]: row for row in inventory}
    deleted = []
    with tempfile.TemporaryDirectory(prefix="cpu-reduce-", dir=scratch) as directory:
        directory = Path(directory)
        request_path = directory / "request.json"
        request = {"schema": "gpuwm-ensemble-diagnostic-reduce.request.v1",
            "member_order": list(member_order), "shape": list(shape),
            "valid_time": str(valid_time),
            "packs": [{"path": str(Path(pack["path"]).resolve()),
                "member_ids": list(pack["member_ids"])} for pack in packs],
            "fields": [{"field": item.field,
                "threshold_bits": np.asarray(item.thresholds, np.float32).view(np.uint32).tolist(),
                "comparison": item.comparison, "paintball": item.paintball,
                "spaghetti": item.spaghetti, "postage_stamp": item.postage_stamp} for item in requests]}
        if member_weights is not None:
            from woof.ensemble.member_weights import validate_weights
            request["member_weights"] = list(validate_weights(member_weights, len(member_order)))
        request_path.write_text(json.dumps(request, allow_nan=False) + "\n", encoding="utf-8")
        buffers = directory / "buffers"
        environment = renderer_env()
        environment["CUDA_VISIBLE_DEVICES"] = ""
        command = [str(renderer), "--ensemble-diagnostic-reduce", str(request_path),
            "--out-dir", str(buffers)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=900, env=environment)
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or "Rust CPU ensemble diagnostic reduction failed")
        receipt = json.loads(result.stdout)
        if (receipt.get("schema") != "gpuwm-ensemble-diagnostic-reduce.v1"
                or receipt.get("members") != len(member_order) or receipt.get("shape") != list(shape)):
            raise RuntimeError("Rust CPU ensemble reduction returned an incompatible roster or grid")
        if member_weights is not None and receipt.get("member_weights") != list(member_weights):
            raise RuntimeError("Rust CPU ensemble reduction did not confirm the declared member weights")
        outputs = {}
        for row in receipt.get("buffers", ()):
            name = row.get("name")
            spec = expected.get(name)
            if spec is None or name in outputs:
                raise RuntimeError("Rust CPU ensemble reduction returned an unexpected or duplicate product")
            path = Path(row["path"])
            if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(buffers.resolve()):
                raise RuntimeError("Rust CPU ensemble reduction buffer escapes its owned directory")
            dtype = np.dtype(row["dtype"])
            if (dtype != np.dtype(spec["dtype"]) or row.get("shape") != list(spec["shape"])
                    or row.get("bytes") != spec["payload_bytes"] or path.stat().st_size != spec["payload_bytes"]):
                raise RuntimeError("Rust CPU ensemble reduction returned an incompatible typed product buffer")
            def revision(stat):
                return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            with path.open("rb") as source:
                before = revision(os.fstat(source.fileno()))
                values = np.fromfile(source, dtype=dtype).reshape(spec["shape"])
                after = revision(os.fstat(source.fileno()))
                if os.name == "nt":
                    # Descriptor ctime is change time, while a Windows path
                    # stat may expose creation time. Compare like owners.
                    with path.open("rb") as addressed:
                        address = revision(os.fstat(addressed.fileno()))
                else:
                    address = revision(path.stat())
                if before != after or before != address or path.is_symlink():
                    raise RuntimeError("Rust CPU ensemble reduction buffer changed while its words were loaded")
            digest = hashlib.sha256(memoryview(values).cast("B")).hexdigest()
            if digest != row.get("sha256"):
                raise RuntimeError("Rust CPU ensemble reduction buffer differs from its committed SHA-256")
            outputs[name] = values
        if set(outputs) != set(expected):
            raise RuntimeError("Rust CPU ensemble reduction omitted a requested product buffer")
        for path in directory.rglob("*"):
            if path.is_file():
                deleted.append({"path": str(path), "bytes": path.stat().st_size})
    return outputs, deleted


def run_renderer_series(renderer: Path, wrfouts, *, store_root: Path,
                        out_dir: Path, products: str, frames: str,
                        width: int | None = None,
                        height: int | None = None, heavy: bool = False,
                        source_label: str | None = None,
                        overlays: Path | None = None,
                        annotate: Path | None = None,
                        streamlines: bool | None = None,
                        theme: str | None = None,
                        section: str | None = None,
                        isotherms: str | None = None,
                        section_across_km: float | None = None,
                        section_size: tuple[int, int] | None = None,
                        section_top_km: float | None = None,
                        fills: list | None = None,
                        from_import: Path | None = None,
                        threads: int | None = None,
                        ) -> tuple[list[Path], list[str],
                                   list[tuple[str, str]]]:
    """One invocation over a whole wrfout SERIES, into ONE store.

    ``rw_wrfbatch``'s CLI has always taken ``wrfout...``; what this adds
    is a Python caller that uses it.  :func:`run_renderer` renders one
    file per store because per-file isolation is the campaign
    convention, and that convention is exactly what a WINDOWED product
    cannot be drawn under: ``qpf_6h`` is a statement about two frames
    six hours apart ("F012 minus F006", in the engine's own catalog
    detail), so a store holding only F012 has nothing to difference
    against and the product is skipped.  The verification door's 6 h
    accumulation is precisely that shape -- two separate hourly wrfout
    files -- and it is why this exists rather than a fourth spelling of
    the same subprocess call.

    The store is the caller's to create and to remove; a series store is
    deliberately NOT per-file, because merging the frames into one run
    timeline is the whole point.

    Event parsing and the failure contract are :func:`run_renderer`'s,
    unchanged: RENDERED/SKIPPED on stdout, FAILED on stderr, and a
    nonzero exit with no FAILED line reported as one failure carrying
    the last stderr line.

    Each per-frame SKIPPED or FAILED line names the file of ITS frame:
    the engine opens the reason with that input's path
    (:func:`frame_attributed`).  This used to put the LAST input in front
    of every line, so a 19-frame series filed its F000 skips against the
    F018 file.  A line the engine did not attribute to a frame -- one
    about the invocation as a whole -- keeps the last input, the frame
    whose valid time the panels carry.
    """

    inputs = [Path(item) for item in wrfouts]
    if not inputs:
        raise ValueError(
            "a render series needs at least one wrfout file; an empty "
            "series would launch the renderer with no input and read its "
            "usage line as a failure")
    subject = inputs[-1]
    command = [
        str(renderer),
        "--store-root", str(store_root),
        "--out-dir", str(out_dir),
        "--products", products,
        "--frames", frames,
    ]
    if from_import is not None:
        command.extend(("--from-import", str(from_import)))
    # The default canvas follows the domain's shape. Pixel dimensions
    # explicitly request a fixed canvas for callers that tile panels.
    if (width is None) != (height is None):
        raise ValueError(
            "a render size is a width AND a height, or neither for the "
            "domain-shaped canvas; half a size would be filled with a guess")
    if width is not None:
        command.extend(("--layout", "fixed", "--width", str(width),
                        "--height", str(height)))
    if heavy:
        command.append("--heavy")
    if source_label:
        command.extend(("--source-label", source_label))
    # The wind layer, when the caller asked for one.  ``None`` adds no
    # argument at all, so an invocation that never mentions the wind is
    # byte-identical to every earlier release and
    # ``RUSTWX_WIND_STREAMLINES`` keeps its existing meaning.
    if streamlines is not None:
        command.append("--streamlines" if streamlines else "--barbs")
    # 2.5.0: map overlays in geographic degrees and panel annotations.
    # Absent, the renderer runs no overlay code and the PNGs are
    # byte-identical to every earlier build -- which is what
    # ``tools/rustwx_render_regression_gate.py`` gates.
    if overlays is not None:
        command.extend(("--overlays", str(overlays)))
    if annotate is not None:
        command.extend(("--annotate", str(annotate)))
    # The render theme: a built-in name (``default``, ``dark``) or a JSON
    # theme file.  ``None`` adds no argument, so an invocation that never
    # names a theme is byte-identical to every earlier release, and the
    # engine's own default look is what it draws.
    if theme is not None:
        command.extend(("--theme", str(theme)))
    # The vertical-section line for ``xsec:`` products, its isotherm set
    # and the optional across-line frame; every one is engine grammar,
    # forwarded verbatim so the engine's own refusals name the mistake.
    if section is not None:
        command.extend(("--section", str(section)))
    if isotherms is not None:
        command.extend(("--isotherms", str(isotherms)))
    if section_across_km is not None:
        command.extend(("--section-across", repr(float(section_across_km))))
    # The size a SECTION is drawn at.  Absent, the engine draws a section
    # landscape 2:1 at the map's width, so a caller that never mentions it
    # is byte-identical to every earlier release for MAP products and gets
    # a cut whose shape is not the map's.
    if section_size is not None:
        width, height = section_size
        command.extend(("--section-size", f"{int(width)}x{int(height)}"))
    # The ceiling of a section's fitted height range.  Absent, the engine
    # keeps its own 14 km, so a caller that never mentions it draws what
    # every earlier release drew.  A published cut was 14 km tall whatever
    # the air in it was doing, which puts a 1 km marine layer in the
    # bottom fourteenth of the frame; the flag existed in the engine from
    # the start and no door forwarded it.
    top_problem = section_top_problem(section_top_km)
    if top_problem is not None:
        raise ValueError(top_problem)
    if section_top_km is not None:
        command.extend(("--section-top-km", repr(float(section_top_km))))
    command.extend(str(path) for path in inputs)
    try:
        environment = renderer_env()
        if threads is not None:
            if type(threads) is not int or threads < 1:
                raise ValueError("renderer threads must be a positive integer")
            environment['RAYON_NUM_THREADS'] = str(threads)
        with _series_command(command, len(inputs), environment) as fitted:
            command, cwd, env = fitted
            result = subprocess.run(
                command, capture_output=True, text=True, errors="replace",
                env=env, cwd=cwd)
    except (OSError, ValueError) as error:
        return [], [f"{subject}: renderer failed to launch: {error}"], []
    # Started elsewhere, the renderer was handed an absolute output
    # folder and names its pictures under it; a caller that gave a
    # relative one gets its own spelling back.
    respell = cwd is not None and not Path(out_dir).is_absolute()
    written: list[Path] = []
    failures: list[str] = []
    skipped: list[tuple[str, str]] = []
    for line in (result.stdout or "").splitlines():
        if line.startswith("RENDERED "):
            _, _, rest = line.partition(" ")
            _, _, path = rest.partition(" ")
            if path:
                written.append(Path(os.path.relpath(path)) if respell
                               else Path(path))
        elif line.startswith("SKIPPED "):
            slug, _, reason = line[len("SKIPPED "):].partition(" ")
            skipped.append((slug, frame_attributed(
                reason or "no reason given", inputs)))
        elif fills is not None:
            row = parse_section_fill(line)
            if row is not None:
                fills.append(row)
    relay_native_warnings(result.stderr)
    for line in (result.stderr or "").splitlines():
        if line.startswith("FAILED "):
            slug, _, error = line[len("FAILED "):].partition(" ")
            path, detail = _frame_of(error, inputs)
            failures.append(f"{path}: {slug} {detail}")
        elif line.startswith(IMPORT_NOTE_PREFIX):
            # One row per distinct note.  The engine writes the note once
            # per frame and names no frame in it, so a series of frames
            # all over the volume ceiling filed the same row against the
            # last input once for every frame.
            skip = import_note_skip(line[len(IMPORT_NOTE_PREFIX):], inputs)
            if skip is not None and skip not in skipped:
                skipped.append(skip)
    if result.returncode != 0:
        tail = [line for line in (result.stderr or "").splitlines()
                if line.strip()]
        detail = tail[-1] if tail else f"exit {result.returncode}"
        if not failures:
            failures.append(f"{subject}: {detail}")
    return written, failures, skipped


#: The native difference receipt: product, units, range and range rule.
DIFFERENCE_EVENT = "DIFFERENCE"


def parse_difference_line(line: str) -> dict | None:
    """Read a drawn-product difference receipt, excluding run-level rows."""

    if not line.startswith(DIFFERENCE_EVENT + " "):
        return None
    fields = line[len(DIFFERENCE_EVENT) + 1:].split(" ")
    if not fields or "=" in fields[0]:
        return None
    row: dict = {"key": fields[0]}
    for item in fields[1:]:
        name, sep, value = item.partition("=")
        if sep:
            row[name] = value
    return row


@contextmanager
def _difference_command(command: list[str], inputs_a: list[Path],
                        inputs_b: list[Path], env: dict[str, str]):
    """Bound both complete difference timelines with launch-owned files."""

    direct = [*command,
              *(part for path in inputs_b for part in ("--diff-against", str(path))),
              *map(str, inputs_a)]
    if _command_units(direct) <= COMMAND_LINE_BUDGET:
        yield direct, env
        return
    import json
    import tempfile

    with tempfile.TemporaryDirectory(prefix="gpuwm-difference-inputs-") as folder:
        inventory_a = Path(folder) / "run-a.json"
        inventory_b = Path(folder) / "run-b.json"
        reduced = [*command, "--diff-inputs-json", str(inventory_b),
                   "--inputs-json", str(inventory_a)]
        if _command_units(reduced) > COMMAND_LINE_BUDGET:
            raise ValueError(
                "renderer difference options and inventory paths exceed the "
                "Windows command-line budget")
        for inventory, paths in ((inventory_a, inputs_a), (inventory_b, inputs_b)):
            inventory.write_text(
                json.dumps([os.path.abspath(path) for path in paths],
                           ensure_ascii=False) + "\n", encoding="utf-8")
        yield reduced, env


def run_renderer_difference(renderer: Path,
                            wrfout_a: Path | list[Path],
                            wrfout_b: Path | list[Path],
                            *, store_root: Path, out_dir: Path,
                            products: str, labels: tuple[str, str],
                            timeidx: int = 0,
                            sheet: bool = False,
                            source_label: str | None = None,
                            theme: str | None = None,
                            overlays: Path | None = None,
                            annotate: Path | None = None,
                            streamlines: bool | None = None,
                            width: int | None = None,
                            height: int | None = None,
                            difference_table: str | None = None,
                            ) -> tuple[list[Path], list[str],
                                       list[tuple[str, str]], list[dict]]:
    """Draw products as run A minus run B for one valid time.

    Each side accepts one file or its complete ordered history context.
    ``timeidx`` selects run A's ordinal in its full valid-time timeline;
    the native renderer selects run B's matching valid time. Windowed
    products retain earlier frames on both sides. Return pictures,
    failures, skips and native difference range receipts.
    ``difference_table`` is the :data:`DIFFERENCE_TABLE_ENV` value this
    render alone draws with (``None`` leaves the caller's environment).
    """

    inputs_a = ([Path(wrfout_a)] if isinstance(wrfout_a, (str, os.PathLike))
                else [Path(path) for path in wrfout_a])
    inputs_b = ([Path(wrfout_b)] if isinstance(wrfout_b, (str, os.PathLike))
                else [Path(path) for path in wrfout_b])
    if not inputs_a or not inputs_b:
        raise ValueError("a run difference needs history files for both runs")
    if isinstance(timeidx, bool) or not isinstance(timeidx, int) or timeidx < 0:
        raise ValueError("a run difference frame index must be a nonnegative integer")
    subject = inputs_a[-1]
    command = [
        str(renderer),
        "--store-root", str(store_root),
        "--out-dir", str(out_dir),
        "--products", products,
        "--diff-label-a", labels[0],
        "--diff-label-b", labels[1],
        "--frames", str(timeidx),
    ]
    if (width is None) != (height is None):
        raise ValueError(
            "a render size is a width AND a height, or neither for the "
            "domain-shaped canvas")
    if width is not None:
        command.extend(("--layout", "fixed", "--width", str(width),
                        "--height", str(height)))
    if sheet:
        command.append("--diff-sheet")
    if source_label:
        command.extend(("--source-label", source_label))
    if theme is not None:
        command.extend(("--theme", str(theme)))
    if overlays is not None:
        command.extend(("--overlays", str(overlays)))
    if annotate is not None:
        command.extend(("--annotate", str(annotate)))
    if streamlines is not None:
        command.append("--streamlines" if streamlines else "--barbs")
    env = renderer_env()
    if difference_table:
        env[DIFFERENCE_TABLE_ENV] = difference_table
    try:
        with _difference_command(command, inputs_a, inputs_b,
                                 env) as fitted:
            command, env = fitted
            result = subprocess.run(
                command, capture_output=True, text=True, errors="replace", env=env)
    except (OSError, ValueError) as error:
        return [], [f"{subject}: renderer failed to launch: {error}"], [], []
    written: list[Path] = []
    failures: list[str] = []
    skipped: list[tuple[str, str]] = []
    differences: list[dict] = []
    for line in (result.stdout or "").splitlines():
        if line.startswith("RENDERED "):
            _, _, rest = line.partition(" ")
            _, _, path = rest.partition(" ")
            if path:
                written.append(Path(path))
        elif line.startswith("SKIPPED "):
            slug, _, reason = line[len("SKIPPED "):].partition(" ")
            skipped.append((slug, reason or "no reason given"))
        else:
            row = parse_difference_line(line)
            if row is not None:
                differences.append(row)
    relay_native_warnings(result.stderr)
    for line in (result.stderr or "").splitlines():
        if line.startswith("FAILED "):
            slug, _, error = line[len("FAILED "):].partition(" ")
            path, detail = _frame_of(error, inputs_a)
            failures.append(f"{path}: {slug} {detail}")
    if result.returncode != 0 and not failures:
        tail = [line for line in (result.stderr or "").splitlines()
                if line.strip()]
        failures.append(f"{subject}: "
                        + (tail[-1] if tail else f"exit {result.returncode}"))
    return written, failures, skipped, differences


__all__ = [
    "CARGO_BUILD_HINT", "COMMAND_LINE_BUDGET", "RENDERER_ABI_MARKER",  # noqa: F822 -- module __getattr__
    "DIFFERENCE_EVENT", "parse_difference_line", "run_renderer_difference",
    "DIFFERENCE_TABLE_ENV", "DIFFERENCE_TABLE_CHOICES",
    "RENDERER_ENV", "fit_series_command",
    "NATIVE_WARNING_PREFIXES", "relay_native_warnings",
    "RENDERER_NAME", "basemap_dir",
    "basemap_candidates", "basemap_remedy", "companion_basemap_dir",
    "crate_dir", "find_renderer", "list_products",
    "probe_renderer", "renderer_candidates", "renderer_env",
    "renderer_remedy", "resolve_basemap_dir", "run_renderer",
    "run_renderer_series", "list_products_series",
    "GENERIC_FAMILIES", "GENERIC_VAR_PREFIX", "GROUP_KEYWORDS",
    "SECTION_PREFIX", "SECTION_TOP_KM_DEFAULT", "SECTION_TOP_KM_RANGE",
    "WINDOW_AXIS_CODES", "VARIABLES_KEYWORD", "frame_attributed",
    "IMPORT_NOTE_PREFIX", "PRESSURE_VOLUME_OMITTED_NOTE",
    "PRESSURE_VOLUME_ABSENT_FIELD_CLAUSE", "PRESSURE_VOLUME_CEILING_CLAUSE", "PRESSURE_LEVEL_FAMILY",
    "import_note_skip",
    "catalog_code", "catalog_rows",
    "catalog_verdict", "drop_storeless_terms",
    "MESH_PREFIXES", "split_section_spec", "product_spec_terms",
    "section_top_problem", "section_line_problem", "is_section_line",
    "SECTION_MIN_LENGTH_KM",
    "SECTION_FILL_EVENT", "parse_section_fill",
]


def run_hubwind_renderer(renderer: Path, manifest: Path):
    """Native WRF columns, table power, time statistics, tower verification and maps."""
    result = subprocess.run([str(renderer), '--hubwind-request', str(manifest)],
                            capture_output=True, text=True, env=renderer_env())
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or 'Rust hubwind products failed')
    maps = [line.split('\t', 1)[1] for line in result.stdout.splitlines()
            if line.startswith('RENDERED\t')]
    # A picture with no valid value (NO VALID TI / NO VALID DATA) is on disk
    # so a reader sees why, and is reported apart: never counted as drawn.
    undrawn = [tuple(line.split('\t', 2)[1:]) for line in result.stdout.splitlines()
               if line.startswith('UNDRAWN\t')]
    receipts = [line.split('\t', 1)[1] for line in result.stdout.splitlines()
                if line.startswith('RECEIPT\t')]
    if len(receipts) != 1 or not Path(receipts[0]).is_file():
        raise RuntimeError('native hubwind renderer did not return its receipt')
    if any(not Path(path).is_file() for path in maps):
        raise RuntimeError('native hubwind renderer returned missing pictures')
    if any(len(row) != 2 for row in undrawn):
        raise RuntimeError('native hubwind renderer returned a malformed undrawn row')
    return {'maps': maps, 'undrawn': [(path, reason) for path, reason in undrawn],
            'receipt': receipts[0], 'renderer': str(renderer), 'request': str(manifest)}


def run_solar_product_renderer(renderer: Path, request: Path):
    """Native solar computation, NOAA verification and production maps."""
    result = subprocess.run([str(renderer), "--solar-products", str(request)],
                            capture_output=True, text=True, env=renderer_env())
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or 'Rust solar products failed')
    specification = __import__('json').loads(request.read_text())
    receipt = Path(specification['out']) / 'receipt.json'
    if not receipt.is_file():
        raise RuntimeError('Rust solar products returned no receipt')
    maps = [line.split('\t', 1)[1] for line in result.stdout.splitlines() if line.startswith('RENDERED\t')]
    if any(not Path(path).is_file() for path in maps):
        raise RuntimeError('Rust solar products returned a missing picture')
    return {'maps': maps, 'receipt': str(receipt), 'renderer': str(renderer)}
