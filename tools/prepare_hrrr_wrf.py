"""Cross-platform native HRRR preparation and stock-WRF export wrapper."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from woof import explain
from woof.progress import prep_stage
from woof.physics_registry import canonical_template_id
from woof.physics_compat import (
    route_physics_profiles,
    KESSLER_PROFILE_ID,
    MORRISON_PROFILE_ID,
    MYNN_NOAHMP_PROFILE_ID,
    MYNN_NOAHMP_RTE_RRTMGP_PROFILE_ID,
    MYNN_PROFILE_ID,
    MYNN_RTE_RRTMGP_PROFILE_ID,
    MYNN_RUC_PROFILE_ID,
    MYNN_RUC_RTE_RRTMGP_PROFILE_ID,
    NOAHMP_PROFILE_ID,
    NSSL2_LEGACY_RRTMG_PROFILE_ID,
    NSSL2_PROFILE_ID,
    P3_LEGACY_RRTMG_PROFILE_ID,
    RUC_PROFILE_ID,
    THOMPSON_LEGACY_RRTMG_PROFILE_ID,
    THOMPSON_PROFILE_ID,
    THOMPSON_SHINHONG_LEGACY_RRTMG_PROFILE_ID,
    WSM6_PROFILE_ID,
)
from woof.hrrr_route_inputs import ROUTE_DEFAULT_PHYSICS_PROFILE
from woof.hrrr_forecast import (
    HRRR_TIME_FORMAT, hrrr_source_window, resolve_cycle_flags)
from woof.hrrr_native_static import sha256_file
from woof.hrrr_prepared_bundle import (
    EXPERIMENT_CONFIG_NAME as PORTABLE_EXPERIMENT_CONFIG_NAME,
    HrrrBundleError, publish_hrrr_prepared_bundle)
from woof.namelist_seal import namelist_extension_invariant
from woof.ingest.preprocess_backend import preprocessing_math_scope


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# The decode pipeline owns this bound and the sealed decoder binary
# enforces it; restating a number here is how this wrapper came to
# advertise 1..64 while the pipeline refused anything over 13 -- and to
# refuse it only after the static build, minutes in.
from tools.hrrr_pipeline import MAX_PIPELINE_WORKERS  # noqa: E402


def _pipeline_workers(value: str) -> str:
    """Parse-time validation for ``--pipeline-workers``.

    argparse raises on the spelling itself, so an unrunnable request
    costs a usage message rather than a static build.
    """
    text = str(value).strip().lower()
    if text == "auto":
        return text
    try:
        workers = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"pipeline-workers must be 1..{MAX_PIPELINE_WORKERS} or auto"
        ) from None
    if workers not in range(1, MAX_PIPELINE_WORKERS + 1):
        raise argparse.ArgumentTypeError(
            f"pipeline-workers must be 1..{MAX_PIPELINE_WORKERS} or auto "
            f"(the HRRR decoder accepts no more than "
            f"{MAX_PIPELINE_WORKERS})")
    return str(workers)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _write_json_create(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _link_file_create(source: Path, destination: Path) -> None:
    """Place ``source`` at a new ``destination``: a hard link, else a verified copy.

    On a drive with no hard links (exFAT), or across two drives, the file
    is copied and proven byte-identical instead, after a check that the
    copy fits; the copy costs its size in disk and the run says so once.
    This used to refuse, citing only that cost.
    """
    from woof.filesystem_paths import link_or_copy_verified

    destination.parent.mkdir(parents=True, exist_ok=True)
    link_or_copy_verified(source, destination)


def _reuse_physical_base_bridge(base: Path, destination: Path, *,
                                source_manifest_sha256: str,
                                static_cache: Path) -> str:
    """Reuse the base's sealed native authority without decoding GRIB again."""
    base = Path(base).resolve()
    header = json.loads((base / "native/prepared-cache/header.json").read_text(encoding="utf-8"))
    identity = header["identity"]
    from woof.ingest.prepared_cache import PreparedCacheReader
    PreparedCacheReader(base / "native/prepared-cache", expected_identity=identity)
    if identity["source_manifest_sha256"] != source_manifest_sha256.lower():
        raise ValueError("physical base preparation uses a different source manifest")
    if identity["static_cache_sha256"] != _sha256(static_cache):
        raise ValueError("physical base preparation uses different native static fields")
    bridge = base / "native/native-bridge"
    observed = _sha256(bridge / "SHA256SUMS")
    if observed != identity["bridge_manifest_sha256"]:
        raise ValueError("physical base bridge manifest differs from its prepared identity")
    paths = list(bridge.rglob("*"))
    for path in paths:
        if path.is_symlink():
            raise ValueError("physical base bridge may not contain symbolic links")
    _verify_manifest_payloads(bridge, _manifest_entries(bridge / "SHA256SUMS"))
    for path in paths:
        if path.is_file():
            _link_file_create(path, destination / path.relative_to(bridge))
    return observed


def _reuse_posted_static(args, output):
    """Reuse the original checked source geography for a provider member."""
    from woof.ensemble.posted_physical import PostedPhysicalProvider
    from woof.ensemble.posted_native import checked_source_inputs, copy_common_artifacts
    provider = PostedPhysicalProvider.open(args.physical_input_provider)
    context = provider.source_context(args.physical_member_index)
    checked = checked_source_inputs(
        context, source="hrrr", experiment_config=context.prepared_root / "experiment.toml",
        wps_namelist=context.prepared_root / "namelist.wps",
        physics_profile=args.physics_profile, expert_acknowledgements=tuple(args.ack),
        history_interval_seconds=args.history_interval_seconds)
    receipt_path = context.prepared_root / "native-static-receipt.json"
    expected = context.source_plan["manifest"]["files"]["static_receipt"]["sha256"]
    if _sha256(receipt_path) != expected:
        raise ValueError("shared HRRR static receipt changed after its source head")
    if args.static_cache is not None and _sha256(args.static_cache) != checked.cache_identity["static_cache_sha256"]:
        raise ValueError("posted member requested different static bytes from its checked source")
    if args.geog_root is not None:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if Path(receipt["geog_root"]).resolve() != args.geog_root.resolve():
            raise ValueError("posted member requested a different geography root from its checked source")
    copied = copy_common_artifacts(context, checked, output,
                                   geometry_name="native-geometry-receipt.json")
    geometry = copied["geometry"]["path"]
    receipt = output / "native-static-receipt.json"
    _link_file_create(receipt_path, receipt)
    context.verify()
    return copied["static"]["path"], receipt, geometry


def _manifest_entries(path: Path) -> dict[str, str]:
    """Parse one sha256sum document without forgiving unsafe inventory."""
    entries: dict[str, str] = {}
    for line_number, raw in enumerate(
            Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not raw:
            continue
        try:
            digest, name = raw.split(None, 1)
        except ValueError as exc:
            raise ValueError(
                f"malformed SHA256SUMS line {line_number}: {path}") from exc
        digest = digest.lower()
        name = name.removeprefix("*").removeprefix("./")
        candidate = Path(name)
        if (len(digest) != 64
                or any(character not in "0123456789abcdef"
                       for character in digest)
                or candidate.is_absolute() or ".." in candidate.parts
                or candidate.as_posix() != name or name in entries):
            raise ValueError(
                f"unsafe or duplicate SHA256SUMS entry on line "
                f"{line_number}: {path}")
        entries[name] = digest
    if not entries:
        raise ValueError(f"SHA256SUMS inventory is empty: {path}")
    return entries


def _verify_manifest_payloads(root: Path, entries: dict[str, str]) -> None:
    for name, digest in entries.items():
        payload = root / Path(name)
        if not payload.is_file():
            raise FileNotFoundError(payload)
        if _sha256(payload) != digest:
            raise ValueError(f"manifest payload changed: {payload}")


def _source_manifest_extension(*, predecessor: Path, extended: Path,
                               source_root: Path, old_hours, new_hours,
                               cycle: datetime, supplemental_paths=()) -> dict[str, object]:
    old_entries = _manifest_entries(predecessor)
    new_entries = _manifest_entries(extended)
    _verify_manifest_payloads(source_root, new_entries)
    changed = sorted(
        name for name, digest in old_entries.items()
        if new_entries.get(name) != digest)
    if changed:
        raise ValueError(
            "extended source manifest changes its sealed prefix: "
            + ", ".join(changed))
    added = sorted(set(new_entries) - set(old_entries))
    lead = int(new_hours[-1])
    expected_added = sorted((
        f"hrrr.t{cycle:%H}z.wrfnatf{lead:02d}.grib2",
        f"hrrr.t{cycle:%H}z.soilf{lead:02d}.grib2",
    ))
    expected_added += [
        path.resolve().relative_to(source_root.resolve()).as_posix()
        for path in supplemental_paths
        if path.resolve().relative_to(source_root.resolve()).as_posix() not in old_entries]
    expected_added = sorted(set(expected_added))
    if added != expected_added or list(new_hours) != list(old_hours) + [lead]:
        raise ValueError(
            "extended source manifest must add exactly the next atmosphere "
            "and soil objects plus explicitly declared supplement objects")
    return {
        "schema": "gpuwm-source-manifest-prefix-extension-v1",
        "predecessor_sha256": _sha256(predecessor),
        "extended_sha256": _sha256(extended),
        "old_source_forecast_hours": list(old_hours),
        "new_source_forecast_hours": list(new_hours),
        "suffix_source_forecast_hours": [int(old_hours[-1]), lead],
        "retained_entries": len(old_entries),
        "retained_entries_sha256": hashlib.sha256(
            _canonical(old_entries).encode("utf-8")).hexdigest(),
        "added_entries": [
            {"path": name, "sha256": new_entries[name]}
            for name in added
        ],
    }


_BRIDGE_HOUR = re.compile(r"^(atmosphere|soil)-f([0-9]{2})/")


def _bridge_manifest_extension(*, predecessor: Path, suffix: Path,
                               output: Path, old_hours, new_hours
                               ) -> dict[str, object]:
    """Publish a full bridge by linking the prefix and one suffix hour."""
    predecessor_manifest = predecessor / "SHA256SUMS"
    suffix_manifest = suffix / "SHA256SUMS"
    old_digest = _sha256(predecessor_manifest)
    suffix_digest = _sha256(suffix_manifest)
    old_entries = _manifest_entries(predecessor_manifest)
    suffix_entries = _manifest_entries(suffix_manifest)
    _verify_manifest_payloads(predecessor, old_entries)
    _verify_manifest_payloads(suffix, suffix_entries)
    old_hours = list(old_hours)
    new_hours = list(new_hours)
    lead = int(new_hours[-1])
    if new_hours != old_hours + [lead]:
        raise ValueError("bridge extension is not exactly one next hour")
    retained = {}
    for name, digest in old_entries.items():
        match = _BRIDGE_HOUR.match(name)
        if match is None or int(match.group(2)) not in old_hours:
            continue
        retained[name] = digest
    appended = {}
    for name, digest in suffix_entries.items():
        match = _BRIDGE_HOUR.match(name)
        if match is not None and int(match.group(2)) == lead:
            appended[name] = digest
    atmosphere_count = sum(
        name.startswith(f"atmosphere-f{lead:02d}/") for name in appended)
    soil_count = sum(
        name.startswith(f"soil-f{lead:02d}/") for name in appended)
    from woof.ingest.native_supplements import (gate_supplement_fields,
                                                gate_soil_surface_fields)
    suffix_gate = dict(row.split("\t", 1) for row in
                       (suffix / "gate.txt").read_text().splitlines() if row)
    suffix_fields = gate_supplement_fields(suffix_gate)
    # The cycle's PBL height (HPBL) is one more atmosphere payload when the
    # wrfnat publishes it (hrrr_grib2_bridge OPTIONAL_SURFACE_SPECS).
    from woof.ingest.native_supplements import gate_optional_surface_fields
    suffix_surface_fields = gate_optional_surface_fields(suffix_gate)
    suffix_soil_fields = gate_soil_surface_fields(suffix_gate)
    optional_soil_count = len(suffix_soil_fields)
    if suffix_fields and ("supplement-inventory.tsv" not in old_entries
                          or "supplement-inventory.tsv" not in suffix_entries):
        raise ValueError("supplement selection receipt is not bound by predecessor/suffix manifests")
    if (atmosphere_count != 22 + len(suffix_fields) + len(suffix_surface_fields)
            or soil_count != 2 + optional_soil_count):
        raise ValueError(
            "suffix bridge lacks its declared atmosphere/supplement and soil fields")
    if set(retained) & set(appended):
        raise ValueError("suffix bridge overwrites a retained payload")
    output.mkdir(parents=True, exist_ok=False)
    for name in sorted(retained):
        _link_file_create(predecessor / name, output / name)
    for name in sorted(appended):
        _link_file_create(suffix / name, output / name)
    prior_gate = dict(
        raw.split("\t", 1)
        for raw in (predecessor / "gate.txt").read_text(
            encoding="utf-8").splitlines() if raw)
    suffix_gate = dict(
        raw.split("\t", 1)
        for raw in (suffix / "gate.txt").read_text(
            encoding="utf-8").splitlines() if raw)
    if (prior_gate.get("forecast_hours")
            != ",".join(map(str, old_hours))
            or suffix_gate.get("forecast_hours")
            != ",".join(map(str, [old_hours[-1], lead]))):
        raise ValueError("bridge gates do not describe the proven windows")
    for key in set(prior_gate) & set(suffix_gate) - {
            "forecast_hours", "series_count"}:
        if prior_gate[key] != suffix_gate[key]:
            raise ValueError(f"suffix bridge changes immutable gate field {key}")
    if gate_supplement_fields(prior_gate) != suffix_fields:
        raise ValueError("supplement inventory changes the sealed prefix; prepare the complete window to change fields")
    if gate_soil_surface_fields(prior_gate) != suffix_soil_fields:
        raise ValueError("optional soil surface inventory changes the sealed prefix; prepare the complete window to change fields")
    prior_gate["forecast_hours"] = ",".join(map(str, new_hours))
    prior_gate["series_count"] = str(len(new_hours))
    gate_path = output / "gate.txt"
    gate_path.write_text(
        "".join(f"{key}\t{value}\n" for key, value in prior_gate.items()),
        encoding="utf-8", newline="\n")
    authority = {
        "schema": "gpuwm-native-hrrr-bridge-prefix-extension-v1",
        "predecessor_sha256": old_digest,
        "suffix_sha256": suffix_digest,
        "old_source_forecast_hours": old_hours,
        "new_source_forecast_hours": new_hours,
        "suffix_source_forecast_hours": [old_hours[-1], lead],
        "retained_entries_sha256": hashlib.sha256(
            _canonical(retained).encode("utf-8")).hexdigest(),
        "added_entries_sha256": hashlib.sha256(
            _canonical(appended).encode("utf-8")).hexdigest(),
    }
    _write_json_create(output / "extension-authority.json", authority)
    output_entries = {**retained, **appended}
    if suffix_fields:
        receipt_name = "supplement-inventory.tsv"
        old_rows = (predecessor / receipt_name).read_text().splitlines()
        new_rows = (suffix / receipt_name).read_text().splitlines()
        if not old_rows or not new_rows or old_rows[0] != new_rows[0]:
            raise ValueError("supplement inventory schema changes during extension")
        terminal_rows = [row for row in new_rows[1:] if int(row.split("\t", 1)[0]) == lead]
        if len(terminal_rows) != len(suffix_fields):
            raise ValueError("supplement inventory lacks the appended source time")
        receipt_path = output / receipt_name
        receipt_path.write_text("\n".join(old_rows + terminal_rows) + "\n", encoding="utf-8", newline="\n")
        output_entries[receipt_name] = _sha256(receipt_path)
    output_entries["gate.txt"] = _sha256(gate_path)
    output_entries["extension-authority.json"] = _sha256(
        output / "extension-authority.json")
    manifest_path = output / "SHA256SUMS"
    with manifest_path.open("x", encoding="utf-8", newline="\n") as stream:
        for name in sorted(output_entries):
            stream.write(f"{output_entries[name]}  ./{name}\n")
    merged_digest = _sha256(manifest_path)
    _verify_manifest_payloads(output, output_entries)
    if (_sha256(predecessor_manifest) != old_digest
            or _sha256(suffix_manifest) != suffix_digest):
        raise ValueError("bridge authority changed during extension")
    return {
        "schema": "gpuwm-bridge-manifest-prefix-extension-v1",
        "predecessor_sha256": old_digest,
        "suffix_sha256": suffix_digest,
        "extended_sha256": merged_digest,
        "old_source_forecast_hours": old_hours,
        "new_source_forecast_hours": new_hours,
        "suffix_source_forecast_hours": [old_hours[-1], lead],
        "retained_entries": len(retained),
        "retained_entries_sha256": authority[
            "retained_entries_sha256"],
        "added_entries": [
            {"path": name, "sha256": appended[name]}
            for name in sorted(appended)
        ],
    }

from woof.ingest.microphysics_cold_start import (
    cold_start_contract, cold_start_seeded_numbers,
)
from woof.physics_compat import single_domain_runtime_switches

#: The cold-start contract for every profile the NATIVE route offers.
#: Keyed off that route's own declaration rather than the shared
#: single-domain menu: this preparation serves one route, and a suite the
#: other fixed-template route offers has no native preparation here to
#: contract for.
_HRRR_COLD_START_CONTRACT = {
    profile: cold_start_contract(single_domain_runtime_switches(profile))
    for profile in route_physics_profiles(
        "tools.hrrr_single_domain_benchmark")
}


def _configuration_arguments(command, args):
    if args.physics_profile is not None:
        command.extend(("--physics-profile", args.physics_profile))
    if args.experiment_config is not None:
        command.extend(("--experiment-config", str(args.experiment_config.resolve())))
    if args.wps_namelist is not None:
        command.extend(("--wps-namelist", str(args.wps_namelist.resolve())))


def _run(command: list[str], env: dict[str, str],
         cwd: Path | None = None) -> None:
    subprocess.run(command, check=True, cwd=REPO if cwd is None else cwd,
                   env=env)


#: How much of a child's error text is kept so its refusal can be
#: repeated by the step that launched it.  A refusal is a paragraph; a
#: traceback in front of it is a few hundred characters more.
REFUSAL_TAIL = 8192


class _RepeatableStream:
    """Pass a child's text straight through, keeping a bounded tail.

    The pass-through is the whole point: the text still reaches this
    process's own stream, which is what the host's diagnostic log is
    fed from, so nothing about what a reader sees changes.  The tail
    exists so the step that launched the child can SAY what the child
    said, instead of pointing at a stream and trusting that the words
    are still on it where the reader is standing.
    """

    def __init__(self, destination) -> None:
        self.destination = destination
        self.tail = ""

    def write(self, text: str) -> int:
        self.destination.write(text)
        self.destination.flush()
        self.tail = (self.tail + text)[-REFUSAL_TAIL:]
        return len(text)

    def flush(self) -> None:
        self.destination.flush()


def _run_keeping_refusal(command: list[str], env: dict[str, str],
                         cwd: Path | None = None) -> tuple[int, str]:
    """Run a child and return its exit code WITH its own error text."""

    from woof.command_output import run_streamed

    recorded = _RepeatableStream(sys.stderr)
    completed = run_streamed(command, sys.stdout, recorded, env=env,
                             cwd=str(REPO if cwd is None else cwd))
    return completed.returncode, recorded.tail


#: Terminal styling, which a child emits whenever its own environment
#: asks for colour.  It is stripped from the KEPT copy only: the
#: pass-through keeps whatever the child wrote, and the copy this step
#: quotes has to be plain text, because the reader of a quoted refusal
#: is often a log file, and because the summariser matches an exception
#: line by its first character.
_TERMINAL_STYLING = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _refusal_sentence(text: str) -> str:
    """The child's own last refusal, with styling and stack frames gone."""

    from woof.prep_output import failure_summary

    plain = _TERMINAL_STYLING.sub("", text).strip()
    if not plain:
        return ""
    sentence = failure_summary(plain).strip()
    if not sentence:
        # Not every child dies through a traceback the summariser can
        # read.  The last thing it said is still what it said, and a
        # step that quotes nothing because the shape was unfamiliar is
        # the defect this exists to close.
        sentence = plain.splitlines()[-1]
    return " ".join(sentence.split())


def _stage_environment() -> dict[str, str]:
    """The environment every stage this preparation launches receives.

    Two keys are composed rather than inherited, and both are here
    because a child that differs from its parent in either one dies
    before it reads any of the user's data:

      * ``PYTHONPATH`` puts THIS tree ahead of whatever is installed,
        so the stage runs the code this preparation is made of;
      * ``WOOF_GIT_EXE`` names the git the parent has already found.
        Every stage binds ``runtime_manifest.provenance`` at start-up,
        that binding is a ``git`` subprocess on a source checkout, and
        a child whose environment does not carry git on ``PATH``
        cannot start one.  Measured: the parent resolved its identity,
        the benchmark it launched raised ``IdentityError`` on the same
        tree one process later.  Passing the resolved path fixes it
        without loosening ``PATH`` -- one key, one executable.
    """

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    try:
        from woof.provenance import GIT_EXE_ENV, git_executable

        resolved = git_executable()
        if resolved is not None:
            env[GIT_EXE_ENV] = resolved
    except Exception:                                   # noqa: BLE001
        # A preparation must not die because git could not be located
        # for its children; the child has its own ladder to fall down.
        pass
    return env


#: The vendored decoder workspace.  Its ``.cargo/config.toml`` is what
#: replaces crates.io with ``vendor/crates-io``, and cargo discovers that
#: file by walking UP from the working directory -- never from
#: ``--manifest-path``.  Building this crate from the repository root
#: therefore bypasses the vendored registry entirely and resolves against
#: crates.io, which on an air-gapped box fails with ``no matching package
#: named '<crate>' found; location searched: crates.io index``.  Every
#: documented build (README, install.sh, install.ps1, CONTRIBUTING) already
#: ``cd``s here; this one now does too.
#:
#: It exists in a source checkout and NOT in a wheel install: a wheel's
#: ``REPO`` is ``site-packages``, which has a ``tools`` package (this
#: file is in it) and no Rust workspace at all.  Building here is
#: therefore the developer path only; see :func:`_decoder`.
BRIDGE_CRATE = REPO / "tools" / "grib1_bridge"


def _decoder(env: dict[str, str], *, as_posted: bool = False) -> Path:
    """The HRRR decoder this preparation will launch.

    This used to resolve exactly one path --
    ``<REPO>/tools/grib1_bridge/target/release/hrrr_grib2_bridge`` --
    and, when it was absent, run ``cargo build`` in a directory that a
    wheel install does not have.  On a pip install ``REPO`` is
    ``site-packages``, so the bridge ``woof setup`` had already staged
    in ``~/.woof/bridges`` was never even looked at and a healthy
    machine refused with a path nobody had ever been told to create.

    The order now is the project's one bridge ladder
    (:func:`woof.bridges.resolve_source_decoder`): the
    ``WOOF_HRRR_DECODER`` override first and fail-loud, then a
    checkout's own build, then ``libexec/bridges``, then
    ``~/.woof/bridges``.  ``woof doctor`` calls the same function, so
    a green report and a runnable preparation are now the same claim --
    and since 1.8.9 that includes the contract check, which the resolver
    performs itself.  It used to sit outside, in doctor only, so a
    binary built before this release's contract passed this door and
    failed the report; a 12-byte text file planted at the override path
    was accepted here.  A stale bridge now refuses at both.

    Building from source stays available and stays last: it is the
    developer convenience it always was, and it can only run where the
    crate actually exists.

    ``env`` is the environment for that build subprocess.  Resolution
    itself reads the PROCESS environment, through the shared ladder --
    which is the point: ``woof doctor``, ingest and this wrapper must
    all be answering about the same machine, and a private copy of the
    environment is how three answers to one question start.
    """

    from woof import bridges

    # An explicit override is the operator's decision and never falls
    # through to anything -- least of all to a build, which would answer
    # a question about one exact file by producing a different one.
    overridden = bool(os.environ.get(bridges.BRIDGE_ENV["hrrr_grib2_bridge"]))
    mode = {"mode": "as_posted"} if as_posted else {}
    try:
        return bridges.resolve_source_decoder("hrrr", **mode)
    except FileNotFoundError:
        if overridden or not BRIDGE_CRATE.is_dir():
            raise
    # A checkout that has simply not built the crate yet: build it where
    # its vendored registry lives, then resolve through the same ladder
    # so the answer is the one every other consumer would have got.
    _run([
        "cargo", "build", "--release", "--locked", "--offline",
        "--bin", "hrrr_grib2_bridge",
    ], env, cwd=BRIDGE_CRATE)
    return bridges.resolve_source_decoder("hrrr", **mode)


#: The dispositions :func:`_stock_wrf_export` can RETURN.  "REFUSED" left
#: this tuple with the 2026-08-04 fail-closed ruling: a requested export
#: that fails now raises instead of returning a receipt, so a PASS report
#: can never again stand over missing WRF-arm outputs.
STOCK_WRF_EXPORT_STATES = ("PASS", "SKIPPED")


def _verified_export_outputs(output: Path) -> dict[str, object]:
    """The export's declared outputs, existing and hash-manifested.

    A PASS claim without its artifacts is a lying receipt: the 2026-08-04
    battery case run lost its WRF-arm inputs to an EMFILE inside the
    converter subprocess while the step's receipt said PASS-shaped things
    around it (out/node19-shakedown/b04/B04-EXPORT-REFUSAL.txt).  So PASS
    is now earned: the manifest must exist, every file it declares must
    exist, and the manifest's own digest rides in the receipt so a later
    reader can bind the file set this step vouched for.
    """

    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(
            "stock-WRF export exited 0 but left no manifest at "
            f"{manifest_path}; refusing to report PASS for outputs that "
            "do not exist")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared = manifest.get("files")
    if not isinstance(declared, dict) or not declared:
        raise RuntimeError(
            f"stock-WRF export manifest {manifest_path} declares no files")
    missing = sorted(name for name in declared
                     if not (output / name).is_file())
    if missing:
        raise RuntimeError(
            "stock-WRF export manifest declares file(s) missing on disk: "
            f"{missing} under {output}")
    return {
        "output": str(output),
        "manifest_sha256": _sha256(manifest_path),
        "files": sorted(declared),
    }


def _stock_wrf_export(command: list[str], env: dict[str, str], *,
                      skip: bool, required: bool,
                      output: Path) -> dict[str, object]:
    """Run the stock-WRF export: SKIPPED on request, else PASS or refuse.

    Two postures met here and the second replaced the first.  The
    original unconditional ``check=True`` took the whole wrapper down on
    any converter refusal, discarding a completed preparation; the fix
    for that swung to warn-and-continue -- and the 2026-08-04 battery
    case run measured the cost: the converter subprocess died on EMFILE
    (an environmental failure, not a representability gap), one warning
    scrolled past, the preparation PASS-logged, and the WRF arm had no
    inputs and no manifest
    (out/node19-shakedown/b04/B04-EXPORT-REFUSAL.txt).  The subprocess
    boundary flattens accurate typed refusals and environmental deaths
    into one returncode, so no middle posture can tell them apart.

    Fail-closed, ruled: a REQUESTED export either PASSes with its
    declared outputs verified on disk and hash-manifested
    (:func:`_verified_export_outputs`), or this step raises.  The
    prepared cache and preparation work remain on disk either way --
    nothing re-runs the expensive part -- and the deliberate opt-out is
    ``--skip-stock-wrf-export``, recorded in the receipt as SKIPPED
    rather than dressed as success.
    """

    if skip:
        return {
            "status": "SKIPPED",
            "reason": "--skip-stock-wrf-export",
            "required": False,
        }
    returncode, error_text = _run_keeping_refusal(command, env)
    if returncode == 0:
        # PASS is earned, not assumed: the declared outputs must exist
        # and hash-manifest, or this step refuses loudly.
        return {"status": "PASS", "required": required,
                **_verified_export_outputs(output)}
    # A requested export that fails is FATAL (2026-08-04 ruling, after
    # the battery case run): the converter died on EMFILE -- an
    # environmental failure a retry with a higher fd limit fixes, not a
    # representability gap -- and the old warn-and-continue posture
    # PASS-logged a preparation that had produced no WRF-arm inputs and
    # no manifest (out/node19-shakedown/b04/B04-EXPORT-REFUSAL.txt).
    # The subprocess boundary flattens the converter's accurate typed
    # refusals and environmental deaths into one returncode, so the only
    # fail-closed reading of a nonzero exit is refusal.  The explicit
    # opt-outs remain: --skip-stock-wrf-export skips the attempt up
    # front and says so in the receipt; a caller whose product really
    # is only the prepared cache passes it deliberately.
    #
    # The converter's own words travel WITH this refusal, rather than as
    # a pointer to a stream.  They used to be left "on stderr", which is
    # true of the terminal and useless to every reader that meets this
    # exit through something else: a caller that keeps the last lines of
    # a failed stage keeps THIS message and the frames under it, while
    # the sentence saying why sits further up, outside the window.  The
    # prepared cache and the preparation work above are complete and
    # remain on disk either way.
    refusal = _refusal_sentence(error_text)
    said = (f'  The converter said: "{refusal}"' if refusal else
            "  The converter wrote nothing to its error stream, which is "
            "itself the thing to report.")
    raise RuntimeError(
        f"stock-WRF export was requested and failed (exit {returncode}); "
        "refusing to PASS a preparation whose declared WRF-arm outputs "
        "were not produced.  The prepared cache and preparation report "
        f"above are complete and stand on disk.{said}  Fix that refusal "
        "and re-run, or pass --skip-stock-wrf-export to deliberately not "
        "produce WRF-arm inputs")


def _namelist_start_time(path: Path) -> datetime | None:
    """The ``&time_control`` start instant of a WRF namelist, if it has one."""

    from woof.namelist_import import parse_namelist

    try:
        control = parse_namelist(path).get("time_control", {})
        parts = {
            key: int(control[f"start_{key}"][0])
            for key in ("year", "month", "day", "hour", "minute", "second")}
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return None
    try:
        return datetime(**parts)
    except ValueError:
        return None


def _namelist_start_advisory(path: Path, model_start_time: datetime,
                             *, cycle: datetime, lead: int) -> None:
    """Say when the namelist's own start time disagrees with model time zero.

    The GPU single-domain route does not read this field -- it builds its
    experiment from ``--cycle`` plus ``--forecast-start-hour``, and the
    stock-WRF export below is handed the derived model start -- so a
    disagreement here changes nothing about THIS run and is not refused.
    It matters to the two consumers that do read it: a stock ``wrf.exe``
    driven from the same namelist, and ``woof.hrrr_hierarchy_direct``,
    which requires the namelist start to BE model time zero and refuses
    otherwise.  A run at lead K whose namelist still says the cycle hour
    is inconsistent by exactly K hours for both of them, so it is said
    out loud, once, with the value that would fix it.
    """

    observed = _namelist_start_time(path)
    if observed is None or observed == model_start_time:
        return
    remedy = (f"Re-emit it with `woof domain --forecast-start-hour {lead}`, "
              "or set its start_* keys to "
              f"{model_start_time:%Y-%m-%d %H:%M:%S}." if lead else
              "Set its start_* keys to "
              f"{model_start_time:%Y-%m-%d %H:%M:%S}.")
    explain.warn(
        f"{path.name} starts at {observed:%Y-%m-%d %H:%M:%S}, but this "
        f"preparation's model time zero is "
        f"{model_start_time:%Y-%m-%d %H:%M:%S} (cycle "
        f"{cycle:%Y-%m-%d %H:%M:%S} + {lead} h).  This run is unaffected "
        "-- the GPU route takes its clock from the cycle and the lead, "
        "not from that field -- but a stock wrf.exe driven from this "
        "namelist would start at the wrong hour, and "
        "woof.hrrr_hierarchy_direct requires the two to agree and will "
        f"refuse.  {remedy}",
        "The single-domain preparer has always ignored this field, so "
        "nothing that ran before starts failing now; before leads were "
        "reachable from the wizard the two instants could not disagree.")


def _positive_finite_seconds(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "history-interval-seconds must be a number") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError(
            "history-interval-seconds must be finite and positive")
    return value


def _positive_count_within(count, bound) -> bool:
    """Whether ``count`` is a whole positive count no larger than ``bound``."""

    def whole(value):
        return isinstance(value, int) and not isinstance(value, bool)

    return whole(count) and whole(bound) and 1 <= count <= bound


def _validated_worker_receipts(
        preparation_report: dict[str, object], *, selected_backend: str,
        requested_preprocess_workers: int | None,
        requested_pipeline_workers: str,
        final_hour: int, reused_physical_base: bool = False,
        reused_posted_source: bool = False) -> tuple[dict, dict, dict]:
    """Validate the public CPU budget and independent decoder receipt."""

    try:
        preprocess_receipt = preparation_report["preparation"][
            "preprocess_backend"]
        preprocess_worker_budget = preparation_report["preparation"][
            "preprocess_worker_budget"]
        pipeline_worker_receipt = preparation_report["pipeline"]["workers"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            "HRRR preparation omitted a worker/backend receipt") from exc
    if (preparation_report.get("status") != "PASS"
            or not isinstance(preprocess_receipt, dict)
            or preprocess_receipt.get("backend") not in {"cpu", "cuda"}):
        raise RuntimeError("HRRR preparation backend receipt is malformed")
    if (selected_backend != "auto"
            and preprocess_receipt["backend"] != selected_backend):
        raise RuntimeError(
            "HRRR preparation backend differs from the public selector")
    # A CPU count is held to AT MOST its request and budget, never to
    # equality: the backend caps its count by the CPUs and memory it
    # measures when it is built, and a busy host can narrow either after
    # the budget was taken (the host held cores 0-3 of a running session
    # scope).  "f00 mapping used 20 native workers; expected 24" refused
    # such a run although the count moves no byte.  More than the request
    # or the budget is the oversubscription these checks exist for.
    cpu_workers = preprocess_receipt.get("workers")
    if (requested_preprocess_workers is not None
            and preprocess_receipt["backend"] == "cpu"
            and not _positive_count_within(
                cpu_workers, requested_preprocess_workers)):
        raise RuntimeError(
            "HRRR CPU preparation worker receipt differs from the request")
    if (not isinstance(preprocess_worker_budget, dict)
            or preprocess_worker_budget.get("schema")
            != "gpuwm-preprocess-worker-budget-v1"
            or preprocess_worker_budget.get("backend")
            != preprocess_receipt["backend"]
            or preprocess_worker_budget.get("applicable")
            is not (preprocess_receipt["backend"] == "cpu")
            or preprocess_worker_budget.get(
                "pipeline_decoder_workers_included") is not False):
        raise RuntimeError("HRRR preprocessing worker-budget receipt is malformed")
    if preprocess_receipt["backend"] == "cpu":
        effective_total = preprocess_worker_budget.get(
            "effective_total_native_workers")
        peak_native = preprocess_worker_budget.get(
            "peak_active_native_workers")
        jobs = preprocess_worker_budget.get("effective_allocation_per_job")
        if (isinstance(effective_total, bool)
                or not isinstance(effective_total, int)
                or effective_total < 1
                or isinstance(peak_native, bool)
                or not isinstance(peak_native, int)
                or peak_native < 1
                or not _positive_count_within(cpu_workers, effective_total)
                or not isinstance(jobs, list)
                or len(jobs) != 1 + 2 * final_hour
                or any(
                    not isinstance(job, dict)
                    or isinstance(job.get("effective_native_workers"), bool)
                    or not isinstance(
                        job.get("effective_native_workers"), int)
                    or not 1 <= job["effective_native_workers"] <= effective_total
                    for job in jobs)
                or peak_native > effective_total):
            raise RuntimeError(
                "HRRR CPU preprocessing worker budget was not honored")
        if (requested_preprocess_workers is not None
                and preprocess_worker_budget.get(
                    "requested_total_native_workers")
                != requested_preprocess_workers):
            raise RuntimeError(
                "HRRR CPU preprocessing budget differs from the request")
    else:
        # The CUDA backend runs the masked surface fields on the host, in
        # the Rust library, on --preprocess-workers threads (every CPU the
        # process may use when none was given): the receipt names that
        # request and the budget the count they ran on.
        masked_chain = preprocess_receipt.get("masked_surface_chain")
        masked_workers = (masked_chain.get("workers")
                          if isinstance(masked_chain, dict) else None)
        host_workers = preprocess_worker_budget.get(
            "host_step_native_workers")
        if (isinstance(host_workers, bool)
                or not isinstance(host_workers, int)
                or host_workers < 1
                or preprocess_worker_budget.get(
                    "peak_active_native_workers") != host_workers):
            raise RuntimeError(
                "HRRR CUDA preprocessing host-step worker budget is malformed")
        expected_masked_workers = (
            "auto" if requested_preprocess_workers is None
            else requested_preprocess_workers)
        if (masked_workers != expected_masked_workers
                or preprocess_worker_budget.get(
                    "requested_total_native_workers")
                != requested_preprocess_workers
                or (requested_preprocess_workers is not None
                    and host_workers != requested_preprocess_workers)):
            raise RuntimeError(
                "HRRR CUDA preparation host-step worker receipt differs "
                "from the request")
    selected_pipeline_workers = (
        pipeline_worker_receipt.get("selected")
        if isinstance(pipeline_worker_receipt, dict) else None)
    if reused_physical_base or reused_posted_source:
        operation = ("reused_posted_native_source" if reused_posted_source
                     else "reused_sealed_native_bridge")
        if (not isinstance(pipeline_worker_receipt, dict)
                or pipeline_worker_receipt.get("operation") != operation
                or pipeline_worker_receipt.get("selected") != 0
                or pipeline_worker_receipt.get("requested") != str(requested_pipeline_workers).strip().lower()):
            raise RuntimeError("physical input did not attest its native source reuse")
        return (preprocess_receipt, preprocess_worker_budget, pipeline_worker_receipt)
    if (not isinstance(pipeline_worker_receipt, dict)
            or pipeline_worker_receipt.get("requested")
            != str(requested_pipeline_workers).strip().lower()
            or isinstance(selected_pipeline_workers, bool)
            or selected_pipeline_workers not in range(
                1, MAX_PIPELINE_WORKERS + 1)):
        raise RuntimeError("HRRR decoder worker receipt is malformed")
    return (preprocess_receipt, preprocess_worker_budget,
            pipeline_worker_receipt)


#: The receipt schema this wrapper's own producer emits.  It is a single
#: constant rather than a set because ``main`` never validates a stored
#: artifact: it runs ``tools/hrrr_single_domain_benchmark.py --prepare-only``
#: itself (below) and reads back the ``report.json`` that invocation just
#: wrote, from the same source tree.  An older-schema receipt therefore
#: cannot reach this consumer, and accepting one would only widen the gate
#: for a case that does not exist.  A prepared cache restored from an older
#: tree does not change that: the benchmark recomputes this receipt from the
#: restored state on every run, so the schema is always the running code's.
HRRR_INITIALIZATION_SCHEMA = "gpuwm-hrrr-microphysics-initialization-v4"

#: Every hydrometeor species the native HRRR analysis decodes.  The producer
#: must account for each of them exactly once, as either retained (mapped to
#: a state field) or discarded under a source-bound WRF-real policy.
_HRRR_DECODED_SOURCE_SPECIES = ("QC", "QR", "QI", "QS", "QG")

_HRRR_FINGERPRINT_KEYS = frozenset({
    "shape", "dtype", "sha256", "nonzero_mask_sha256", "nonzero_count",
    "minimum", "maximum",
})


def _validated_retention_evidence(
        initialization: dict[str, object]) -> dict[str, object]:
    """Bind the producer's analyzed-hydrometeor correspondence evidence.

    Validated as fields, not as a version string: the species partition, the
    per-species decoded-source and initialized-state fingerprints, and the
    self-declared evidentiary strength of each retention claim.  Vacuity is
    reported, never refused -- a cloud-free analysis is a legitimate
    preparation, and it is the receipt's job to say that its retention claim
    for that species was not earned.
    """

    correspondence = initialization.get("source_to_state_correspondence")
    discarded = initialization.get("discarded_source_species")
    masses = initialization.get("state_mass_fields")
    evidence = initialization.get("retention_evidence")
    summary = initialization.get("retention_evidence_summary")
    disposition = initialization.get("vertical_disposition")
    if (initialization.get("source_mass_fields")
            != list(_HRRR_DECODED_SOURCE_SPECIES)
            or not isinstance(correspondence, dict)
            or not isinstance(discarded, dict)
            or not isinstance(masses, dict)
            or not isinstance(evidence, dict)
            or not isinstance(summary, dict)
            or not isinstance(disposition, dict)
            or set(correspondence) | set(discarded)
            != set(_HRRR_DECODED_SOURCE_SPECIES)
            or set(correspondence) & set(discarded)
            or set(evidence) != set(correspondence)
            or not correspondence):
        raise RuntimeError(
            "HRRR preparation omitted analyzed-hydrometeor correspondence "
            "evidence")
    from woof.ingest.real import (
        validate_hrrr_hydrometeor_vertical_disposition,
    )

    vacuous = []
    proven = []
    partially_excluded = []
    excluded = []
    source_fingerprints = {}
    initialized_fingerprints = {}
    for source_name, state_name in sorted(correspondence.items()):
        mass = masses.get(state_name)
        species = evidence[source_name]
        if (not isinstance(state_name, str)
                or not isinstance(mass, dict)
                or not _HRRR_FINGERPRINT_KEYS <= set(mass)
                or mass.get("decoded_source_field") != source_name
                or not isinstance(mass.get("decoded_source"), dict)
                or not _HRRR_FINGERPRINT_KEYS <= set(mass["decoded_source"])
                or not isinstance(species, dict)
                or species.get("state_field") != state_name
                or species.get("strength") not in (
                    "PROVEN", "PARTIALLY_WRF_EXCLUDED", "WRF_EXCLUDED",
                    "VACUOUS")
                or int(species.get("source_nonzero_count", -1)) < 0
                or int(species.get("target_influencing_source_count", -1)) < 0
                or int(species.get("wrf_excluded_source_count", -1)) < 0
                or int(species.get("state_nonzero_count", -1)) < 0
                or not isinstance(
                    species.get("vertical_disposition_schema"), str)
                or not isinstance(
                    species.get("vertical_disposition_labels_sha256"), str)):
            raise RuntimeError(
                "HRRR preparation omitted analyzed-hydrometeor "
                "correspondence evidence")
        source_nonzero = int(species["source_nonzero_count"])
        if (source_nonzero != int(mass["decoded_source"]["nonzero_count"])
                or int(species["state_nonzero_count"])
                != int(mass["nonzero_count"])):
            raise RuntimeError(
                "HRRR preparation analyzed-hydrometeor retention evidence "
                f"for {source_name} disagrees with its own fingerprints")
        influencing = int(species["target_influencing_source_count"])
        wrf_excluded = int(species["wrf_excluded_source_count"])
        if influencing + wrf_excluded != source_nonzero:
            raise RuntimeError(
                "HRRR preparation analyzed-hydrometeor retention strength "
                f"for {source_name} contradicts its source partition")
        if influencing > 0 and int(mass["nonzero_count"]) == 0:
            raise RuntimeError(
                "HRRR preparation lost every nonzero analyzed "
                f"{source_name} on the way into state.{state_name}")
        if influencing == 0 and int(mass["nonzero_count"]) != 0:
            raise RuntimeError(
                "HRRR preparation created analyzed hydrometeor mass without "
                f"target support for {source_name}->state.{state_name}")
        source_fingerprints[source_name] = mass["decoded_source"]
        initialized_fingerprints[source_name] = {
            key: mass[key] for key in _HRRR_FINGERPRINT_KEYS}
        if species["strength"] == "VACUOUS":
            vacuous.append(source_name)
        elif species["strength"] == "PROVEN":
            proven.append(source_name)
        elif species["strength"] == "PARTIALLY_WRF_EXCLUDED":
            partially_excluded.append(source_name)
        else:
            excluded.append(source_name)

    try:
        validated_disposition = \
            validate_hrrr_hydrometeor_vertical_disposition(
                source_fingerprints, initialized_fingerprints, disposition)
    except ValueError as exc:
        raise RuntimeError(
            "HRRR preparation hydrometeor vertical disposition evidence "
            f"is invalid: {exc}") from exc
    for source_name, species in evidence.items():
        validated = validated_disposition["species"][source_name]
        if (species["strength"] != validated["strength"]
                or species["target_influencing_source_count"]
                != validated["target_influencing_source_count"]
                or species["wrf_excluded_source_count"]
                != validated["wrf_excluded_source_count"]
                or species["vertical_disposition_schema"]
                != validated_disposition["schema"]
                or species["vertical_disposition_labels_sha256"]
                != validated["labels_sha256"]):
            raise RuntimeError(
                "HRRR preparation analyzed-hydrometeor retention evidence "
                f"for {source_name} disagrees with its WRF support partition")
    if partially_excluded or (excluded and proven):
        expected_strength = "PARTIALLY_WRF_EXCLUDED"
    elif excluded:
        expected_strength = "WRF_EXCLUDED"
    elif not proven:
        expected_strength = "VACUOUS"
    elif vacuous:
        expected_strength = "PARTIALLY_VACUOUS"
    else:
        expected_strength = "PROVEN"
    if (summary.get("vacuous_species") != vacuous
            or summary.get("proven_species") != proven
            or summary.get("partially_excluded_species")
            != partially_excluded
            or summary.get("excluded_species") != excluded
            or summary.get("strength") != expected_strength):
        raise RuntimeError(
            "HRRR preparation analyzed-hydrometeor retention summary "
            "disagrees with its per-species evidence")
    return summary


def _validated_physics_receipt(
        preparation_report: dict[str, object], *,
        requested_profile: str | None, expected_selection=None) -> dict[str, object]:
    """Retain one exact runner profile and its cold-start evidence."""

    physics = preparation_report.get("physics")
    # Both sides through the alias table: the benchmark records the current
    # profile ID, while a request or a chain document written before the
    # 2.8.4 rename may still carry the old one for the same physics.
    if (not isinstance(physics, dict)
            or physics.get("schema") != "gpuwm-prepared-physics-profile-v1"
            or canonical_template_id(physics.get("profile"))
            != canonical_template_id(requested_profile)):
        raise RuntimeError(
            "HRRR preparation physics receipt differs from the request")
    if expected_selection is not None:
        from woof.hrrr_configuration import resolved_run_settings
        expected = resolved_run_settings(expected_selection)
        observed = physics.get("resolved")
        # Sealed suffix preparation runs only its one-hour subwindow. The
        # independently checked forcing window owns this one changing value.
        if not isinstance(observed, dict) or any(
                observed.get(name) != value for name, value in expected.items()
                if name != "run_seconds"):
            raise RuntimeError("HRRR preparation resolved physics differs from the request")
    initialization = physics.get("hrrr_initialization")
    selection = expected_selection
    if selection is None:
        if requested_profile is None:
            raise RuntimeError("configured preparation requires independent physics authority")
        selection = single_domain_runtime_switches(requested_profile)
    expected_wrf_fields, expected_state_fields = cold_start_contract(selection)
    if (not isinstance(initialization, dict)
            or initialization.get("schema") != HRRR_INITIALIZATION_SCHEMA
            or initialization.get("source_absent_wrf_fields")
            != list(expected_wrf_fields)
            or not isinstance(
                initialization.get("state_source_absent_fields"), dict)
            or set(initialization["state_source_absent_fields"])
            != set(expected_state_fields)):
        raise RuntimeError(
            "HRRR preparation omitted deterministic cold-start evidence")
    _validated_retention_evidence(initialization)
    # The Thompson numbers carry real.exe's seed rule, not exact
    # everywhere (A99, A110): the producer proves the allocation value
    # where the paired mass is zero and a finite seeded number above zero,
    # counted against the closure's seed receipt, where it is not.
    seeded_numbers = cold_start_seeded_numbers(selection)
    for name, (expected_float32, expected_uint32_bits) \
            in expected_state_fields.items():
        observed = initialization["state_source_absent_fields"][name]
        if name in seeded_numbers:
            mass_name, seed_key = seeded_numbers[name]
            seeded_cells = (observed.get("seeded_cells")
                            if isinstance(observed, dict) else None)
            if (not isinstance(observed, dict)
                    or set(observed) != {
                        "expected_float32", "expected_uint32_bits",
                        "all_exact_expected_where_mass_is_zero",
                        "paired_mass_field", "seed_receipt",
                        "seeded_cells", "seeded_all_finite_above_zero"}
                    or isinstance(observed.get("expected_float32"), bool)
                    or not isinstance(
                        observed.get("expected_float32"), (int, float))
                    or float(observed["expected_float32"])
                    != expected_float32
                    or isinstance(observed.get("expected_uint32_bits"), bool)
                    or observed.get("expected_uint32_bits")
                    != expected_uint32_bits
                    or observed.get(
                        "all_exact_expected_where_mass_is_zero") is not True
                    or observed.get("seeded_all_finite_above_zero")
                    is not True
                    or observed.get("paired_mass_field") != mass_name
                    or observed.get("seed_receipt") != seed_key
                    or isinstance(seeded_cells, bool)
                    or not isinstance(seeded_cells, int)
                    or seeded_cells < 0):
                raise RuntimeError(
                    "HRRR preparation omitted deterministic cold-start "
                    "evidence")
            continue
        if (not isinstance(observed, dict)
                or set(observed) != {
                    "expected_float32", "expected_uint32_bits",
                    "all_exact_expected"}
                or isinstance(observed.get("expected_float32"), bool)
                or not isinstance(
                    observed.get("expected_float32"), (int, float))
                or float(observed["expected_float32"]) != expected_float32
                or isinstance(observed.get("expected_uint32_bits"), bool)
                or observed.get("expected_uint32_bits")
                != expected_uint32_bits
                or observed.get("all_exact_expected") is not True):
            raise RuntimeError(
                "HRRR preparation omitted deterministic cold-start evidence")
    return physics


#: _sealed_extension's marker for "no resolved block was handed in".
_UNRESOLVED = object()


def _resolved_static_highres(experiment_tables, companion_source, run_config):
    """The ``[static.highres]`` block this preparation binds.

    Read from the root configuration :func:`resolve_root_experiment`
    resolved, with the same call the benchmark makes on the same tables,
    so the static build, a sealed extension and the benchmark's
    ``require_prepared_highres`` all see one block.  A namelist-only
    preparation has no configuration file to hand the static builder, and
    a 1 km root that took the grid-spacing default used to be built on the
    baseline and then refused by the benchmark.  ``run_config`` is the
    root's run, whose urban selectors pick the land-cover legend, as
    :func:`woof.static.highres_production.load_static_highres` picks it
    for the builder.
    """
    from woof.static.highres_production import resolve_static_highres
    return resolve_static_highres(
        experiment_tables, source=str(companion_source),
        base_dir=Path(companion_source).parent, run_config=run_config)


def _receipt_attests_smoothing(receipt_path: Path) -> bool:
    """True when a prebuilt native static's receipt records terrain smoothing.

    tools/hrrr_build_native_static.py writes ``terrain_smoothing`` only for
    a static built under a non-default d01 smoothing.  A receipt that does
    not read as JSON is answered False here: the builder and the benchmark
    verify it against its static and refuse it there.
    """
    try:
        receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return False
    return isinstance(receipt, dict) and "terrain_smoothing" in receipt


def _sealed_extension(args, *, valid_time: datetime,
                      source_forecast_hours, output: Path,
                      env: dict[str, str], decoder: Path,
                      started: float,
                      configured=None,
                      namelist_invariant: dict[str, str],
                      static_highres=_UNRESOLVED) -> int:
    """Prepare only the shared terminal/new-hour slab and append it.

    ``static_highres`` is the block :func:`_resolved_static_highres`
    returns for the root configuration, so a namelist-only extension
    compares the grid-spacing default its preparation binds.  Left out, it
    is read from ``--experiment-config`` alone.
    """
    from woof.ingest.prepared_cache import (
        PreparedCacheReader, extend_prepared_cache,
    )

    prior = args.extend_root_preparation.resolve()
    required_prior = {
        "wrapper": prior / "public-wrapper-result.json",
        "cache": prior / "native" / "prepared-cache",
        "report": prior / "native" / "preparation-report" / "report.json",
        "bridge": prior / "native" / "native-bridge",
        "source_snapshot": prior / "native" / "source-manifest.snapshot",
        "static": prior / "native-static.npz",
        "static_receipt": prior / "native-static-receipt.json",
        "geometry": prior / "native-geometry-receipt.json",
    }
    for path in required_prior.values():
        if not path.exists():
            raise FileNotFoundError(path)
    prior_wrapper = json.loads(
        required_prior["wrapper"].read_text(encoding="utf-8"))
    old_hours = prior_wrapper.get("source_forecast_hours")
    new_hours = list(source_forecast_hours)
    if (prior_wrapper.get("status") != "PASS"
            or prior_wrapper.get("prepared_cache_contract", {}).get("mode")
            != "sealed-prefix-v1"
            or not isinstance(old_hours, list) or len(old_hours) < 2
            or old_hours != list(range(len(old_hours)))
            or new_hours != old_hours + [len(old_hours)]
            or args.forecast_start_hour != 0
            or args.run_seconds != new_hours[-1] * 3600):
        raise ValueError(
            "sealed root extension must append exactly the next hour to a "
            "zero-based sealed predecessor")
    # A predecessor sealed under a profile's old ID (before the 2.8.4
    # rename) ran the same physics as its current ID and still extends.
    if (prior_wrapper.get("source_cycle") != valid_time.isoformat()
            or canonical_template_id(
                prior_wrapper.get("physics", {}).get("profile"))
            != canonical_template_id(args.physics_profile)
            or float(prior_wrapper.get("history_interval_seconds", -1.0))
            != float(args.history_interval_seconds)):
        raise ValueError("sealed predecessor belongs to another run contract")
    source_manifest = args.source_manifest.resolve()
    observed_source_sha = _sha256(source_manifest)
    if observed_source_sha != args.source_manifest_sha256.lower():
        raise ValueError("source manifest digest differs from the request")
    from woof.ingest.native_supplements import supplement_bindings
    source_proof = _source_manifest_extension(
        predecessor=required_prior["source_snapshot"],
        extended=source_manifest, source_root=args.source_root.resolve(),
        old_hours=old_hours, new_hours=new_hours, cycle=valid_time,
        supplemental_paths=[path for _, path in supplement_bindings(getattr(args, "supplement", ()))])
    prior_header = json.loads(
        (required_prior["cache"] / "header.json").read_text(
            encoding="utf-8"))
    hydrometeor_initialization = prior_header.get("metadata", {}).get(
        "hydrometeor_initialization")
    vertical_disposition = (
        hydrometeor_initialization.get("vertical_disposition")
        if isinstance(hydrometeor_initialization, dict) else None)
    if (not isinstance(hydrometeor_initialization, dict)
            or hydrometeor_initialization.get("schema")
            != "gpuwm-real-hydrometeor-correspondence-v2"
            or not isinstance(vertical_disposition, dict)
            or vertical_disposition.get("schema")
            != "gpuwm-wrf-real-hydrometeor-vertical-disposition-v1"):
        raise ValueError(
            "sealed predecessor lacks exhaustive WRF hydrometeor "
            "disposition evidence; rebuild it with the current GPUWM "
            "before extending")
    prior_identity = prior_header.get("identity")
    if (not isinstance(prior_identity, dict)
            or prior_header.get("metadata", {}).get(
                "forcing_extension_mode") != "sealed-prefix-v1"):
        raise ValueError("predecessor prepared cache is not prefix sealed")
    PreparedCacheReader(
        required_prior["cache"], expected_identity=prior_identity).verify_all()
    from woof.static.highres_production import (
        load_static_highres, prepared_highres_settings_match)
    requested_highres = (
        load_static_highres(getattr(args, "experiment_config", None))
        if static_highres is _UNRESOLVED else static_highres)
    previous_highres = prior_identity.get("source_identity", {}).get("static_highres")
    # A sealed extension appends forcing and keeps every existing static
    # byte. It cannot change an active overlay under a new declaration,
    # nor start one its predecessor was sealed without.
    previous_active = isinstance(previous_highres, dict) and previous_highres.get("enabled")
    if (previous_active or (requested_highres is not None and requested_highres.enabled)) \
            and not prepared_highres_settings_match(previous_highres, requested_highres):
        raise ValueError("sealed extension changes high-resolution statics while retaining "
                         "the predecessor's static bytes; rebuild preparation for the "
                         "changed geography settings")
    prior_namelist_invariant = prior_identity.get(
        "namelist_extension_invariant")
    if prior_namelist_invariant is None:
        raise ValueError(
            "sealed predecessor lacks the namelist extension invariant; "
            "rebuild it with the current GPUWM")
    if namelist_invariant != prior_namelist_invariant:
        raise ValueError(
            "native namelist changes immutable fields from the sealed "
            "predecessor")
    if _sha256(required_prior["static"]) != prior_identity.get(
            "static_cache_sha256"):
        raise ValueError("static cache differs from sealed predecessor")
    if args.domain_spec is None:
        raise ValueError("sealed HRRR extension requires --domain-spec")

    output.mkdir(parents=True, exist_ok=False)
    for role in ("static", "static_receipt", "geometry"):
        _link_file_create(required_prior[role], output / required_prior[role].name)
    native = output / "native"
    native.mkdir()
    snapshot = native / "source-manifest.snapshot"
    with snapshot.open("xb") as stream:
        stream.write(source_manifest.read_bytes())
    if _sha256(snapshot) != observed_source_sha:
        raise ValueError("source manifest changed while it was snapshotted")
    extension_work = native / "extension-work"
    extension_work.mkdir()
    suffix_start = old_hours[-1]
    lead = new_hours[-1]
    series = extension_work / f"hrrr-f{suffix_start:02d}-f{lead:02d}-series.tsv"
    rows = []
    for hour in (suffix_start, lead):
        atmosphere = args.source_root.resolve() / (
            f"hrrr.t{valid_time:%H}z.wrfnatf{hour:02d}.grib2")
        soil = args.source_root.resolve() / (
            f"hrrr.t{valid_time:%H}z.soilf{hour:02d}.grib2")
        for path in (atmosphere, soil):
            if not path.is_file():
                raise FileNotFoundError(path)
        from woof.ingest.native_supplements import series_supplement_suffix
        suffix = series_supplement_suffix(getattr(args, "supplement", ()))
        rows.append(f"{hour}\t{atmosphere}\t{soil}{suffix}\n")
    series.write_text("".join(rows), encoding="utf-8", newline="\n")
    suffix_bridge = extension_work / "native-bridge"
    suffix_cache = extension_work / "prepared-cache"
    suffix_report_root = extension_work / "preparation-report"
    benchmark = [
        sys.executable, str(REPO / "tools" / "hrrr_single_domain_benchmark.py"),
        "--bridge", str(suffix_bridge),
        "--valid-time", args.valid_time,
        "--forecast-start-hour", str(suffix_start),
        "--forecast-end-hour", str(lead),
        "--pipeline-series", str(series),
        "--pipeline-decoder", str(decoder),
        "--pipeline-signals", str(extension_work / "pipeline-signals"),
        "--pipeline-workers", str(args.pipeline_workers),
        "--source-root", str(args.source_root.resolve()),
        "--source-manifest", str(source_manifest),
        "--source-manifest-sha256", observed_source_sha,
        "--static-cache", str(required_prior["static"]),
        "--static-receipt", str(required_prior["static_receipt"]),
        "--namelist-input", str(args.namelist_input.resolve()),
        "--prepared-cache", str(suffix_cache),
        "--prepare-only", "--run-seconds", "3600",
        "--history-interval-seconds", str(args.history_interval_seconds),
        "--namelist-extension-suffix",
        "--outdir", str(suffix_report_root),
        "--domain-spec", str(args.domain_spec.resolve()),
        "--preprocess-backend", args.preprocess_backend,
    ]
    _configuration_arguments(benchmark, args)
    for acknowledgement in args.ack:
        benchmark.extend(("--ack", acknowledgement))
    if args.prepare_workers is not None:
        benchmark.extend(("--prepare-workers", str(args.prepare_workers)))
    if args.preprocess_workers is not None:
        benchmark.extend(("--preprocess-workers", str(args.preprocess_workers)))
    if args.cpu_preprocess_bridge is not None:
        benchmark.extend((
            "--cpu-preprocess-bridge",
            str(args.cpu_preprocess_bridge.resolve())))
    prepare_started = time.perf_counter()
    _run(benchmark, env)
    prepare_seconds = time.perf_counter() - prepare_started
    suffix_report_path = suffix_report_root / "report.json"
    suffix_report = json.loads(suffix_report_path.read_text(encoding="utf-8"))
    if (suffix_report.get("status") != "PASS"
            or suffix_report.get("source_forecast_hours")
            != [suffix_start, lead]
            or suffix_report.get("model_forcing_hours") != [0, 1]):
        raise RuntimeError("suffix preparation report has another time window")
    preprocess_receipt, preprocess_budget, pipeline_receipt = \
        _validated_worker_receipts(
            suffix_report, selected_backend=args.preprocess_backend,
            requested_preprocess_workers=args.preprocess_workers,
            requested_pipeline_workers=args.pipeline_workers, final_hour=1)
    physics_receipt = _validated_physics_receipt(
        suffix_report, requested_profile=args.physics_profile,
        expected_selection=configured.root.run)
    merged_bridge = native / "native-bridge"
    bridge_proof = _bridge_manifest_extension(
        predecessor=required_prior["bridge"], suffix=suffix_bridge,
        output=merged_bridge, old_hours=old_hours, new_hours=new_hours)
    suffix_header = json.loads(
        (suffix_cache / "header.json").read_text(encoding="utf-8"))
    suffix_identity = suffix_header.get("identity")
    if not isinstance(suffix_identity, dict):
        raise ValueError("suffix cache has no identity")
    new_identity = deepcopy(prior_identity)
    new_identity["bridge_manifest_sha256"] = bridge_proof[
        "extended_sha256"]
    new_identity["source_manifest_sha256"] = observed_source_sha
    new_identity["namelist_sha256"] = _sha256(
        args.namelist_input.resolve())
    new_identity["namelist_extension_invariant"] = namelist_invariant
    new_identity["forcing_hours"] = list(range(lead + 1))
    new_identity["domain_config"]["run"]["run_seconds"] = float(
        args.run_seconds)
    new_source_identity = deepcopy(prior_identity["source_identity"])
    new_source_identity["source_forecast_hours"] = new_hours
    new_source_identity["model_forcing_hours"] = list(range(lead + 1))
    new_identity["source_identity"] = new_source_identity
    prior_user = deepcopy(prior_header["metadata"]["user"])
    prior_user.update({
        "last_valid_time": (valid_time + timedelta(hours=lead)).isoformat(),
        "source_forecast_hours": new_hours,
        "model_forcing_hours": list(range(lead + 1)),
        "forcing_hours": list(range(lead + 1)),
        "source_manifest_extension": source_proof,
        "bridge_manifest_extension": bridge_proof,
    })
    cache_receipt = extend_prepared_cache(
        native / "prepared-cache", predecessor=required_prior["cache"],
        suffix=suffix_cache, identity=new_identity, metadata=prior_user,
        source_manifest_extension=source_proof,
        bridge_manifest_extension=bridge_proof)
    evidence = native / "extension-evidence"
    evidence.mkdir()
    _link_file_create(
        suffix_cache / "header.json", evidence / "suffix-cache-header.json")
    _link_file_create(
        suffix_report_path, evidence / "suffix-preparation-report.json")
    _write_json_create(evidence / "source-manifest-extension.json", source_proof)
    _write_json_create(evidence / "bridge-manifest-extension.json", bridge_proof)
    suffix_record = dict(cache_receipt["suffix"])
    suffix_record.pop("path", None)
    suffix_record.update({
        "retained": False,
        "header_snapshot": str(
            (evidence / "suffix-cache-header.json").resolve()),
    })
    cache_receipt["suffix"] = suffix_record
    _write_json_create(evidence / "prepared-cache-extension.json", cache_receipt)
    prior_report = json.loads(
        required_prior["report"].read_text(encoding="utf-8"))
    combined_report = deepcopy(prior_report)
    # A predecessor report is useful only as a base for fields whose
    # authority genuinely did not change.  Its input block names the old
    # bridge, old source digest, and shorter forcing window, so copying it
    # would publish a self-contradictory PASS report.  Reconstruct every
    # input identity from the newly published artifacts instead.
    combined_input = {
        "bridge": str(merged_bridge.resolve()),
        "bridge_manifest_sha256": bridge_proof["extended_sha256"],
        "source_manifest_sha256": observed_source_sha,
        "source_cycle": valid_time.isoformat(),
        "model_start_time": valid_time.isoformat(),
        "source_forecast_hours": new_hours,
        "model_forcing_hours": list(range(lead + 1)),
        "forcing_hours": list(range(lead + 1)),
        "native_static_cache": str((output / "native-static.npz").resolve()),
        "native_static_cache_sha256": _sha256(output / "native-static.npz"),
        "native_static_receipt": str(
            (output / "native-static-receipt.json").resolve()),
        "namelist_input_sha256": _sha256(args.namelist_input.resolve()),
        "native_physics_profile": args.physics_profile,
    }
    combined_report.update({
        "status": "PASS",
        "source_cycle": valid_time.isoformat(),
        "model_start_time": valid_time.isoformat(),
        "source_forecast_hours": new_hours,
        "model_forcing_hours": list(range(lead + 1)),
        "forcing_hours": list(range(lead + 1)),
        "source_identity": new_source_identity,
        "input": combined_input,
        "prepared_cache": cache_receipt,
        "extension": {
            "schema": "gpuwm-root-preparation-extension-v1",
            "predecessor": str(prior),
            "source_manifest": source_proof,
            "bridge_manifest": bridge_proof,
            "suffix_cache_content_sha256": suffix_header[
                "content_sha256"],
        },
        "physics": suffix_report["physics"],
        "pipeline": suffix_report.get("pipeline"),
        "preparation": suffix_report.get("preparation"),
    })
    preparation_report_path = native / "preparation-report" / "report.json"
    _write_json_create(preparation_report_path, combined_report)
    export_command = [
        sys.executable, "-m", "woof.wrf_direct",
        "--prepared-cache", str(native / "prepared-cache"),
        "--static-cache", str(output / "native-static.npz"),
        "--geometry-receipt", str(output / "native-geometry-receipt.json"),
        "--output", str(output / "wrf-native-input"),
        "--valid-time", valid_time.strftime("%Y-%m-%d_%H:%M:%S"),
    ]
    export_command.extend(("--physics-profile", args.physics_profile)
                          if args.physics_profile is not None else ("--experiment-config-suite",))
    for acknowledgement in args.ack:
        export_command.extend(("--ack", acknowledgement))
    export_started = time.perf_counter()
    stock_wrf_export = _stock_wrf_export(
        export_command, env, skip=args.skip_stock_wrf_export,
        required=args.require_stock_wrf_export,
        output=output / "wrf-native-input")
    export_seconds = time.perf_counter() - export_started
    # The two-frame cache was construction scratch.  Its header/report are
    # retained above; every old payload and the new decoded hour are already
    # hard-linked into the immutable products, so retaining the full scratch
    # state would waste one complete root state per leg.
    shutil.rmtree(extension_work)
    result = {
        "status": "PASS",
        "stock_wrf_export": stock_wrf_export,
        "wrf_input": (str(output / "wrf-native-input")
                      if stock_wrf_export["status"] == "PASS" else None),
        "source_cycle": valid_time.isoformat(),
        "model_start_time": valid_time.isoformat(),
        "source_forecast_hours": new_hours,
        "model_forcing_hours": list(range(lead + 1)),
        "forcing_hours": list(range(lead + 1)),
        "history_interval_seconds": float(args.history_interval_seconds),
        "static_seconds": 0.0,
        "prepare_seconds": prepare_seconds,
        "export_seconds": export_seconds,
        "total_seconds": time.perf_counter() - started,
        "preprocess_backend": preprocess_receipt,
        "preprocess_worker_budget": preprocess_budget,
        "pipeline_decoder_workers": pipeline_receipt,
        "preparation_report": str(preparation_report_path),
        "physics": physics_receipt,
        "prepared_cache_contract": {
            "mode": "sealed-prefix-v1",
            "operation": "extend-one-hour",
            "predecessor": str(prior),
            "content_sha256": cache_receipt["content_sha256"],
        },
    }
    _write_json_create(output / "public-wrapper-result.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--source-manifest", type=Path,
        help="the fetch's SHA256SUMS (required unless --as-posted)")
    parser.add_argument(
        "--source-manifest-sha256",
        help="its sha256 (required unless --as-posted)")
    parser.add_argument(
        "--as-posted", type=Path, dest="as_posted",
        help=("the as-posted fetch's posting/ folder in --source-root "
              "(A136): each lead is decoded once its posted marker is "
              "there, the head is published on the first two leads, and "
              "the seal writes the source manifest from the leads' markers, "
              "so --source-manifest and its digest are not given"))
    parser.add_argument("--supplement", action="append", default=[],
                        help="PMSL=GRIB donor, bound by the source manifest; repeat for multiple files")
    static = parser.add_mutually_exclusive_group(required=True)
    static.add_argument("--geog-root", type=Path)
    static.add_argument("--static-cache", type=Path)
    parser.add_argument("--static-receipt", type=Path)
    parser.add_argument("--physical-input-store", type=Path,
                        help="sealed native physical input for the complete forcing window")
    parser.add_argument("--physical-input-provider", type=Path,
                        help="posted native physical provider with a frozen member plan")
    parser.add_argument("--physical-member-index", type=int,
                        help="original recipe member index in the posted provider")
    parser.add_argument("--physical-output-store", type=Path,
                        help="capture mapped physical snapshots before initialization")
    parser.add_argument("--physical-base-prepared", type=Path,
                        help="reuse the sealed native bridge from this physical input's base preparation")
    parser.add_argument("--domain-spec", type=Path)
    parser.add_argument("--experiment-config", type=Path)
    parser.add_argument("--namelist-input", type=Path, required=True)
    parser.add_argument(
        "--wps-namelist", type=Path,
        help=("YOUR &geogrid namelist.wps, bound into the portable "
              "bundle in place of the one this route renders from the "
              "domain it was given.  The portable bundle -- proof.json, "
              "a role-keyed source manifest, the experiment authority "
              "and namelist.wps -- is published on every run, so this "
              "flag changes WHICH namelist is bound, never WHETHER the "
              "bundle exists"))
    parser.add_argument(
        "--physics-profile",
        default=None,
        # An old profile ID reads as its current ID, as the benchmark this
        # wrapper drives reads it; otherwise the benchmark's report names
        # the current ID and the receipt check below refuses the run after
        # the whole preparation has finished.
        type=canonical_template_id,
        help="optional equality assertion against a named physics template",
    )
    parser.add_argument(
        "--ack", action="append", default=[],
        help="registry-owned expert physics acknowledgement id; repeatable")
    parser.add_argument(
        "--cycle",
        help="the HRRR CYCLE this window comes from, YYYY-MM-DD_HH:MM:SS "
             "(UTC).  Model time zero is cycle + --forecast-start-hour and "
             "is never typed")
    parser.add_argument(
        "--valid-time",
        help="deprecated spelling of --cycle (it always meant the cycle "
             "on this command); accepted unchanged for v1.4.0 scripts")
    parser.add_argument(
        "--forecast-start-hour", type=int, default=0,
        help="absolute HRRR cycle-relative source lead used for model time zero")
    parser.add_argument(
        "--forecast-end-hour", type=int,
        help="inclusive absolute source lead; must cover --run-seconds exactly")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--sealed-prepared-cache", action="store_true",
        help=("opt in to a prefix-sealed root preparation for operational "
              "one-hour extension"))
    parser.add_argument(
        "--extend-root-preparation", type=Path,
        help=("sealed predecessor root; decode only its terminal overlap and "
              "the one newly arrived hour"))
    parser.add_argument("--run-seconds", type=int)
    parser.add_argument(
        "--history-interval-seconds", type=_positive_finite_seconds,
        default=None,
        help=("future GPUWM history cadence bound into the prepared-cache "
              "identity; preparation itself writes no history frames"),
    )
    parser.add_argument(
        "--pipeline-workers", type=_pipeline_workers, default="8",
        help=(f"HRRR decode workers: 1..{MAX_PIPELINE_WORKERS} or auto "
              "(the decoder's own bound)"))
    parser.add_argument("--prepare-workers", type=int)
    parser.add_argument(
        "--preprocess-backend", choices=("cuda", "cpu", "auto"),
        default="auto")
    parser.add_argument("--preprocess-workers", type=int)
    parser.add_argument("--cpu-preprocess-bridge", type=Path)
    export = parser.add_mutually_exclusive_group()
    export.add_argument(
        "--skip-stock-wrf-export", action="store_true",
        help="do not write the stock-WRF wrfinput/wrfbdy export at all.  "
             "The GPU hierarchy consumes the prepared cache directly and "
             "never reads that export; skipping it is the right answer "
             "whenever stock wrf.exe is not the consumer")
    export.add_argument(
        "--require-stock-wrf-export", action="store_true",
        help="accepted for compatibility; a requested export that fails "
             "is ALWAYS this command's failure since the 2026-08-04 "
             "fail-closed ruling (a warn-and-continue posture PASS-logged "
             "a preparation with no WRF-arm outputs).  The explicit "
             "opt-out is --skip-stock-wrf-export")
    explain.add_explain_flag(parser)
    return parser


def _require_microphysics_tables(profile: str) -> None:
    """Refuse an unstaged lookup-table set before any paid stage runs.

    Presence only, in one sentence naming ``woof fetch-tables``.  The
    exact sizes and SHA-256 digests are re-checked by the root
    preparation's own profile binding
    (``tools/hrrr_single_domain_benchmark._microphysics_table_
    authority``) and again at load, so this is the early half of a gate
    that still fails closed later, not a second opinion.
    """

    from woof.core.thompson_contract import (
        CLASSIC_TABLE_ASSETS, MP_PHYSICS as THOMPSON_MP_PHYSICS)
    from woof.physics_compat import single_domain_runtime_switches
    from woof.table_assets import require_thompson_tables

    switches = (single_domain_runtime_switches(profile)
                if isinstance(profile, str) else vars(profile))
    mp_physics = int(switches["mp_physics"])
    if mp_physics == THOMPSON_MP_PHYSICS:
        require_thompson_tables(assets=CLASSIC_TABLE_ASSETS)
    elif mp_physics == 28:
        # mp=28 reads the classic set plus CCN_ACTIVATE.BIN from the same
        # root; staged here (first use) before the forcing download, not
        # refused at root preparation after it.
        from woof.core.thompson_aerosol_contract import AEROSOL_TABLE_ASSETS

        require_thompson_tables(
            assets=(*CLASSIC_TABLE_ASSETS, *AEROSOL_TABLE_ASSETS))


def main(argv: list[str] | None = None) -> int:
    """This tool's front door; the body is :func:`_prepare_from_argv`.

    Same reason as `woof.cli.main`: `--explain` lives in module state
    because library warnings have no `args` in reach, and this function
    is called in-process (the tests import this module and call it), so
    a stamp that outlives the call leaks an explanation layer into the
    next one.  `explain_scope` bounds it to this invocation.
    """

    with explain.explain_scope(False):
        return _prepare_from_argv(argv)


def _configured_defaults(args):
    if args.run_seconds is not None and args.history_interval_seconds is not None:
        return
    from woof.hrrr_configuration import native_configuration_defaults
    configured = native_configuration_defaults(
        experiment_config=args.experiment_config,
        namelist_input=getattr(args, "namelist_input", None),
        domain_spec=getattr(args, "domain_spec", None),
        wps_namelist=getattr(args, "wps_namelist", None),
        physics_profile=getattr(args, "physics_profile", None),
        acknowledgements=tuple(getattr(args, "ack", ())))
    if args.run_seconds is None:
        args.run_seconds = configured.run_seconds
    if args.history_interval_seconds is None:
        args.history_interval_seconds = (configured.root.full_history_interval_s
                                         or configured.root.history_interval_s)


def _validate_prepared_history_cadence(preparation_report, configured):
    # The benchmark reports the parsed writer alarm. With a quick look,
    # that alarm is shorter than the full-frame interval passed as an
    # override to the experiment document.
    observed = preparation_report.get("history_interval_seconds")
    if (isinstance(observed, bool) or not isinstance(observed, (int, float))
            or float(observed) != float(configured.root.history_interval_s)):
        raise RuntimeError(
            "HRRR prepared-cache history cadence differs from the request")


#: What this wrapper hands a chained native preparation
#: (``tools/hrrr_single_domain_benchmark.py --chained-bundle``).
CHAINED_BUNDLE_SCHEMA = "gpuwm-hrrr-chained-bundle-v1"


def _chains(args) -> bool:
    """Whether this preparation publishes a head a forecast can start on.

    On by default, as every chained route is (``WOOF_CHAINED_PREP=0``
    turns it off).  Not for a prefix-sealed cache, which the stream
    controller extends one sealed hour at a time, and not in an
    installation with no forecast to start on a head.
    """

    from woof.ingest.boundary_stream import chained_enabled, forecast_installed

    config=getattr(args,'experiment_config',None)
    if config is not None:
        from woof.experiment import load_experiment
        if load_experiment(config).nudging is not None:
            return False  # full-domain analysis series are published with the sealed bundle
    elif getattr(args,'namelist_input',None) is not None:
        from woof.namelist_import import parse_namelist,active_nudging_selectors
        if active_nudging_selectors(parse_namelist(args.namelist_input).get('fdda',{})):
            return False

    return (not args.sealed_prepared_cache
            and getattr(args, "physical_input_store", None) is None
            and (getattr(args, "physical_output_store", None) is None
                 or getattr(args, "as_posted", None) is not None)
            and chained_enabled()
            and forecast_installed())


def _verified_chained_handoff(handoff, output: Path) -> dict:
    """The handoff a chained preparation published, held to the files it names.

    The chained preparation writes ``proof.json`` itself, at its seal; a
    handoff whose digests are not the files beside it would send the
    forecast stage pins it refuses one stage later.
    """

    proof = output / "proof.json"
    header = output / "native" / "prepared-cache" / "header.json"
    manifest = output / "source-input-manifest.json"
    if (not isinstance(handoff, dict) or not proof.is_file()
            or handoff.get("proof_sha256") != _sha256(proof)
            or handoff.get("source_manifest_sha256") != _sha256(manifest)
            or handoff.get("prepared_content_sha256") != json.loads(
                header.read_text(encoding="utf-8")).get("content_sha256")):
        raise RuntimeError(
            "the chained HRRR preparation's portable handoff does not "
            f"describe the proof, manifest and cache in {output}")
    return dict(handoff)


#: Where an as-posted bundle's seal writes the source manifest it authors
#: (tools/hrrr_single_domain_benchmark.py POSTED_SOURCE_MANIFEST).
POSTED_SOURCE_MANIFEST = "native/posted-source/SHA256SUMS"


def _as_posted_refusal(args) -> str | None:
    """Why this preparation cannot read its leads as they post, or ``None``.

    Each names the breakage it prevents.
    """

    if (args.source_manifest is None) != (args.source_manifest_sha256 is None):
        return ("--source-manifest and --source-manifest-sha256 are given "
                "together")
    if args.as_posted is None:
        if args.source_manifest is None:
            return ("--source-manifest and --source-manifest-sha256 are "
                    "required unless --as-posted")
        return None
    if args.source_manifest is not None:
        return ("--as-posted writes the source manifest at its seal from the "
                "leads' markers; a --source-manifest given beside it would "
                "bind a second, different manifest")
    if args.supplement:
        return ("--as-posted reads the fetch's own leads; a PMSL donor is "
                "bound into the source manifest before anything is decoded, "
                "so a donor run fetches the whole window first")
    if args.sealed_prepared_cache or args.extend_root_preparation is not None:
        return ("--as-posted and a prefix-sealed cache are mutually "
                "exclusive: the sealed prefix binds its source manifest "
                "before it decodes")
    return None


def _prepare_from_argv(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if (args.physical_input_provider is None) != (args.physical_member_index is None):
        raise ValueError("physical-input-provider and physical-member-index must be supplied together")
    if args.physical_input_provider is not None:
        if (args.as_posted is None or args.physical_input_store is not None
                or args.physical_base_prepared is not None or args.physical_output_store is not None):
            raise ValueError("posted physical input requires --as-posted and no sealed input or base preparation")
    if args.physical_input_store is not None and args.as_posted is not None:
        raise ValueError("sealed physical input cannot replace an as-posted provider")
    if any(value is not None for value in (args.physical_input_store, args.physical_output_store, args.physical_input_provider)):
        if args.sealed_prepared_cache or args.extend_root_preparation is not None:
            raise ValueError("physical input preparation cannot extend a sealed forcing prefix")
    if (args.physical_base_prepared is not None
            and args.physical_input_store is None and args.physical_output_store is None):
        raise ValueError("physical-base-prepared requires physical-input-store or physical-output-store")
    refusal = _as_posted_refusal(args)
    if refusal is not None:
        raise ValueError(refusal)
    from woof.ingest.native_supplements import supplement_bindings
    for _, path in supplement_bindings(args.supplement):
        if not path.is_file():
            raise FileNotFoundError(f"declared PMSL donor is missing: {path}")
    _configured_defaults(args)
    with preprocessing_math_scope(args.preprocess_backend, cpu_bridge=args.cpu_preprocess_bridge,
                                  workers=args.preprocess_workers):
        return _run_configured(args)


def _run_configured(args):
    explain.set_explain(explain.explain_enabled(args))
    cycle, _legacy = resolve_cycle_flags(
        args.cycle, args.valid_time,
        tool="prepare_hrrr_wrf", legacy_means="the HRRR cycle",
        warn=explain.warn)
    # Kept under its old name inside this function: every receipt key and
    # every downstream flag below already spells this instant the cycle.
    valid_time = cycle
    source_forecast_hours = hrrr_source_window(
        cycle=valid_time, start_hour=args.forecast_start_hour,
        run_seconds=args.run_seconds, end_hour=args.forecast_end_hour)
    model_forcing_hours = tuple(range(len(source_forecast_hours)))
    final_hour = model_forcing_hours[-1]
    model_start_time = valid_time + timedelta(
        hours=source_forecast_hours[0])
    # The selected suite's microphysics lookup tables, resolved and
    # byte-checked BEFORE the decoder runs and before a single source
    # file is read.  This preparation used to resolve no tables at all --
    # its only table gate lived in the benchmark runner and fired for one
    # profile behind two environment variables -- so an mp8 suite reached
    # GPU setup unstaged, minutes of paid work later.  Naming it here also
    # removed the reason this route's default had to be a suite that
    # needs no tables, which is why the default could be flipped to a
    # full-radiation one at all.
    from woof.hrrr_configuration import resolve_root_experiment
    from woof.ingest.hrrr_target import load_hrrr_target_domain
    from woof.vertical_contract import explicit_vertical_from_wrf_namelist
    target = load_hrrr_target_domain(args.domain_spec)
    vertical = explicit_vertical_from_wrf_namelist(
        args.namelist_input, expected_nz=target.nz, context="native preparation")
    configured, experiment_tables = resolve_root_experiment(
        target=target, vertical=vertical, namelist_input=args.namelist_input,
        start_time=model_start_time, run_seconds=args.run_seconds,
        experiment_config=args.experiment_config, wps_namelist=args.wps_namelist,
        physics_profile=args.physics_profile, acknowledgements=tuple(args.ack),
        history_interval_seconds=args.history_interval_seconds)
    from woof.case_data import optional_case_data_from_tables
    from woof.ingest.native_supplements import (native_pressure_policy,
                                                 require_native_pressure_field,
                                                 supplement_bindings)
    companion_source = args.experiment_config or args.namelist_input
    declared_case = optional_case_data_from_tables(
        experiment_tables, source=str(companion_source), base_dir=Path(companion_source).parent)
    highres = _resolved_static_highres(
        experiment_tables, companion_source, configured.root.run)
    require_native_pressure_field(native_pressure_policy(args.namelist_input, declared_case),
                                  bindings=supplement_bindings(args.supplement))
    _require_microphysics_tables(configured.root.run)
    if args.prepare_workers is not None and args.prepare_workers not in range(1, 33):
        raise ValueError("prepare-workers must be between 1 and 32")
    if args.preprocess_workers is not None and args.preprocess_workers < 1:
        raise ValueError("preprocess-workers must be positive")
    if (args.preprocess_backend != "cpu"
            and args.cpu_preprocess_bridge is not None):
        raise ValueError(
            "cpu-preprocess-bridge requires preprocess-backend cpu")
    if args.extend_root_preparation is not None \
            and not args.sealed_prepared_cache:
        raise ValueError(
            "extend-root-preparation requires --sealed-prepared-cache")
    if args.sealed_prepared_cache and args.forecast_start_hour != 0:
        raise ValueError(
            "sealed prepared-cache orchestration requires forecast start 0")
    # --pipeline-workers is validated by its argparse type, at parse
    # time, against the decode pipeline's own ceiling.
    as_posted = args.as_posted is not None
    if not as_posted:
        if len(args.source_manifest_sha256) != 64:
            raise ValueError(
                "source-manifest-sha256 must contain 64 hexadecimal digits")
        int(args.source_manifest_sha256, 16)
    source_root = args.source_root.resolve()
    namelist_input = args.namelist_input.resolve()
    # As posted, the seal writes the source manifest inside the bundle.
    source_manifest = (args.output_root.resolve() / POSTED_SOURCE_MANIFEST
                       if as_posted else args.source_manifest.resolve())
    for path in ((source_root, namelist_input) if as_posted
                 else (source_root, source_manifest, namelist_input)):
        if not path.exists():
            raise FileNotFoundError(path)
    if as_posted and (args.as_posted.resolve().parent != source_root
                      or not args.as_posted.is_dir()):
        raise ValueError(
            f"--as-posted names {args.as_posted}, not the posting folder of "
            f"--source-root {source_root}, whose leads this preparation reads")
    namelist_invariant = (
        namelist_extension_invariant(
            namelist_input, cycle=valid_time, run_seconds=args.run_seconds)
        if args.sealed_prepared_cache else None)
    if args.geog_root is not None and args.domain_spec is None:
        raise ValueError("domain-spec is required with geog-root")
    if args.static_cache is not None and args.static_receipt is None:
        raise ValueError("static-receipt is required with static-cache")
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output root: {output}")
    env = _stage_environment()
    decoder = (None if args.physical_input_provider is not None or args.physical_base_prepared is not None
               else _decoder(env, **({"as_posted": True} if as_posted else {})))
    started = time.perf_counter()
    if args.sealed_prepared_cache:
        observed_manifest = _sha256(source_manifest)
        if observed_manifest != args.source_manifest_sha256.lower():
            raise ValueError("source manifest digest differs from the request")
    if args.extend_root_preparation is not None:
        if args.wps_namelist is not None:
            # A sealed extension joins one more hour onto a root that
            # already published its own authorities; binding a second,
            # different namelist beside them would leave two proofs
            # describing one bundle.  Refuse rather than silently drop
            # the flag.
            raise ValueError(
                "--wps-namelist binds a namelist into a NEW preparation's "
                "portable bundle; a sealed extension inherits the root's "
                "own authorities and must not republish them")
        return _sealed_extension(
            args, valid_time=valid_time,
            source_forecast_hours=source_forecast_hours,
            output=output, env=env, decoder=decoder, started=started,
            configured=configured, namelist_invariant=namelist_invariant,
            static_highres=highres)
    output.mkdir(parents=True)

    if args.experiment_config is not None:
        static_arguments = [
            "--experiment-config", str(args.experiment_config.resolve()),
            "--case-date", model_start_time.date().isoformat()]
    elif highres is not None:
        # No configuration file to name: hand the builder the carrier the
        # root configuration resolved, the one the benchmark checks the
        # static receipt against, as the identity a seal records.  The
        # [static] table alone carried neither d01's terrain smoothing nor
        # the urban legend, and a carrier holding only smoothing has no
        # [static] table at all.
        from woof.static.highres_production import static_highres_identity
        static_arguments = [
            "--static-highres",
            json.dumps(static_highres_identity(highres), sort_keys=True),
            "--case-date", model_start_time.date().isoformat()]
    else:
        static_arguments = []
    if (configured.root.run.sf_lake_physics == 1
            and configured.root.run.use_lakedepth == 1):
        static_arguments.append("--lake-depth")
    # The terrain-drag statistics the configured physics reads (gwd_opt,
    # topo_wind).  Without them the forecast seam refused the prepared root
    # after the whole preparation (prepared_terrain_drag_missing; QA R4).
    from woof.static.orographic import required_static_fields
    drag_fields = required_static_fields(
        configured.root.run.topo_wind or 0, configured.root.run.gwd_opt or 0)
    if drag_fields:
        static_arguments += ["--orographic", ",".join(drag_fields)]
    static_domain_arguments = ([] if args.domain_spec is None else [
        "--domain-spec", str(args.domain_spec.resolve())])
    geometry_receipt = output / "native-geometry-receipt.json"
    # Each step is said as a step record (woof.progress.prep_stage) on
    # stderr, where `woof go` reads this program's output and puts it on
    # the run's stream.  Without them a single-domain HRRR run page showed
    # only the stage for the whole preparation.
    with prep_stage("root_static", label="Prepare root static fields"):
        if args.physical_input_provider is not None:
            static_cache, static_receipt, geometry_receipt = _reuse_posted_static(args, output)
        elif args.geog_root is not None:
            static_cache = output / "native-static.npz"
            static_receipt = output / "native-static-receipt.json"
            _run([
                sys.executable, str(REPO / "tools" / "hrrr_build_native_static.py"),
                "--geog-root", str(args.geog_root.resolve()),
                *static_domain_arguments,
                "--output", str(static_cache), "--receipt", str(static_receipt),
                *static_arguments,
            ], env)
        else:
            static_cache = args.static_cache.resolve()
            static_receipt = args.static_receipt.resolve()
            for path in (static_cache, static_receipt):
                if not path.is_file():
                    raise FileNotFoundError(path)
            # A prefix-sealed root is a self-contained predecessor authority.
            # Extension cannot depend on the operator retaining two external
            # paths which were not part of that root, and copying them once per
            # hourly generation would defeat the hard-link-only storage contract.
            # Publish the exact verified files under the same canonical names the
            # geog-built route uses.  The same-filesystem refusal is intentional:
            # every future generation links these bytes again.
            #
            # A PORTABLE bundle needs the same thing for a different reason:
            # the forecast front door resolves the static cache relative to
            # --prepared-root, and a root whose static geography lives on
            # some other path the operator happened to pass is not a bundle
            # anyone else can bind.  Same link, same canonical names, same
            # bytes -- and a reused external static stays reused.
            #
            # Unconditional, because the portable bundle is published on
            # every run now.  It used to be gated on the two flags that
            # implied a portable product, which left the default --static-cache
            # run publishing nothing: the bundle writer refuses a static cache
            # outside the root ("is outside the bundle root"), so the one
            # decision that made the authority publishable was taken only when
            # someone had already asked for it by name.
            sealed_static = output / "native-static.npz"
            sealed_receipt = output / "native-static-receipt.json"
            # A carrier smoothing d01 goes through the builder too, whose
            # root seam refuses a prebuilt static that does not attest that
            # smoothing; linked as it is, the cache's terrain would run
            # under the setting's name.  So does a prebuilt static whose
            # receipt attests a d01 smoothing: the builder refuses one built
            # under another smoothing than this preparation asks for, and
            # linked, a default run integrated smoothed terrain the root
            # seam passes (a default request is not checked there).
            from woof.static.terrain_smoothing import smoothing_for
            if ((highres is not None and (
                    highres.enabled
                    or not smoothing_for(highres, 1).is_default))
                    or _receipt_attests_smoothing(static_receipt)):
                _run([
                    sys.executable, str(REPO / "tools" / "hrrr_build_native_static.py"),
                    "--static-cache", str(static_cache), "--static-receipt", str(static_receipt),
                    *static_domain_arguments,
                    "--output", str(sealed_static), "--receipt", str(sealed_receipt),
                    *static_arguments,
                ], env)
            else:
                if drag_fields:
                    stored = json.loads(static_receipt.read_text(
                        encoding="utf-8")).get("array_sha256", {})
                    missing_drag = [name for name in drag_fields
                                    if name not in stored]
                    if missing_drag:
                        raise ValueError(
                            "prepared_terrain_drag_missing: the static cache "
                            f"{static_cache} has no {', '.join(missing_drag)} "
                            "required by the configured terrain drag; build "
                            "the static from --geog-root instead")
                _link_file_create(static_cache, sealed_static)
                _link_file_create(static_receipt, sealed_receipt)
            static_cache = sealed_static
            static_receipt = sealed_receipt
        geometry_command = [
            sys.executable,
            str(REPO / "tools" / "write_hrrr_native_geometry_receipt.py"),
            "--static-cache", str(static_cache),
            "--hrrr-static-receipt", str(static_receipt),
            "--output", str(geometry_receipt),
        ]
        if args.domain_spec is not None:
            geometry_command.extend(("--domain-spec", str(args.domain_spec.resolve())))
        if args.physical_input_provider is None:
            _run(geometry_command, env)
    static_seconds = time.perf_counter() - started

    native = output / "native"
    native.mkdir()
    series = native / (
        f"hrrr-f{source_forecast_hours[0]:02d}-"
        f"f{source_forecast_hours[-1]:02d}-series.tsv")
    rows = []
    for hour in source_forecast_hours:
        atmosphere = source_root / (
            f"hrrr.t{valid_time:%H}z.wrfnatf{hour:02d}.grib2")
        soil = source_root / f"hrrr.t{valid_time:%H}z.soilf{hour:02d}.grib2"
        for path in (atmosphere, soil):
            # As posted, a later lead's files arrive with its marker.
            if not as_posted and not path.is_file():
                raise FileNotFoundError(path)
        from woof.ingest.native_supplements import series_supplement_suffix
        suffix = series_supplement_suffix(getattr(args, "supplement", ()))
        rows.append(f"{hour}\t{atmosphere}\t{soil}{suffix}\n")
    series.write_text("".join(rows), encoding="utf-8")
    _namelist_start_advisory(
        namelist_input, model_start_time, cycle=valid_time,
        lead=source_forecast_hours[0])
    benchmark = [
        sys.executable, str(REPO / "tools" / "hrrr_single_domain_benchmark.py"),
        "--bridge", str(native / "native-bridge"),
        "--cycle", valid_time.strftime(HRRR_TIME_FORMAT),
        "--forecast-start-hour", str(source_forecast_hours[0]),
        "--forecast-end-hour", str(source_forecast_hours[-1]),
        "--pipeline-series", str(series),
        *(("--pipeline-decoder", str(decoder)) if decoder is not None else ()),
        "--pipeline-signals", str(native / "pipeline-signals"),
        "--pipeline-workers", str(args.pipeline_workers),
        "--source-root", str(source_root),
        "--source-manifest", str(source_manifest),
        *(("--as-posted", str(args.as_posted.resolve())) if as_posted else
          ("--source-manifest-sha256", args.source_manifest_sha256.lower())),
        "--static-cache", str(static_cache),
        "--static-receipt", str(static_receipt),
        "--namelist-input", str(namelist_input),
        "--prepared-cache", str(native / "prepared-cache"),
        "--prepare-only", "--run-seconds", str(args.run_seconds),
        "--history-interval-seconds", str(args.history_interval_seconds),
        "--outdir", str(native / "preparation-report"),
    ]
    _configuration_arguments(benchmark, args)
    # Unconditional.  Only the benchmark holds the experiment tables this
    # route builds in code, so only it can render the authority a
    # config-driven stage will bind -- and every prepared tree needs one,
    # because every prepared tree is something `woof sim` should be able
    # to run.  It used to be published only when --wps-namelist was
    # supplied, which is how a finished tree could exist that no forecast
    # front door would accept.
    benchmark.extend((
        "--publish-experiment-config",
        str(output / PORTABLE_EXPERIMENT_CONFIG_NAME)))
    if args.sealed_prepared_cache:
        benchmark.append("--sealed-prepared-cache")
    for acknowledgement in args.ack:
        benchmark.extend(("--ack", acknowledgement))
    if args.prepare_workers is not None:
        benchmark.extend(("--prepare-workers", str(args.prepare_workers)))
    benchmark.extend(("--preprocess-backend", args.preprocess_backend))
    if args.preprocess_workers is not None:
        benchmark.extend(("--preprocess-workers", str(args.preprocess_workers)))
    if args.cpu_preprocess_bridge is not None:
        benchmark.extend((
            "--cpu-preprocess-bridge",
            str(args.cpu_preprocess_bridge.resolve())))
    for flag, path in (("--physical-input-store", args.physical_input_store),
                       ("--physical-input-provider", args.physical_input_provider),
                       ("--physical-output-store", args.physical_output_store)):
        if path is not None:
            benchmark.extend((flag, str(path.resolve())))
    if args.physical_member_index is not None:
        benchmark.extend(("--physical-member-index", str(args.physical_member_index)))
    if args.domain_spec is not None:
        benchmark.extend(("--domain-spec", str(args.domain_spec.resolve())))
    if args.physical_base_prepared is not None:
        bridge_sha = _reuse_physical_base_bridge(
            args.physical_base_prepared, native / "native-bridge",
            source_manifest_sha256=args.source_manifest_sha256,
            static_cache=static_cache)
        for flag in ("--pipeline-series", "--pipeline-decoder", "--pipeline-signals"):
            if flag in benchmark:
                index = benchmark.index(flag)
                del benchmark[index:index + 2]
        benchmark.extend(("--manifest-sha256", bridge_sha))
    chained = _chains(args)
    if chained:
        # The authorities the portable proof binds, so the preparation
        # publishes the bundle's head the moment its start state exists
        # and its forecast can start on it.
        chain_document = native / "chained-bundle.json"
        _write_json_create(chain_document, {
            "schema": CHAINED_BUNDLE_SCHEMA,
            "output_root": str(output),
            "static_cache": str(static_cache),
            "static_receipt": str(static_receipt),
            "geometry_receipt": str(geometry_receipt),
            "source_manifest": str(source_manifest),
            "wps_namelist": (None if args.wps_namelist is None
                             else str(args.wps_namelist.resolve())),
            "experiment_config": str(output / PORTABLE_EXPERIMENT_CONFIG_NAME),
            "physics_profile": args.physics_profile,
            "acknowledgements": list(args.ack),
            "domain_spec": (None if args.domain_spec is None
                            else str(args.domain_spec.resolve())),
            "requested": {
                "preprocess_backend": args.preprocess_backend,
                "preprocess_workers": args.preprocess_workers,
                "pipeline_workers": args.pipeline_workers,
            },
        })
        benchmark.extend(("--chained-bundle", str(chain_document)))
    prepare_started = time.perf_counter()
    with prep_stage("root_prepare",
                    label="Build the start state and boundaries"):
        _run(benchmark, env)
    prepare_seconds = time.perf_counter() - prepare_started
    preparation_report_path = native / "preparation-report" / "report.json"
    preparation_report = json.loads(
        preparation_report_path.read_text(encoding="utf-8"))
    if args.sealed_prepared_cache:
        snapshot = native / "source-manifest.snapshot"
        with snapshot.open("xb") as stream:
            stream.write(source_manifest.read_bytes())
        if _sha256(snapshot) != args.source_manifest_sha256.lower():
            raise ValueError(
                "source manifest changed during sealed preparation")
    _validate_prepared_history_cadence(preparation_report, configured)
    (preprocess_receipt, preprocess_worker_budget,
     pipeline_worker_receipt) = _validated_worker_receipts(
         preparation_report,
         selected_backend=args.preprocess_backend,
         requested_preprocess_workers=args.preprocess_workers,
         requested_pipeline_workers=args.pipeline_workers,
         final_hour=final_hour,
         reused_physical_base=args.physical_base_prepared is not None,
         reused_posted_source=args.physical_input_provider is not None)
    physics_receipt = _validated_physics_receipt(
        preparation_report, requested_profile=args.physics_profile,
        expected_selection=configured.root.run)

    bridge_manifest = native / "native-bridge" / "SHA256SUMS"
    export_started = time.perf_counter()
    export_command = [
        sys.executable, "-m", "woof.wrf_direct",
        "--prepared-cache", str(native / "prepared-cache"),
        "--static-cache", str(static_cache),
        "--geometry-receipt", str(geometry_receipt),
        "--output", str(output / "wrf-native-input"),
        # woof.wrf_direct's --valid-time is the valid time of the state
        # being exported, which IS model time zero.  That flag is not part
        # of the cycle/model-start collision: an exported wrfinput has
        # exactly one valid time and no lead to be read against.
        "--valid-time", model_start_time.strftime(HRRR_TIME_FORMAT),
    ]
    export_command.extend(("--physics-profile", args.physics_profile)
                          if args.physics_profile is not None else ("--experiment-config-suite",))
    for acknowledgement in args.ack:
        export_command.extend(("--ack", acknowledgement))
    with prep_stage("wrf_export", label="Companion WRF files") as export_stage:
        stock_wrf_export = _stock_wrf_export(
            export_command, env,
            skip=args.skip_stock_wrf_export,
            required=args.require_stock_wrf_export,
            output=output / "wrf-native-input")
        export_stage["outcome"] = ("not_requested"
                                   if stock_wrf_export["status"] == "SKIPPED"
                                   else "produced")
    export_seconds = time.perf_counter() - export_started

    # ---- the portable half, published on every run ---------------------
    # Everything above this line is the certified native preparation and
    # is untouched.  This adds the documents a config-driven forecast
    # stage binds, from values already on disk: the cache's own source
    # identity, the preparation report's preprocessing receipt, and
    # digests of files this wrapper just wrote.
    #
    # It used to be gated on --wps-namelist, because the caller's
    # namelist was the one authority this route could not produce.  It
    # can now: the route holds the experiment those numbers come from,
    # and the bundle writer renders a namelist.wps from it and checks it
    # with the forecast stage's own geometry validator.  So publication
    # is the DEFAULT, and --wps-namelist means "bind mine instead of the
    # rendered one".  MEASURED 2026-08-18: a prep that exited 0 with
    # every wrfinput/wrfbdy on disk left a tree `woof sim` refused,
    # because nothing here had published the digests it binds.
    #
    # A publication that cannot be completed does not fail the
    # preparation -- the native tree it would have described is finished
    # and valid, and the native benchmark runs it.  The reason is
    # recorded and printed instead, and `woof sim` reads it back off
    # this receipt when it refuses the tree, so the chain says the same
    # thing at both ends.
    portable_bundle = None
    portable_bundle_refusal = None
    chained_handoff = (preparation_report.get("portable_bundle")
                       if chained else None)
    if chained_handoff is not None:
        # Published by the chained preparation at its seal, from the same
        # function and the same values this wrapper would pass it.
        portable_bundle = _verified_chained_handoff(chained_handoff, output)
    else:
        portable_bundle, portable_bundle_refusal = _publish_one_shot_bundle(
            args, output=output, native=native, static_cache=static_cache,
            static_receipt=static_receipt, geometry_receipt=geometry_receipt,
            bridge_manifest=bridge_manifest, namelist_input=namelist_input,
            source_manifest=source_manifest, valid_time=valid_time,
            source_forecast_hours=source_forecast_hours,
            model_forcing_hours=model_forcing_hours,
            preprocess_receipt=preprocess_receipt,
            namelist_invariant=namelist_invariant)
    return _finish_wrapper(
        args, output=output, native=native, valid_time=valid_time,
        model_start_time=model_start_time,
        source_forecast_hours=source_forecast_hours,
        model_forcing_hours=model_forcing_hours,
        bridge_manifest=bridge_manifest, started=started,
        static_seconds=static_seconds, prepare_seconds=prepare_seconds,
        export_seconds=export_seconds, stock_wrf_export=stock_wrf_export,
        portable_bundle=portable_bundle,
        portable_bundle_refusal=portable_bundle_refusal,
        preprocess_receipt=preprocess_receipt,
        preprocess_worker_budget=preprocess_worker_budget,
        pipeline_worker_receipt=pipeline_worker_receipt,
        preparation_report_path=preparation_report_path,
        physics_receipt=physics_receipt)


def _publish_one_shot_bundle(
        args, *, output, native, static_cache, static_receipt,
        geometry_receipt, bridge_manifest, namelist_input, source_manifest,
        valid_time, source_forecast_hours, model_forcing_hours,
        preprocess_receipt, namelist_invariant):
    """Publish the portable bundle over a sealed cache; ``(handoff, refusal)``."""

    portable_bundle = None
    portable_bundle_refusal = None
    try:
        # Read inside the guard on purpose: every input the publication
        # needs is an input the publication can fail on, and a run that
        # cannot publish still has a native tree worth keeping.
        cache_header = json.loads(
            (native / "prepared-cache" / "header.json").read_text(
                encoding="utf-8"))
        portable_bundle = publish_hrrr_prepared_bundle(
            output_root=output,
            prepared_cache=native / "prepared-cache",
            static_cache=static_cache, static_receipt=static_receipt,
            geometry_receipt=geometry_receipt,
            bridge_manifest=bridge_manifest,
            namelist_input=namelist_input,
            wps_namelist=(None if args.wps_namelist is None
                          else args.wps_namelist.resolve()),
            source_manifest=source_manifest,
            experiment_config=output / PORTABLE_EXPERIMENT_CONFIG_NAME,
            source_cycle=valid_time,
            source_forecast_hours=source_forecast_hours,
            model_forcing_hours=model_forcing_hours,
            # The VALIDATED receipt, not the raw report field: this is
            # the one the wrapper already proved describes the backend
            # that was asked for.
            preprocessing=preprocess_receipt,
            source_identity=cache_header["identity"]["source_identity"],
            physics_profile=args.physics_profile,
            expert_acknowledgements=tuple(args.ack),
            domain_spec=(args.domain_spec.resolve()
                         if args.domain_spec is not None else None),
            namelist_extension_invariant=namelist_invariant)
    except (HrrrBundleError, OSError, TypeError, ValueError, KeyError) as error:
        portable_bundle_refusal = (
            f"{type(error).__name__}: {error}")
        print(
            "prepare-hrrr-wrf: the native preparation finished, and the "
            "portable authorities a config-driven forecast binds "
            f"(proof.json, {PORTABLE_EXPERIMENT_CONFIG_NAME}, namelist.wps) "
            f"were NOT published: {portable_bundle_refusal}\n"
            "  what still works: the native bundle in this output root, run "
            "by tools/hrrr_single_domain_benchmark.py --prepared-cache\n"
            "  what does not: `woof sim` on this root, which binds those "
            "authorities and will say so",
            file=sys.stderr)
    return portable_bundle, portable_bundle_refusal


def _finish_wrapper(
        args, *, output, native, valid_time, model_start_time,
        source_forecast_hours, model_forcing_hours, bridge_manifest,
        started, static_seconds, prepare_seconds, export_seconds,
        stock_wrf_export, portable_bundle, portable_bundle_refusal,
        preprocess_receipt, preprocess_worker_budget,
        pipeline_worker_receipt, preparation_report_path,
        physics_receipt) -> int:
    """Write ``public-wrapper-result.json`` and print it."""

    result = {
        "status": "PASS",
        "stock_wrf_export": stock_wrf_export,
        # The three pins a forecast stage passes beside --prepared-root,
        # or None with the reason beside it when the publication could
        # not be completed.  Same rule as bridge_manifest_sha256 below:
        # each stage publishes the digests the stage after it must bind
        # -- and when it cannot, it records why, here, where `woof sim`
        # reads it back so both ends of the chain say the same thing.
        "portable_bundle": portable_bundle,
        "portable_bundle_refusal": portable_bundle_refusal,
        # Named only when it exists.  A path in a receipt is a claim
        # that the artifact is there; a skipped or refused export must
        # not leave one behind for a reader to trip over.
        "wrf_input": (str(output / "wrf-native-input")
                      if stock_wrf_export["status"] == "PASS" else None),
        "source_cycle": valid_time.isoformat(),
        "model_start_time": model_start_time.isoformat(),
        # What the single-domain FORECAST stage asks for next and nothing
        # printed.  Same rule the hierarchy follows with its own
        # preparation_receipt_sha256: each stage publishes the digest the
        # stage after it must bind, so the printed chain can be walked
        # without hashing a file by hand.  Named only when the artifact is
        # there -- a digest in a receipt is a claim that the file exists.
        "bridge_manifest_sha256": (
            sha256_file(bridge_manifest) if bridge_manifest.is_file()
            else None),
        "source_forecast_hours": list(source_forecast_hours),
        "model_forcing_hours": list(model_forcing_hours),
        # Backward-compatible alias: exporter/cache forcing is model-relative.
        "forcing_hours": list(model_forcing_hours),
        "history_interval_seconds": float(args.history_interval_seconds),
        "static_seconds": static_seconds,
        "prepare_seconds": prepare_seconds,
        "export_seconds": export_seconds,
        "total_seconds": time.perf_counter() - started,
        "preprocess_backend": preprocess_receipt,
        "preprocess_worker_budget": preprocess_worker_budget,
        "pipeline_decoder_workers": pipeline_worker_receipt,
        "preparation_report": str(preparation_report_path),
        "physics": physics_receipt,
    }
    if args.sealed_prepared_cache:
        prepared_header = json.loads(
            (native / "prepared-cache" / "header.json").read_text(
                encoding="utf-8"))
        if prepared_header.get("metadata", {}).get(
                "forcing_extension_mode") != "sealed-prefix-v1":
            raise RuntimeError(
                "benchmark omitted requested prepared-cache prefix seal")
        result["prepared_cache_contract"] = {
            "mode": "sealed-prefix-v1",
            "operation": "initial",
            "content_sha256": prepared_header.get("content_sha256"),
        }
    (output / "public-wrapper-result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
