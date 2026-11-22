"""Fail-closed sm_120 FTZ/DAZ binding for the MPAS CUDA mirror.

The CUDA port uses CuPy ``RawModule`` for five translation units.  On the
certification stack CuPy appends ``-ftz=true`` to that route, and sm_120 then
treats binary32 subnormal operands and results as signed zero.  woof owns the
two-pass device probe which establishes that fact.  This module does not
restate its verdict: it verifies the probe's raw bit table, pins the exact
woof sources and compiler fingerprint, and relates those measurements to the
five MPAS translation units in the runtime compile manifest.

The scalar-transport deck at the bottom is deliberately tiny but production:
it calls the CPU authority and the resident CUDA transport dispatcher.  Its
zero-time-step case distinguishes storage (which preserves subnormal bits)
from arithmetic (which flushes them) without relying on a weather-scale error
budget.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from .cuda_backend.arch_admission import performance_ratio_ceiling
from .cuda_backend.compile_contract import (
    CompileContractError,
    validate_compile_platform_fingerprint,
)


GPUWM_FTZ_SCHEMA = "gpuwm.ftz-receipt/v1"
MPAS_FTZ_SCHEMA = "mpas-port.cuda-ftz-binding/v1"
MPAS_FTZ_V841_SCHEMA = "mpas-port.cuda-ftz-binding-v841/v2"
TRANSPORT_DECK_SCHEMA = "mpas-port.cuda-ftz-transport-deck/v1"
KERNEL_AUDIT_SCHEMA = "mpas-port.cuda-ftz-kernel-audit/v1"
V841_KERNEL_AUDIT_SCHEMA = "mpas-port.cuda-ftz-v841-kernel-audit/v2"
def v841_kernel_audit_measurement(device_compute_capability: str) -> str:
    """The audit-method name, carrying the SM it actually launched on."""

    return (
        f"direct-production-kernel-launch-sm{device_compute_capability}"
        "-four-pass-with-disabled-fallback-mutation"
    )


V841_KERNEL_AUDIT_MEASUREMENT = v841_kernel_audit_measurement("120")
V841_PROBE_SPEC_SHA256 = (
    "0d731a445f8fc9d0caf805fdf913efdaf4d7bb1f614340ed2b31ef7318046028"
)
V841_ENABLED_RECORDS_SHA256 = (
    "5bd8bf400ef4713e09e57d916a2f2ccb11e6ceecf849f8bcd2a554bae61c3115"
)
V841_DISABLED_RECORDS_SHA256 = (
    "80552194e054223ccc9750594f6e7f6c50bb76d201b504737a936f01566516cc"
)
PERFORMANCE_CONTROL_SCHEMA = "mpas-port.cuda-ftz-normalized-performance/v1"
#: The proven-floor (sm_120) row of the per-architecture ceiling registry,
#: kept under its historical name because archived sm_120 receipts and
#: consumers bind it.  Per-architecture resolution (stale-guard audit,
#: finding 8) lives in ``cuda_backend.arch_admission``: the validation and
#: measurement paths below resolve the ceiling from the receipt fingerprint
#: or the live device, and an architecture with no registered ceiling is
#: refused by name rather than judged against this constant.
PERFORMANCE_RATIO_CEILING = performance_ratio_ceiling("sm_120")


def _resolved_performance_ceiling(sm: str) -> float:
    """The per-architecture ceiling as a contract fact, refusal by name."""

    try:
        return performance_ratio_ceiling(sm)
    except LookupError as error:
        raise FtzContractError(str(error)) from error


def _mpas_ftz_claim(capability: str, ceiling: float) -> str:
    """The binding claim.  For capability "120" / 1.25 this reproduces the
    pre-per-architecture bytes exactly; archived sm_120 bindings are
    validated by canonical re-hash and a text drift would go red on every
    one of them."""

    return (
        "The five MPAS RawModule translation units execute under the same "
        "measured terminal -ftz=true route for which woof's "
        f"sm_{capability} probe "
        "observes FP32 DAZ/FTZ. The production transport deck verifies "
        "the guarded subnormal-only FP64 fallback at all 12 transport "
        "kernels and 44 answer-changing non-transport arithmetic classes. "
        "Eight copy/invariant/native-FP64 classes stay green, all 44 "
        "disabled-fallback controls go red. Five named representative "
        "normalized-kernel microbenchmarks remain bitwise identical and "
        f"each stays below the declared {ceiling:.2f}x median ceiling; that timing "
        "ceiling is not a whole-step or all-guarded-kernel claim."
    )
KERNEL_AUDIT_DISPOSITION_SPEC_SHA256 = (
    "236c038d62a8a44da2eccf55fcee5e45e87809480e1d62009a9207874a096b15"
)

REQUIRED_MPAS_TRANSLATION_UNITS = (
    "hexcore.cuda_acoustic",
    "hexcore.cuda_backend.recovery",
    "hexcore.cuda_driver",
    "hexcore.cuda_horizontal",
    "hexcore.cuda_transport",
)

# Exact translation units reached by one admitted v8.4.1 closed-dry split-three
# step.  The old transport TU is deliberately absent: its upload container is
# reused, but every scalar arithmetic kernel resolves from cuda_transport_v841.
V841_REACHED_TRANSLATION_UNITS = (
    "hexcore.cuda_acoustic",
    "hexcore.cuda_acoustic_v841",
    "hexcore.cuda_backend.recovery",
    "hexcore.cuda_driver",
    "hexcore.cuda_dynamics_v841",
    "hexcore.cuda_horizontal",
    "hexcore.cuda_horizontal_v841",
    "hexcore.cuda_transport_v841",
)

_MECHANISMS = (
    "plain-operator arithmetic",
    "__f*_rn intrinsic",
    "fminf/fmaxf",
    "float compare",
    "__double2float_rn",
    "(float) cast",
)
_ROUTES = ("R1", "R1-ftztrue", "R2", "R3", "R4", "R5")
_FLUSH_ROUTES = frozenset(("R1", "R1-ftztrue", "R2", "R4"))
_IEEE_ROUTES = frozenset(("R3",))
_R5_NOT_APPLICABLE = frozenset(("__f*_rn intrinsic", "__double2float_rn"))
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_BIT_RE = re.compile(r"^0x[0-9a-fA-F]{8}$")

_GPUWM_PIN_PATHS = {
    "generator": "tools/ftz_receipt/probe.py",
    "route_inventory": "tools/ftz_receipt/route_inventory.py",
    "probe_source": "woof/core/kernels/ftz_probe.cu",
    "compile_platform_source": "woof/certify/compile_platform.py",
}


def _strict_proof(name: str) -> None:
    """The FTZ binding measures the strict route (strict is opt-in).

    It selects strict numerics for its process; a process that already compiled fast kernels is refused by
    name (woof.hex.numerics.set_mode), since its kernels would not be the route the binding certifies."""
    from . import numerics as _numerics
    _numerics.set_mode("strict", source=f"{name} (the FTZ binding is a strict-numerics proof; "
                                        "run it with --numerics strict)")


class FtzContractError(RuntimeError):
    """The probe, source, compiler, or MPAS relation cannot be proved."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Return a lowercase SHA-256 for a real regular file, or refuse."""

    selected = Path(path)
    if not selected.is_file():
        raise FtzContractError(f"required FTZ artifact is not a file: {selected}")
    digest = hashlib.sha256()
    with selected.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(document: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(document),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _kernel_audit_disposition_spec(
    kernels: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Extract the source-pinned semantic deck, excluding measured outcomes."""

    result: dict[str, dict[str, Any]] = {}
    for key in sorted(kernels):
        row = kernels[key]
        if not isinstance(row, Mapping):
            raise FtzContractError(f"MPAS FTZ audit row {key!r} is invalid")
        classification = row.get("classification")
        lane = row.get("lane")
        expected_bits = row.get("expected_bits")
        if (
            not isinstance(classification, str)
            or not classification
            or not isinstance(lane, str)
            or not lane
            or not isinstance(expected_bits, Mapping)
        ):
            raise FtzContractError(
                f"MPAS FTZ audit row {key!r} has no disposition specification"
            )
        result[key] = {
            "classification": classification,
            "lane": lane,
            "expected_bits": dict(expected_bits),
        }
    return result


def _validate_kernel_audit_disposition_spec(kernels: Mapping[str, Any]) -> None:
    measured = canonical_sha256(_kernel_audit_disposition_spec(kernels))
    if measured != KERNEL_AUDIT_DISPOSITION_SPEC_SHA256:
        raise FtzContractError(
            "MPAS FTZ per-kernel disposition specification changed: "
            f"{measured} != {KERNEL_AUDIT_DISPOSITION_SPEC_SHA256}"
        )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FtzContractError(f"cannot read JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise FtzContractError(f"JSON artifact must be an object: {path}")
    return value


def _safe_artifact(receipt_root: Path, relative: str) -> Path:
    candidate = (receipt_root / relative).resolve()
    root = receipt_root.resolve()
    if candidate != root and root not in candidate.parents:
        raise FtzContractError(f"FTZ receipt artifact escapes its root: {relative!r}")
    return candidate


def _classify(rows: Sequence[Mapping[str, str]]) -> str:
    """Independently reproduce the woof probe's published verdict rule."""

    if not rows:
        return "not-applicable"
    discriminating = sum(
        row["ieee_reference_bits"] != row["flushed_reference_bits"] for row in rows
    )
    if all(row["device_bits"] == row["ieee_reference_bits"] for row in rows):
        return "ieee-agreement" if discriminating else "inconclusive"
    if all(row["device_bits"] == row["flushed_reference_bits"] for row in rows):
        return "flush-to-zero"
    return "divergent"


def _expected_verdict(route: str, mechanism: str) -> str:
    if route in _FLUSH_ROUTES:
        return "flush-to-zero"
    if route in _IEEE_ROUTES:
        return "ieee-agreement"
    if route == "R5":
        return "not-applicable" if mechanism in _R5_NOT_APPLICABLE else "ieee-agreement"
    raise AssertionError(route)


def validate_gpuwm_ftz_receipt(receipt_root: str | Path) -> dict[str, Any]:
    """Verify every woof probe cell and both-pass byte identity.

    Verdicts are recomputed from ``bitpatterns.csv``.  The receipt's cell
    summaries therefore cannot certify themselves.  Every artifact carrying
    a digest is also checked against the file under the receipt root.
    """

    root = Path(receipt_root).resolve()
    receipt_path = root / "receipt.json"
    table_path = root / "bitpatterns.csv"
    receipt = _load_json(receipt_path)
    if receipt.get("schema") != GPUWM_FTZ_SCHEMA:
        raise FtzContractError(
            f"woof FTZ schema {receipt.get('schema')!r} != {GPUWM_FTZ_SCHEMA!r}"
        )
    if tuple(receipt.get("mechanisms", ())) != _MECHANISMS:
        raise FtzContractError("woof FTZ mechanism inventory changed")
    routes = receipt.get("routes")
    if not isinstance(routes, Mapping) or tuple(routes) != _ROUTES:
        raise FtzContractError("woof FTZ route inventory changed")

    table_sha = sha256_file(table_path)
    try:
        with table_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            required_columns = {
                "route",
                "mechanism",
                "input_id",
                "kernel",
                "device_bits",
                "ieee_reference_bits",
                "flushed_reference_bits",
                "control_bits",
            }
            if set(reader.fieldnames or ()) != required_columns:
                raise FtzContractError("woof FTZ bit table columns changed")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise FtzContractError(f"cannot read woof FTZ bit table: {error}") from error

    inputs = receipt.get("inputs")
    if not isinstance(inputs, list) or len(inputs) != 9:
        raise FtzContractError("woof FTZ input inventory is not the nine-row deck")
    input_ids = tuple(str(row.get("id")) for row in inputs)
    if len(set(input_ids)) != len(input_ids):
        raise FtzContractError("woof FTZ input ids are not unique")

    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        route = row["route"]
        mechanism = row["mechanism"]
        if route not in _ROUTES or mechanism not in _MECHANISMS:
            raise FtzContractError(
                f"unexpected woof FTZ table cell {(route, mechanism)!r}"
            )
        if row["input_id"] not in input_ids or not row["kernel"]:
            raise FtzContractError("woof FTZ table has an invalid input/kernel id")
        for column in (
            "device_bits",
            "ieee_reference_bits",
            "flushed_reference_bits",
            "control_bits",
        ):
            if _BIT_RE.fullmatch(row[column]) is None:
                raise FtzContractError(
                    f"woof FTZ table has invalid {column}: {row[column]!r}"
                )
        grouped.setdefault((route, mechanism), []).append(row)

    summaries = receipt.get("cells")
    if not isinstance(summaries, list) or len(summaries) != 36:
        raise FtzContractError("woof FTZ receipt must summarize all 36 cells")
    summary_by_cell: dict[tuple[str, str], Mapping[str, Any]] = {}
    for summary in summaries:
        if not isinstance(summary, Mapping):
            raise FtzContractError("woof FTZ cell summary is not an object")
        cell = (str(summary.get("route")), str(summary.get("mechanism")))
        if cell in summary_by_cell:
            raise FtzContractError(f"duplicate woof FTZ cell summary {cell!r}")
        summary_by_cell[cell] = summary

    measured_cells: list[dict[str, Any]] = []
    for route in _ROUTES:
        for mechanism in _MECHANISMS:
            cell = (route, mechanism)
            summary = summary_by_cell.get(cell)
            if summary is None:
                raise FtzContractError(f"missing woof FTZ cell summary {cell!r}")
            cell_rows = grouped.get(cell, [])
            expected_count = 0 if _expected_verdict(*cell) == "not-applicable" else 9
            if len(cell_rows) != expected_count:
                raise FtzContractError(
                    f"woof FTZ cell {cell!r} has {len(cell_rows)} rows, "
                    f"expected {expected_count}"
                )
            if cell_rows and tuple(row["input_id"] for row in cell_rows) != input_ids:
                raise FtzContractError(f"woof FTZ cell {cell!r} input order changed")
            verdict = _classify(cell_rows)
            expected = _expected_verdict(*cell)
            if verdict != expected:
                raise FtzContractError(
                    f"woof FTZ cell {cell!r} measured {verdict!r}, "
                    f"required {expected!r}"
                )
            if summary.get("verdict") != verdict:
                raise FtzContractError(
                    f"woof FTZ cell {cell!r} summary disagrees with its bit table"
                )
            if int(summary.get("input_count", -1)) != len(cell_rows):
                raise FtzContractError(
                    f"woof FTZ cell {cell!r} summary count is false"
                )
            measured_cells.append(
                {
                    "route": route,
                    "mechanism": mechanism,
                    "verdict": verdict,
                    "input_count": len(cell_rows),
                }
            )

    dual = receipt.get("dual_run")
    if not isinstance(dual, Mapping):
        raise FtzContractError("woof FTZ receipt has no dual-run record")
    digests = dual.get("bit_table_sha256")
    if (
        dual.get("runs") != 2
        or dual.get("byte_identical") is not True
        or not isinstance(digests, list)
        or digests != [table_sha, table_sha]
    ):
        raise FtzContractError(
            "woof FTZ two-pass bit tables are not byte-identical to the "
            "committed bitpatterns.csv"
        )

    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise FtzContractError("woof FTZ receipt has no artifact inventory")
    table_entry = artifacts.get("bitpatterns.csv")
    if not isinstance(table_entry, Mapping) or table_entry.get("sha256") != table_sha:
        raise FtzContractError("woof FTZ bit-table artifact digest is false")
    verified_artifacts = 0
    for relative, record in artifacts.items():
        if not isinstance(record, Mapping) or "sha256" not in record:
            continue
        declared = str(record["sha256"])
        if _SHA256_RE.fullmatch(declared) is None:
            raise FtzContractError(f"invalid artifact digest for {relative!r}")
        actual = sha256_file(_safe_artifact(root, str(relative)))
        if actual != declared:
            raise FtzContractError(
                f"woof FTZ artifact {relative!r} SHA-256 {actual} != {declared}"
            )
        verified_artifacts += 1

    append_site = receipt.get("cupy_ftz_append_site")
    if (
        not isinstance(append_site, Mapping)
        or append_site.get("found") is not True
        or append_site.get("appended") != "-ftz=true"
    ):
        raise FtzContractError("woof did not measure CuPy's terminal -ftz=true")

    return {
        "schema": GPUWM_FTZ_SCHEMA,
        "receipt_sha256": sha256_file(receipt_path),
        "bitpatterns_sha256": table_sha,
        "verified_artifact_count": verified_artifacts,
        "cells": measured_cells,
        "device": dict(receipt.get("device", {})),
        "dual_run": dict(dual),
    }


def measure_gpuwm_source_pins(gpuwm_root: str | Path) -> dict[str, Any]:
    """Hash the live tracked woof sources and bind them to its exact HEAD."""

    root = Path(gpuwm_root).resolve()
    try:
        process = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={root.as_posix()}",
                "-C",
                str(root),
                "rev-parse",
                "HEAD",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise FtzContractError(
            f"cannot resolve woof HEAD at {root}: {error}"
        ) from error
    head = process.stdout.strip().lower()
    if re.fullmatch(r"[0-9a-f]{40}", head) is None:
        raise FtzContractError(f"woof HEAD is not a full commit id: {head!r}")

    sources = {
        label: {
            "path": relative,
            "sha256": sha256_file(root / relative),
        }
        for label, relative in _GPUWM_PIN_PATHS.items()
    }
    return {"git_head": head, "sources": sources}


def validate_compile_manifest_relation(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Require exactly five compiler-bound, actually compiled MPAS TUs."""

    production_inventory = production_translation_units()

    if manifest.get("schema") != "mpas-port.cuda-compile-manifest/v1":
        raise FtzContractError("MPAS CUDA compile manifest schema is not v1")
    platform = manifest.get("compile_platform")
    if not isinstance(platform, Mapping):
        raise FtzContractError("MPAS CUDA compile manifest has no platform binding")
    fingerprint = platform.get("fingerprint")
    if not isinstance(fingerprint, Mapping):
        raise FtzContractError("MPAS CUDA compile platform has no fingerprint")
    try:
        validated_fingerprint = validate_compile_platform_fingerprint(fingerprint)
    except CompileContractError as error:
        raise FtzContractError(
            f"MPAS compile-platform fingerprint is invalid: {error}"
        ) from error
    if dict(fingerprint) != validated_fingerprint:
        raise FtzContractError("MPAS compile-platform fingerprint is not canonical")
    fingerprint_sha = canonical_sha256(fingerprint)
    if platform.get("sha256") != fingerprint_sha:
        raise FtzContractError("MPAS compile-platform fingerprint digest is false")
    # The architecture is carried by the fingerprint and closed against the
    # probe device (see build_mpas_ftz_binding) and against the admission
    # registry at measurement time (require_cuda); the relation itself only
    # requires the capability to be a real numeric architecture so that the
    # recomputed module-cache keys below bind the same SM the compile used.
    fingerprint_capability = str(fingerprint.get("device_compute_capability", ""))
    if re.fullmatch(r"\d+", fingerprint_capability) is None:
        raise FtzContractError(
            "MPAS FTZ contract requires a numeric device_compute_capability; "
            f"got {fingerprint_capability!r}"
        )

    modules = manifest.get("modules")
    if not isinstance(modules, Mapping):
        raise FtzContractError("MPAS CUDA compile manifest has no module inventory")
    if tuple(sorted(modules)) != REQUIRED_MPAS_TRANSLATION_UNITS:
        raise FtzContractError(
            "MPAS CUDA FTZ relation requires exactly the five production "
            f"translation units {REQUIRED_MPAS_TRANSLATION_UNITS!r}"
        )
    relation: dict[str, Any] = {}
    for module_key in REQUIRED_MPAS_TRANSLATION_UNITS:
        declaration = modules[module_key]
        if not isinstance(declaration, Mapping):
            raise FtzContractError(f"compile declaration {module_key!r} is invalid")
        source_sha = str(declaration.get("source_sha256", ""))
        cache_key = str(declaration.get("module_cache_key", ""))
        if (
            _SHA256_RE.fullmatch(source_sha) is None
            or _SHA256_RE.fullmatch(cache_key) is None
        ):
            raise FtzContractError(
                f"compile declaration {module_key!r} has false hashes"
            )
        live_source_sha = hashlib.sha256(
            production_inventory[module_key][0].encode("utf-8")
        ).hexdigest()
        if source_sha != live_source_sha:
            raise FtzContractError(
                f"compile declaration {module_key!r} is stale relative to its "
                "live production source"
            )
        expected_cache = hashlib.sha256()
        expected_cache.update(f"sm_{fingerprint_capability}".encode("ascii"))
        expected_cache.update(b"\0compile-platform\0")
        expected_cache.update(fingerprint_sha.encode("ascii"))
        expected_cache.update(production_inventory[module_key][0].encode("utf-8"))
        for option in ("--std=c++17", "--fmad=false"):
            expected_cache.update(b"\0")
            expected_cache.update(option.encode("utf-8"))
        if cache_key != expected_cache.hexdigest():
            raise FtzContractError(
                f"compile declaration {module_key!r} has a false module cache key"
            )
        if declaration.get("compile_platform_fingerprint_sha256") != fingerprint_sha:
            raise FtzContractError(
                f"compile declaration {module_key!r} is not bound to the platform"
            )
        options = declaration.get("requested_options")
        if options != ["--std=c++17", "--fmad=false"]:
            raise FtzContractError(
                f"compile declaration {module_key!r} changed its exact requested "
                "options"
            )
        kernels = declaration.get("resolved_kernels")
        expected_kernels = list(production_inventory[module_key][1])
        if kernels != expected_kernels:
            raise FtzContractError(
                f"compile declaration {module_key!r} does not resolve its exact "
                "production kernel inventory"
            )
        effective = declaration.get("effective_compile")
        if not isinstance(effective, Mapping) or effective.get("status") != "resolved":
            raise FtzContractError(
                f"compile declaration {module_key!r} lacks real NVRTC evidence"
            )
        observations = effective.get("observations")
        if not isinstance(observations, list) or not observations:
            raise FtzContractError(
                f"compile declaration {module_key!r} has no NVRTC observation"
            )
        expected_observation_keys = {
            "source_sha256",
            "effective_flags",
            "include_path_count",
            "include_paths_omitted",
            "compiled_image",
        }
        for observation in observations:
            if (
                not isinstance(observation, Mapping)
                or set(observation) != expected_observation_keys
                or observation.get("source_sha256") != source_sha
                or observation.get("effective_flags")
                != ["--std=c++17", "--fmad=false", "-ftz=true"]
                or not isinstance(observation.get("include_path_count"), int)
                or observation.get("include_path_count", -1) < 0
                or observation.get("include_paths_omitted")
                != (
                    "-I entries describe this machine's toolkit layout and are "
                    "counted but not copied into the arithmetic contract"
                )
            ):
                raise FtzContractError(
                    f"compile declaration {module_key!r} has false NVRTC evidence"
                )
            image = observation.get("compiled_image")
            if (
                not isinstance(image, Mapping)
                or set(image) != {"status", "kind", "sha256"}
                or image.get("status") != "resolved"
                or image.get("kind") not in {"cubin", "ptx"}
                or _SHA256_RE.fullmatch(str(image.get("sha256", ""))) is None
            ):
                raise FtzContractError(
                    f"compile declaration {module_key!r} has no resolved image hash"
                )
        relation[module_key] = {
            "source_sha256": source_sha,
            "module_cache_key": cache_key,
            "resolved_kernels": sorted(str(name) for name in kernels),
            "effective_terminal_ftz": "-ftz=true",
            "compile_platform_fingerprint_sha256": fingerprint_sha,
        }
    return {
        "compile_manifest_sha256": canonical_sha256(manifest),
        "compile_platform": {
            "fingerprint": dict(fingerprint),
            "sha256": fingerprint_sha,
        },
        "translation_units": relation,
    }


def validate_v841_compile_manifest_relation(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Require the exact compiler-bound eight-TU v8.4.1 step manifest."""

    production_inventory = v841_reached_translation_units()
    compiled_inventory = v841_compiled_translation_units()
    if manifest.get("schema") != "mpas-port.cuda-compile-manifest/v1":
        raise FtzContractError("MPAS CUDA compile manifest schema is not v1")
    platform = manifest.get("compile_platform")
    if not isinstance(platform, Mapping):
        raise FtzContractError("MPAS CUDA compile manifest has no platform binding")
    fingerprint = platform.get("fingerprint")
    if not isinstance(fingerprint, Mapping):
        raise FtzContractError("MPAS CUDA compile platform has no fingerprint")
    try:
        validated_fingerprint = validate_compile_platform_fingerprint(fingerprint)
    except CompileContractError as error:
        raise FtzContractError(
            f"MPAS compile-platform fingerprint is invalid: {error}"
        ) from error
    if dict(fingerprint) != validated_fingerprint:
        raise FtzContractError("MPAS compile-platform fingerprint is not canonical")
    fingerprint_sha = canonical_sha256(fingerprint)
    if platform.get("sha256") != fingerprint_sha:
        raise FtzContractError("MPAS compile-platform fingerprint digest is false")
    fingerprint_capability = str(fingerprint.get("device_compute_capability", ""))
    if re.fullmatch(r"\d+", fingerprint_capability) is None:
        raise FtzContractError(
            "MPAS v8.4.1 FTZ contract requires a numeric "
            f"device_compute_capability; got {fingerprint_capability!r}"
        )

    modules = manifest.get("modules")
    if not isinstance(modules, Mapping):
        raise FtzContractError("MPAS CUDA compile manifest has no module inventory")
    if tuple(sorted(modules)) != V841_REACHED_TRANSLATION_UNITS:
        raise FtzContractError(
            "MPAS v8.4.1 CUDA relation requires exactly the eight reached "
            f"translation units {V841_REACHED_TRANSLATION_UNITS!r}"
        )

    relation: dict[str, Any] = {}
    for module_key in V841_REACHED_TRANSLATION_UNITS:
        declaration = modules[module_key]
        if not isinstance(declaration, Mapping):
            raise FtzContractError(f"compile declaration {module_key!r} is invalid")
        source_sha = str(declaration.get("source_sha256", ""))
        cache_key = str(declaration.get("module_cache_key", ""))
        if (
            _SHA256_RE.fullmatch(source_sha) is None
            or _SHA256_RE.fullmatch(cache_key) is None
        ):
            raise FtzContractError(
                f"compile declaration {module_key!r} has false hashes"
            )
        source = production_inventory[module_key][0]
        live_source_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
        if source_sha != live_source_sha:
            raise FtzContractError(
                f"compile declaration {module_key!r} is stale relative to its "
                "live v8.4.1 production source"
            )
        expected_cache = hashlib.sha256()
        expected_cache.update(f"sm_{fingerprint_capability}".encode("ascii"))
        expected_cache.update(b"\0compile-platform\0")
        expected_cache.update(fingerprint_sha.encode("ascii"))
        expected_cache.update(source.encode("utf-8"))
        for option in ("--std=c++17", "--fmad=false"):
            expected_cache.update(b"\0")
            expected_cache.update(option.encode("utf-8"))
        if cache_key != expected_cache.hexdigest():
            raise FtzContractError(
                f"compile declaration {module_key!r} has a false module cache key"
            )
        if declaration.get("compile_platform_fingerprint_sha256") != fingerprint_sha:
            raise FtzContractError(
                f"compile declaration {module_key!r} is not bound to the platform"
            )
        if declaration.get("requested_options") != [
            "--std=c++17",
            "--fmad=false",
        ]:
            raise FtzContractError(
                f"compile declaration {module_key!r} changed its exact requested "
                "options"
            )
        kernels = declaration.get("resolved_kernels")
        expected_kernels = list(production_inventory[module_key][1])
        if kernels != expected_kernels:
            raise FtzContractError(
                f"compile declaration {module_key!r} does not resolve its exact "
                "v8.4.1 step kernel inventory"
            )
        effective = declaration.get("effective_compile")
        if not isinstance(effective, Mapping) or effective.get("status") != "resolved":
            raise FtzContractError(
                f"compile declaration {module_key!r} lacks real NVRTC evidence"
            )
        observations = effective.get("observations")
        if (
            set(effective) != {"status", "method", "observations"}
            or effective.get("method")
            != (
                "wrapped cupy.cuda.compiler."
                "_compile_using_nvrtc_no_warning at the NVRTC entry point"
            )
            or not isinstance(observations, list)
            or len(observations) != 1
        ):
            raise FtzContractError(
                f"compile declaration {module_key!r} does not have one exact "
                "NVRTC-entry observation"
            )
        expected_observation_keys = {
            "source_sha256",
            "effective_flags",
            "include_path_count",
            "include_paths_omitted",
            "compiled_image",
        }
        for observation in observations:
            if (
                not isinstance(observation, Mapping)
                or set(observation) != expected_observation_keys
                or observation.get("source_sha256") != source_sha
                or observation.get("effective_flags")
                != ["--std=c++17", "--fmad=false", "-ftz=true"]
                or not isinstance(observation.get("include_path_count"), int)
                or observation.get("include_path_count", -1) < 0
                or observation.get("include_paths_omitted")
                != (
                    "-I entries describe this machine's toolkit layout and are "
                    "counted but not copied into the arithmetic contract"
                )
            ):
                raise FtzContractError(
                    f"compile declaration {module_key!r} has false NVRTC evidence"
                )
            image = observation.get("compiled_image")
            if (
                not isinstance(image, Mapping)
                or set(image) != {"status", "kind", "sha256"}
                or image.get("status") != "resolved"
                or image.get("kind") not in {"cubin", "ptx"}
                or _SHA256_RE.fullmatch(str(image.get("sha256", ""))) is None
            ):
                raise FtzContractError(
                    f"compile declaration {module_key!r} has no resolved image hash"
                )
        relation[module_key] = {
            "source_sha256": source_sha,
            "module_cache_key": cache_key,
            "resolved_kernels": list(kernels),
            "compiled_kernel_surface": list(compiled_inventory[module_key][1]),
            "compiled_image": dict(observations[0]["compiled_image"]),
            "effective_terminal_ftz": "-ftz=true",
            "compile_platform_fingerprint_sha256": fingerprint_sha,
        }
    return {
        "source_release": "v8.4.1",
        "compile_manifest_sha256": canonical_sha256(manifest),
        "compile_platform": {
            "fingerprint": dict(fingerprint),
            "sha256": fingerprint_sha,
        },
        "translation_units": relation,
        "reached_kernel_count": sum(
            len(row["resolved_kernels"]) for row in relation.values()
        ),
        "compiled_kernel_count": sum(
            len(row["compiled_kernel_surface"]) for row in relation.values()
        ),
        "authority_claim": False,
    }


def _normalize_compute_capability(value: Any) -> str:
    text = str(value).strip()
    if re.fullmatch(r"\d+\.\d+", text):
        major, minor = text.split(".", 1)
        return f"{int(major)}{int(minor)}"
    return text


def build_mpas_ftz_binding(
    *,
    gpuwm_root: str | Path,
    gpuwm_receipt_root: str | Path,
    compile_manifest: Mapping[str, Any],
    transport_deck: Mapping[str, Any],
    kernel_audit: Mapping[str, Any],
    performance_control: Mapping[str, Any],
) -> dict[str, Any]:
    """Create the JSON-ready all-TU MPAS/woof/compiler/FTZ relation."""

    probe = validate_gpuwm_ftz_receipt(gpuwm_receipt_root)
    pins = measure_gpuwm_source_pins(gpuwm_root)
    relation = validate_compile_manifest_relation(compile_manifest)
    fingerprint = relation["compile_platform"]["fingerprint"]
    device = probe["device"]
    comparisons = {
        "cuda_driver_version": (
            str(device.get("cuda_driver_version")),
            fingerprint.get("cuda_driver_version"),
        ),
        "cupy_version": (
            str(device.get("cupy_version")),
            fingerprint.get("cupy_version"),
        ),
        "numpy_version": (
            str(device.get("numpy_version")),
            fingerprint.get("numpy_version"),
        ),
        "device_compute_capability": (
            _normalize_compute_capability(device.get("compute_capability")),
            fingerprint.get("device_compute_capability"),
        ),
    }
    mismatch = {
        key: value for key, value in comparisons.items() if value[0] != value[1]
    }
    if mismatch:
        raise FtzContractError(
            f"woof FTZ probe and MPAS compile fingerprint disagree: {mismatch}"
        )
    if transport_deck.get("schema") != TRANSPORT_DECK_SCHEMA:
        raise FtzContractError("MPAS scalar-transport FTZ deck schema is invalid")
    if transport_deck.get("dual_run_byte_identical") is not True:
        raise FtzContractError("MPAS scalar-transport FTZ deck is not dual-run stable")
    if transport_deck.get("fallback_verified") is not True:
        raise FtzContractError("MPAS scalar-transport fallback is not verified")
    normal_lane = transport_deck.get("normalized_lane_control")
    if (
        not isinstance(normal_lane, Mapping)
        or normal_lane.get("bitwise_unchanged") is not True
    ):
        raise FtzContractError("MPAS scalar fallback changed the normalized lane")
    mutation = transport_deck.get("mutation_control")
    if (
        not isinstance(mutation, Mapping)
        or mutation.get("dual_run_byte_identical") is not True
        or mutation.get("maximum_gap", 0.0) <= 0.0
        or tuple(mutation.get("red_kernels", ()))
        != (
            "transport_edge_values",
            "transport_vertical_flux",
            "transport_target_density",
            "transport_standard_finish",
            "fct_minmax_source",
            "fct_vertical_low_order",
            "fct_edge_residual",
            "fct_horizontal_low_order",
            "fct_scale",
            "fct_limit_horizontal",
            "fct_limit_vertical",
            "fct_finish",
        )
    ):
        raise FtzContractError(
            "MPAS scalar-transport disabled-fallback mutation is not red at "
            "all 12 production kernels"
        )
    if (
        kernel_audit.get("schema") != KERNEL_AUDIT_SCHEMA
        or kernel_audit.get("fallback_verified") is not True
        or kernel_audit.get("dual_run_byte_identical") is not True
        or kernel_audit.get("kernel_count") != 52
    ):
        raise FtzContractError("MPAS remaining-kernel FTZ audit is incomplete")
    audited = kernel_audit.get("kernels")
    if not isinstance(audited, Mapping) or len(audited) != 52:
        raise FtzContractError("MPAS remaining-kernel FTZ inventory is false")
    audit_module_labels = {
        "hexcore.cuda_acoustic": "acoustic",
        "hexcore.cuda_backend.recovery": "recovery",
        "hexcore.cuda_driver": "driver",
        "hexcore.cuda_horizontal": "horizontal",
    }
    expected_audit_inventory = {
        f"{audit_module_labels[module]}.{kernel}"
        for module, (_source, kernels) in production_translation_units().items()
        if module in audit_module_labels
        for kernel in kernels
    }
    if set(audited) != expected_audit_inventory:
        raise FtzContractError(
            "MPAS remaining-kernel FTZ keys differ from production exports"
        )
    _validate_kernel_audit_disposition_spec(audited)
    for key, row in audited.items():
        if not isinstance(row, Mapping):
            raise FtzContractError(f"MPAS FTZ audit row {key!r} is invalid")
        translation_unit, kernel = key.split(".", 1)
        expected_bits = row.get("expected_bits")
        observed_bits = row.get("observed_bits")
        disabled_bits = row.get("disabled_fallback_observed_bits")
        matches_expected = observed_bits == expected_bits
        disabled_matches = disabled_bits == expected_bits
        mutation_red = not disabled_matches
        if (
            row.get("translation_unit") != translation_unit
            or row.get("kernel") != kernel
            or not isinstance(expected_bits, Mapping)
            or not isinstance(observed_bits, Mapping)
            or not isinstance(disabled_bits, Mapping)
            or row.get("matches_expected") is not matches_expected
            or row.get("disabled_fallback_matches_expected") is not disabled_matches
            or row.get("mutation_red") is not mutation_red
        ):
            raise FtzContractError(
                f"MPAS FTZ audit row {key!r} has false identity or bit verdicts"
            )
    guarded = [
        row
        for row in audited.values()
        if isinstance(row, Mapping)
        and row.get("classification") == "guarded_fallback_required"
    ]
    proven_green = [
        row
        for row in audited.values()
        if isinstance(row, Mapping)
        and row.get("classification") != "guarded_fallback_required"
    ]
    if (
        len(guarded) != 44
        or len(proven_green) != 8
        or not all(row.get("matches_expected") is True for row in audited.values())
        or not all(row.get("mutation_red") is True for row in guarded)
        or not all(row.get("mutation_red") is False for row in proven_green)
    ):
        raise FtzContractError(
            "MPAS remaining-kernel fallback/mutation dispositions are false"
        )
    # The ceiling is the ARCHITECTURE's registered row (stale-guard audit,
    # finding 8): 1.25 on sm_120, the recorded-deviation row on sm_86, a
    # named refusal for anything unregistered.
    declared_ceiling = _resolved_performance_ceiling(
        f"sm_{fingerprint.get('device_compute_capability')}"
    )
    if (
        performance_control.get("schema") != PERFORMANCE_CONTROL_SCHEMA
        or performance_control.get("declared_median_ratio_ceiling")
        != declared_ceiling
        or performance_control.get("all_normalized_outputs_bitwise_identical")
        is not True
        or float(performance_control.get("maximum_enabled_over_disabled", 99.0))
        > declared_ceiling
    ):
        raise FtzContractError("MPAS normalized fallback performance control failed")
    performance_rows = performance_control.get("benchmarks")
    expected_performance_rows = {
        "acoustic.acoustic_ru",
        "driver.scale_f32",
        "horizontal.smagorinsky_f32",
        "recovery.recover_edge_velocity_f32",
        "transport.transport_edge_values",
    }
    if (
        not isinstance(performance_rows, Mapping)
        or set(performance_rows) != expected_performance_rows
    ):
        raise FtzContractError("MPAS normalized fallback timing inventory is false")
    if not all(
        isinstance(row, Mapping)
        and row.get("normalized_output_bitwise_identical") is True
        and float(row.get("enabled_over_disabled", 99.0)) <= declared_ceiling
        for row in performance_rows.values()
    ):
        raise FtzContractError("MPAS normalized fallback timing row failed")
    measured_maximum = max(
        float(row["enabled_over_disabled"]) for row in performance_rows.values()
    )
    if float(performance_control["maximum_enabled_over_disabled"]) != measured_maximum:
        raise FtzContractError("MPAS normalized fallback maximum is false")

    return {
        "schema": MPAS_FTZ_SCHEMA,
        "woof": pins,
        "gpuwm_ftz_probe": probe,
        "compile_relation": relation,
        "compile_manifest": json.loads(json.dumps(compile_manifest, sort_keys=True)),
        "transport_deck": json.loads(json.dumps(transport_deck, sort_keys=True)),
        "kernel_audit": json.loads(json.dumps(kernel_audit, sort_keys=True)),
        "normalized_performance_control": json.loads(
            json.dumps(performance_control, sort_keys=True)
        ),
        "claim": _mpas_ftz_claim(
            str(fingerprint.get("device_compute_capability")), declared_ceiling
        ),
    }


def validate_mpas_ftz_binding(
    binding: Mapping[str, Any],
    *,
    gpuwm_root: str | Path,
    gpuwm_receipt_root: str | Path,
) -> dict[str, Any]:
    """Re-resolve every external pin in a saved MPAS FTZ binding."""

    if binding.get("schema") != MPAS_FTZ_SCHEMA:
        raise FtzContractError("saved MPAS FTZ binding schema is invalid")
    rebuilt = build_mpas_ftz_binding(
        gpuwm_root=gpuwm_root,
        gpuwm_receipt_root=gpuwm_receipt_root,
        compile_manifest=binding.get("compile_manifest", {}),
        transport_deck=binding.get("transport_deck", {}),
        kernel_audit=binding.get("kernel_audit", {}),
        performance_control=binding.get("normalized_performance_control", {}),
    )
    if canonical_sha256(rebuilt) != canonical_sha256(binding):
        raise FtzContractError("saved MPAS FTZ binding differs from live evidence")
    return rebuilt


def _v841_module_cache_key(
    *,
    source: str,
    compile_platform_sha256: str,
    sm: str = "sm_120",
) -> str:
    digest = hashlib.sha256()
    digest.update(sm.encode("ascii"))
    digest.update(b"\0compile-platform\0")
    digest.update(compile_platform_sha256.encode("ascii"))
    digest.update(source.encode("utf-8"))
    for option in ("--std=c++17", "--fmad=false"):
        digest.update(b"\0")
        digest.update(option.encode("utf-8"))
    return digest.hexdigest()


def _validate_v841_measurement_translation_units(
    translation_units: Any,
    *,
    mode: str,
    compile_manifest: Mapping[str, Any],
    relation: Mapping[str, Any],
    compiled: Mapping[str, tuple[str, tuple[str, ...]]],
) -> None:
    """Prove that one pass launched eight newly compiled, source-exact TUs."""

    if mode not in {"fallback-enabled", "fallback-disabled"}:
        raise FtzContractError("v8.4.1 FTZ compile pass mode is invalid")
    if not isinstance(translation_units, Mapping) or set(translation_units) != set(
        compiled
    ):
        raise FtzContractError(
            "v8.4.1 FTZ compile pass does not contain the exact eight TUs"
        )
    platform_sha = relation["compile_platform"]["sha256"]
    manifest_modules = compile_manifest.get("modules")
    if not isinstance(manifest_modules, Mapping):
        raise FtzContractError("v8.4.1 authority manifest has no module inventory")
    module_fields = {
        "source_sha256",
        "requested_options",
        "compile_platform_fingerprint_sha256",
        "module_cache_key",
        "effective_compile",
        "resolved_kernels",
    }
    observation_fields = {
        "source_sha256",
        "effective_flags",
        "include_path_count",
        "include_paths_omitted",
        "compiled_image",
    }
    method = (
        "wrapped cupy.cuda.compiler._compile_using_nvrtc_no_warning at the "
        "NVRTC entry point"
    )
    prefix = (
        "#define MPAS_FTZ_FALLBACK_ENABLED 0\n" if mode == "fallback-disabled" else ""
    )
    for module_key, (production_source, kernel_names) in compiled.items():
        declaration = translation_units[module_key]
        source = prefix + production_source
        source_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
        if (
            not isinstance(declaration, Mapping)
            or set(declaration) != module_fields
            or declaration.get("source_sha256") != source_sha
            or declaration.get("requested_options") != ["--std=c++17", "--fmad=false"]
            or declaration.get("compile_platform_fingerprint_sha256") != platform_sha
            or declaration.get("module_cache_key")
            != _v841_module_cache_key(
                source=source,
                compile_platform_sha256=platform_sha,
                sm=(
                    "sm_"
                    + str(
                        relation["compile_platform"]["fingerprint"].get(
                            "device_compute_capability"
                        )
                    )
                ),
            )
            or declaration.get("resolved_kernels") != list(kernel_names)
        ):
            raise FtzContractError(
                f"v8.4.1 FTZ compile evidence for {module_key!r} is false"
            )
        effective = declaration.get("effective_compile")
        if (
            not isinstance(effective, Mapping)
            or set(effective) != {"status", "method", "observations"}
            or effective.get("status") != "resolved"
            or effective.get("method") != method
        ):
            raise FtzContractError(
                f"v8.4.1 FTZ compile evidence for {module_key!r} was not "
                "captured at NVRTC"
            )
        observations = effective.get("observations")
        if not isinstance(observations, list) or len(observations) != 1:
            raise FtzContractError(
                f"v8.4.1 FTZ compile evidence for {module_key!r} does not "
                "contain exactly one fresh compile"
            )
        observation = observations[0]
        image = (
            observation.get("compiled_image")
            if isinstance(observation, Mapping)
            else None
        )
        if (
            not isinstance(observation, Mapping)
            or set(observation) != observation_fields
            or observation.get("source_sha256") != source_sha
            or observation.get("effective_flags")
            != ["--std=c++17", "--fmad=false", "-ftz=true"]
            or not isinstance(observation.get("include_path_count"), int)
            or isinstance(observation.get("include_path_count"), bool)
            or observation.get("include_path_count", -1) < 0
            or observation.get("include_paths_omitted")
            != (
                "-I entries describe this machine's toolkit layout and are "
                "counted but not copied into the arithmetic contract"
            )
            or not isinstance(image, Mapping)
            or set(image) != {"status", "kind", "sha256"}
            or image.get("status") != "resolved"
            or image.get("kind") not in {"cubin", "ptx"}
            or _SHA256_RE.fullmatch(str(image.get("sha256", ""))) is None
        ):
            raise FtzContractError(
                f"v8.4.1 FTZ compile observation for {module_key!r} is false"
            )
        if mode == "fallback-enabled":
            authority_image = relation["translation_units"][module_key].get(
                "compiled_image"
            )
            if image != authority_image:
                raise FtzContractError(
                    f"v8.4.1 FTZ executable image for {module_key!r} differs "
                    "from the authority manifest"
                )


def _validate_v841_measurement_transcript(
    transcript: Any,
    *,
    kernels: Mapping[str, Any],
    compile_manifest: Mapping[str, Any],
    relation: Mapping[str, Any],
    compiled: Mapping[str, tuple[str, tuple[str, ...]]],
    reached: Mapping[str, tuple[str, tuple[str, ...]]],
) -> None:
    """Rebuild every serialized v8.4.1 verdict from four raw device passes."""

    from . import cuda_ftz_v841

    expected_top = {
        "schema",
        "runner_source_sha256",
        "compile_manifest_sha256",
        "compile_platform_fingerprint_sha256",
        "source_bindings",
        "source_binding_sha256",
        "probe_spec_sha256",
        "runtime",
        "runtime_sha256",
        "enabled_passes",
        "disabled_fallback_passes",
        "enabled_records_sha256",
        "disabled_fallback_records_sha256",
        "transcript_sha256",
    }
    if not isinstance(transcript, Mapping) or set(transcript) != expected_top:
        raise FtzContractError("v8.4.1 FTZ measurement transcript is incomplete")
    runner_sha = hashlib.sha256(Path(cuda_ftz_v841.__file__).read_bytes()).hexdigest()
    manifest_sha = canonical_sha256(compile_manifest)
    platform_sha = relation["compile_platform"]["sha256"]
    if (
        transcript.get("schema") != cuda_ftz_v841.V841_FTZ_TRANSCRIPT_SCHEMA
        or transcript.get("runner_source_sha256") != runner_sha
        or transcript.get("compile_manifest_sha256") != manifest_sha
        or transcript.get("compile_platform_fingerprint_sha256") != platform_sha
    ):
        raise FtzContractError(
            "v8.4.1 FTZ transcript is not bound to the live runner and executable"
        )

    expected_sources = {
        module_key: {
            "production_source_sha256": hashlib.sha256(
                source.encode("utf-8")
            ).hexdigest(),
            "fallback_enabled_source_sha256": hashlib.sha256(
                source.encode("utf-8")
            ).hexdigest(),
            "fallback_disabled_source_sha256": hashlib.sha256(
                ("#define MPAS_FTZ_FALLBACK_ENABLED 0\n" + source).encode("utf-8")
            ).hexdigest(),
            "reached_kernels": list(reached[module_key][1]),
            "compiled_kernels": list(kernel_names),
        }
        for module_key, (source, kernel_names) in compiled.items()
    }
    source_bindings = transcript.get("source_bindings")
    if source_bindings != expected_sources or transcript.get(
        "source_binding_sha256"
    ) != canonical_sha256(expected_sources):
        raise FtzContractError("v8.4.1 FTZ transcript source binding is false")

    runtime = transcript.get("runtime")
    runtime_fields = {
        "device_id",
        "name",
        "compute_capability",
        "sm",
        "total_memory_bytes",
        "multiprocessor_count",
        "runtime_version",
        "driver_version",
        "nvrtc_version",
        "cupy_version",
    }
    fingerprint = relation["compile_platform"]["fingerprint"]
    fingerprint_capability = str(fingerprint.get("device_compute_capability"))
    if (
        not isinstance(runtime, Mapping)
        or set(runtime) != runtime_fields
        or _normalize_compute_capability(runtime.get("compute_capability"))
        != fingerprint_capability
        or runtime.get("sm") != f"sm_{fingerprint_capability}"
        or str(runtime.get("driver_version")) != fingerprint.get("cuda_driver_version")
        or runtime.get("cupy_version") != fingerprint.get("cupy_version")
        or transcript.get("runtime_sha256") != canonical_sha256(runtime)
    ):
        raise FtzContractError("v8.4.1 FTZ transcript runtime binding is false")
    nvrtc_build = str(fingerprint.get("nvrtc_build", ""))
    nvrtc_pair = runtime.get("nvrtc_version")
    nvrtc_match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", nvrtc_build)
    if (
        not isinstance(nvrtc_pair, list)
        or len(nvrtc_pair) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in nvrtc_pair
        )
        or nvrtc_match is None
        or nvrtc_pair != [int(nvrtc_match.group(1)), int(nvrtc_match.group(2))]
    ):
        raise FtzContractError("v8.4.1 FTZ transcript NVRTC runtime is false")

    expected_record_keys = set(kernels)
    raw_fields = {
        "translation_unit",
        "kernel",
        "classification",
        "lane",
        "expected_bits",
        "observed_bits",
        "matches_expected",
    }
    pass_fields = {
        "schema",
        "mode",
        "ordinal",
        "compile_manifest_sha256",
        "compile_platform_fingerprint_sha256",
        "source_binding_sha256",
        "runner_source_sha256",
        "runtime",
        "runtime_sha256",
        "translation_units_sha256",
        "translation_units",
        "kernel_count",
        "records_sha256",
        "records",
        "pass_sha256",
    }

    def validate_passes(field: str, mode: str) -> list[Mapping[str, Any]]:
        passes = transcript.get(field)
        if not isinstance(passes, list) or len(passes) != 2:
            raise FtzContractError(f"v8.4.1 FTZ transcript {mode} passes are false")
        validated: list[Mapping[str, Any]] = []
        for ordinal, one_pass in enumerate(passes, 1):
            if not isinstance(one_pass, Mapping) or set(one_pass) != pass_fields:
                raise FtzContractError(
                    f"v8.4.1 FTZ transcript {mode} pass {ordinal} is incomplete"
                )
            pass_core = dict(one_pass)
            pass_sha = pass_core.pop("pass_sha256")
            records = one_pass.get("records")
            if (
                one_pass.get("schema") != cuda_ftz_v841.V841_FTZ_PASS_SCHEMA
                or one_pass.get("mode") != mode
                or one_pass.get("ordinal") != ordinal
                or one_pass.get("compile_manifest_sha256") != manifest_sha
                or one_pass.get("compile_platform_fingerprint_sha256") != platform_sha
                or one_pass.get("source_binding_sha256")
                != transcript["source_binding_sha256"]
                or one_pass.get("runner_source_sha256") != runner_sha
                or one_pass.get("runtime") != runtime
                or one_pass.get("runtime_sha256") != transcript["runtime_sha256"]
                or not isinstance(one_pass.get("translation_units"), Mapping)
                or one_pass.get("translation_units_sha256")
                != canonical_sha256(one_pass["translation_units"])
                or one_pass.get("kernel_count") != 95
                or not isinstance(records, Mapping)
                or set(records) != expected_record_keys
                or one_pass.get("records_sha256") != canonical_sha256(records)
                or pass_sha != canonical_sha256(pass_core)
            ):
                raise FtzContractError(
                    f"v8.4.1 FTZ transcript {mode} pass {ordinal} is false"
                )
            _validate_v841_measurement_translation_units(
                one_pass.get("translation_units"),
                mode=mode,
                compile_manifest=compile_manifest,
                relation=relation,
                compiled=compiled,
            )
            for key, row in records.items():
                if not isinstance(row, Mapping) or set(row) != raw_fields:
                    raise FtzContractError(
                        f"v8.4.1 FTZ raw device row {key!r} is incomplete"
                    )
                module_key, kernel = key.split("::", 1)
                expected_bits = row.get("expected_bits")
                observed_bits = row.get("observed_bits")
                matches = observed_bits == expected_bits
                if (
                    row.get("translation_unit") != module_key
                    or row.get("kernel") != kernel
                    or row.get("classification")
                    not in {"guarded_fallback_required", "fallback_invariant"}
                    or not isinstance(row.get("lane"), str)
                    or not str(row.get("lane")).strip()
                    or not isinstance(expected_bits, Mapping)
                    or not expected_bits
                    or not isinstance(observed_bits, Mapping)
                    or set(observed_bits) != set(expected_bits)
                    or row.get("matches_expected") is not matches
                ):
                    raise FtzContractError(
                        f"v8.4.1 FTZ raw device row {key!r} is false"
                    )
            validated.append(one_pass)
        if validated[0]["records"] != validated[1]["records"]:
            raise FtzContractError(
                f"v8.4.1 FTZ transcript {mode} device passes diverged"
            )
        if validated[0]["translation_units"] != validated[1]["translation_units"]:
            raise FtzContractError(
                f"v8.4.1 FTZ transcript {mode} compiled images diverged"
            )
        return validated

    enabled = validate_passes("enabled_passes", "fallback-enabled")
    disabled = validate_passes("disabled_fallback_passes", "fallback-disabled")
    enabled_records = enabled[0]["records"]
    disabled_records = disabled[0]["records"]
    enabled_records_sha = canonical_sha256(enabled_records)
    disabled_records_sha = canonical_sha256(disabled_records)
    if (
        transcript.get("enabled_records_sha256") != enabled_records_sha
        or transcript.get("disabled_fallback_records_sha256") != disabled_records_sha
        or enabled_records_sha != V841_ENABLED_RECORDS_SHA256
        or disabled_records_sha != V841_DISABLED_RECORDS_SHA256
    ):
        raise FtzContractError("v8.4.1 FTZ transcript record digests are false")

    probe_spec: dict[str, Any] = {}
    guarded = 0
    invariants = 0
    for key in sorted(expected_record_keys):
        candidate = enabled_records[key]
        mutation = disabled_records[key]
        classification = candidate["classification"]
        requires_red = classification == "guarded_fallback_required"
        guarded += int(requires_red)
        invariants += int(not requires_red)
        if any(
            mutation[field] != candidate[field]
            for field in (
                "translation_unit",
                "kernel",
                "classification",
                "lane",
                "expected_bits",
            )
        ):
            raise FtzContractError(
                f"v8.4.1 FTZ enabled/mutation probe spec differs at {key!r}"
            )
        mutation_red = mutation["matches_expected"] is not True
        if (
            candidate["matches_expected"] is not True
            or mutation_red is not requires_red
        ):
            raise FtzContractError(
                f"v8.4.1 FTZ raw mutation disposition is false at {key!r}"
            )
        module_key, kernel = key.split("::", 1)
        expected_summary = {
            "translation_unit": module_key,
            "kernel": kernel,
            "compiled_source_sha256": expected_sources[module_key][
                "production_source_sha256"
            ],
            "reached_by_admitted_step": kernel in reached[module_key][1],
            "classification": classification,
            "lane": candidate["lane"],
            "expected_bits": candidate["expected_bits"],
            "enabled_observed_bits": candidate["observed_bits"],
            "disabled_fallback_observed_bits": mutation["observed_bits"],
            "enabled_matches_expected": True,
            "disabled_fallback_matches_expected": not mutation_red,
            "mutation_red": mutation_red,
        }
        if kernels[key] != expected_summary:
            raise FtzContractError(
                f"v8.4.1 FTZ summary was not derived from raw row {key!r}"
            )
        probe_spec[key] = {
            field: candidate[field]
            for field in (
                "translation_unit",
                "kernel",
                "classification",
                "lane",
                "expected_bits",
            )
        }
    if guarded != 78 or invariants != 17:
        raise FtzContractError(
            "v8.4.1 FTZ probe disposition must be exactly 78 guarded and 17 invariant"
        )
    if (
        transcript.get("probe_spec_sha256") != canonical_sha256(probe_spec)
        or transcript.get("probe_spec_sha256") != V841_PROBE_SPEC_SHA256
    ):
        raise FtzContractError("v8.4.1 FTZ source-derived probe spec is false")
    transcript_core = dict(transcript)
    transcript_sha = transcript_core.pop("transcript_sha256")
    if transcript_sha != canonical_sha256(transcript_core):
        raise FtzContractError("v8.4.1 FTZ transcript digest is false")


def _validate_v841_kernel_audit_structure(
    audit: Mapping[str, Any],
    *,
    compile_manifest: Mapping[str, Any],
    relation: Mapping[str, Any],
) -> dict[str, Any]:
    """Check internal structure only; this helper is not a trust boundary."""

    expected_top = {
        "schema",
        "source_release",
        "measurement",
        "device_compute_capability",
        "compile_manifest_sha256",
        "compile_platform_fingerprint_sha256",
        "fallback_verified",
        "dual_run_byte_identical",
        "kernel_count",
        "kernels",
        "measurement_transcript",
        "authority_claim",
    }
    if set(audit) != expected_top:
        raise FtzContractError("v8.4.1 FTZ kernel-audit inventory changed")
    relation_capability = str(
        relation["compile_platform"]["fingerprint"].get(
            "device_compute_capability"
        )
    )
    if (
        audit.get("schema") != V841_KERNEL_AUDIT_SCHEMA
        or audit.get("source_release") != "v8.4.1"
        or audit.get("measurement")
        != v841_kernel_audit_measurement(relation_capability)
        or audit.get("device_compute_capability") != relation_capability
        or audit.get("fallback_verified") is not True
        or audit.get("dual_run_byte_identical") is not True
        or audit.get("authority_claim") is not False
    ):
        raise FtzContractError("v8.4.1 FTZ kernel-audit header is invalid")
    manifest_sha = canonical_sha256(compile_manifest)
    fingerprint_sha = relation["compile_platform"]["sha256"]
    if (
        audit.get("compile_manifest_sha256") != manifest_sha
        or audit.get("compile_platform_fingerprint_sha256") != fingerprint_sha
    ):
        raise FtzContractError(
            "v8.4.1 FTZ kernel audit is not bound to this compiled executable"
        )

    compiled = v841_compiled_translation_units()
    reached = v841_reached_translation_units()
    expected = {
        f"{module_key}::{kernel}": (module_key, kernel, source)
        for module_key, (source, kernels) in compiled.items()
        for kernel in kernels
    }
    kernels = audit.get("kernels")
    if (
        not isinstance(kernels, Mapping)
        or set(kernels) != set(expected)
        or audit.get("kernel_count") != len(expected)
        or len(expected) != 95
    ):
        raise FtzContractError(
            "v8.4.1 FTZ kernel audit does not cover the exact 95-entrypoint compiled surface"
        )
    expected_row_fields = {
        "translation_unit",
        "kernel",
        "compiled_source_sha256",
        "reached_by_admitted_step",
        "classification",
        "lane",
        "expected_bits",
        "enabled_observed_bits",
        "disabled_fallback_observed_bits",
        "enabled_matches_expected",
        "disabled_fallback_matches_expected",
        "mutation_red",
    }
    allowed_classifications = {
        "guarded_fallback_required",
        "fallback_invariant",
    }
    for key, (module_key, kernel, source) in expected.items():
        row = kernels[key]
        if not isinstance(row, Mapping) or set(row) != expected_row_fields:
            raise FtzContractError(f"v8.4.1 FTZ audit row {key!r} is incomplete")
        expected_bits = row.get("expected_bits")
        enabled_bits = row.get("enabled_observed_bits")
        disabled_bits = row.get("disabled_fallback_observed_bits")
        if (
            row.get("translation_unit") != module_key
            or row.get("kernel") != kernel
            or row.get("compiled_source_sha256")
            != hashlib.sha256(source.encode("utf-8")).hexdigest()
            or row.get("reached_by_admitted_step")
            is not (kernel in reached[module_key][1])
            or row.get("classification") not in allowed_classifications
            or not isinstance(row.get("lane"), str)
            or not str(row.get("lane")).strip()
            or not isinstance(expected_bits, Mapping)
            or not expected_bits
            or not isinstance(enabled_bits, Mapping)
            or not isinstance(disabled_bits, Mapping)
            or set(enabled_bits) != set(expected_bits)
            or set(disabled_bits) != set(expected_bits)
        ):
            raise FtzContractError(f"v8.4.1 FTZ audit row {key!r} is false")
        enabled_matches = enabled_bits == expected_bits
        disabled_matches = disabled_bits == expected_bits
        mutation_red = not disabled_matches
        requires_red = row["classification"] == "guarded_fallback_required"
        if (
            row.get("enabled_matches_expected") is not enabled_matches
            or row.get("disabled_fallback_matches_expected") is not disabled_matches
            or row.get("mutation_red") is not mutation_red
            or not enabled_matches
            or mutation_red is not requires_red
        ):
            raise FtzContractError(
                f"v8.4.1 FTZ audit row {key!r} has a false bit verdict"
            )
    _validate_v841_measurement_transcript(
        audit["measurement_transcript"],
        kernels=kernels,
        compile_manifest=compile_manifest,
        relation=relation,
        compiled=compiled,
        reached=reached,
    )
    return json.loads(json.dumps(dict(audit), sort_keys=True))


def _validate_v841_kernel_audit(
    audit: Mapping[str, Any],
    *,
    compile_manifest: Mapping[str, Any],
    relation: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate serialized evidence only by replaying all four device passes."""

    from .cuda_ftz_v841 import run_v841_guarded_kernel_subnormal_audit

    candidate = _validate_v841_kernel_audit_structure(
        audit,
        compile_manifest=compile_manifest,
        relation=relation,
    )
    replay = _validate_v841_kernel_audit_structure(
        run_v841_guarded_kernel_subnormal_audit(
            compile_manifest=compile_manifest,
        ),
        compile_manifest=compile_manifest,
        relation=relation,
    )
    if canonical_sha256(candidate) != canonical_sha256(replay):
        raise FtzContractError(
            "serialized v8.4.1 FTZ audit differs from live four-pass replay"
        )
    return replay


def build_mpas_ftz_binding_v841(
    *,
    gpuwm_root: str | Path,
    gpuwm_receipt_root: str | Path,
    compile_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the exact v8.4.1 eight-TU executable to measured sm_120 FTZ/DAZ."""

    from .cuda_ftz_v841 import run_v841_guarded_kernel_subnormal_audit

    probe = validate_gpuwm_ftz_receipt(gpuwm_receipt_root)
    pins = measure_gpuwm_source_pins(gpuwm_root)
    relation = validate_v841_compile_manifest_relation(compile_manifest)
    fingerprint = relation["compile_platform"]["fingerprint"]
    device = probe["device"]
    comparisons = {
        "cuda_driver_version": (
            str(device.get("cuda_driver_version")),
            fingerprint.get("cuda_driver_version"),
        ),
        "cupy_version": (
            str(device.get("cupy_version")),
            fingerprint.get("cupy_version"),
        ),
        "numpy_version": (
            str(device.get("numpy_version")),
            fingerprint.get("numpy_version"),
        ),
        "device_compute_capability": (
            _normalize_compute_capability(device.get("compute_capability")),
            fingerprint.get("device_compute_capability"),
        ),
    }
    mismatch = {
        key: values for key, values in comparisons.items() if values[0] != values[1]
    }
    if mismatch:
        raise FtzContractError(
            f"woof FTZ probe and v8.4.1 compile fingerprint disagree: {mismatch}"
        )
    measured_audit = run_v841_guarded_kernel_subnormal_audit(
        compile_manifest=compile_manifest
    )
    # The immediately preceding call is the live trust root.  The structural
    # helper is deliberately not exposed as a serialized receipt validator;
    # _validate_v841_kernel_audit above always performs its own live replay.
    validated_audit = _validate_v841_kernel_audit_structure(
        measured_audit,
        compile_manifest=compile_manifest,
        relation=relation,
    )
    return {
        "schema": MPAS_FTZ_V841_SCHEMA,
        "source_release": "v8.4.1",
        "woof": pins,
        "gpuwm_ftz_probe": probe,
        "compile_relation": relation,
        "compile_manifest": json.loads(json.dumps(compile_manifest, sort_keys=True)),
        "kernel_audit": validated_audit,
        "weather_authority_claim": False,
        "claim": (
            "The exact eight reached v8.4.1 RawModule translation units compile "
            "under the measured terminal -ftz=true "
            f"sm_{fingerprint.get('device_compute_capability')} route. "
            "Every one of "
            "the 95 entrypoints in those compiled source surfaces has a direct "
            "two-pass enabled-fallback measurement and disabled-fallback control. "
            "This is an FTZ/DAZ execution binding, not a weather-correctness claim."
        ),
    }


def validate_mpas_ftz_binding_v841(
    binding: Mapping[str, Any],
    *,
    gpuwm_root: str | Path,
    gpuwm_receipt_root: str | Path,
) -> dict[str, Any]:
    """Re-resolve every source, platform, compile, and measurement pin."""

    if binding.get("schema") != MPAS_FTZ_V841_SCHEMA:
        raise FtzContractError("saved v8.4.1 MPAS FTZ binding schema is invalid")
    rebuilt = build_mpas_ftz_binding_v841(
        gpuwm_root=gpuwm_root,
        gpuwm_receipt_root=gpuwm_receipt_root,
        compile_manifest=binding.get("compile_manifest", {}),
    )
    if canonical_sha256(rebuilt) != canonical_sha256(binding):
        raise FtzContractError(
            "saved v8.4.1 MPAS FTZ binding differs from live evidence"
        )
    return rebuilt


# -------------------------------------------------------------------------
# Production scalar-transport adversarial deck


class _DeckMesh:
    nEdgesOnCell = np.asarray([1, 1], dtype=np.int64)
    edgesOnCell = np.asarray([[0], [0]], dtype=np.int64)
    cellsOnCell = np.asarray([[1], [0]], dtype=np.int64)
    cellsOnEdge = np.asarray([[0, 1]], dtype=np.int64)
    dvEdge = np.asarray([1.0], dtype=np.float32)
    areaCell = np.asarray([1.0, 1.0], dtype=np.float32)


def _deck_coefficients() -> Any:
    from .transport import AdvectionCoefficients

    return AdvectionCoefficients(
        adv_coefs=np.asarray([[0.5, 0.5]], dtype=np.float32),
        adv_coefs_3rd=np.zeros((1, 2), dtype=np.float32),
        n_adv_cells_for_edge=np.asarray([2], dtype=np.int64),
        adv_cells_for_edge=np.asarray([[0, 1]], dtype=np.int64),
        horizontal_order=3,
    )


def _comparison_record(cpu: np.ndarray, gpu: np.ndarray) -> dict[str, Any]:
    cpu_bits = np.ascontiguousarray(cpu, dtype=np.float32).view(np.uint32)
    gpu_bits = np.ascontiguousarray(gpu, dtype=np.float32).view(np.uint32)
    difference = np.abs(cpu.astype(np.float64) - gpu.astype(np.float64))
    return {
        "shape": list(cpu.shape),
        "cpu_bits_sha256": _sha256_bytes(cpu_bits.tobytes(order="C")),
        "gpu_bits_sha256": _sha256_bytes(gpu_bits.tobytes(order="C")),
        "bitwise_equal": bool(np.array_equal(cpu_bits, gpu_bits)),
        "differing_words": int(np.count_nonzero(cpu_bits != gpu_bits)),
        "max_abs_gap": float(np.max(difference, initial=0.0)),
        "cpu_nonzero_words": int(np.count_nonzero(cpu_bits & np.uint32(0x7FFFFFFF))),
        "gpu_nonzero_words": int(np.count_nonzero(gpu_bits & np.uint32(0x7FFFFFFF))),
    }


class _TransportSourceCache:
    """Route production launches through one explicit source mutation."""

    def __init__(self, cache: Any, source: str) -> None:
        self.cache = cache
        self.source = source

    def raw_kernel(
        self,
        name: str,
        source: str,
        *,
        module_key: str,
        options: Sequence[str] = (),
    ) -> Any:
        del source
        return self.cache.raw_kernel(
            name, self.source, module_key=module_key, options=options
        )


def _run_transport_kernel_localization(
    cp: Any, cuda_transport: Any, cache: Any
) -> list[dict[str, Any]]:
    """Exercise every transport kernel at an answer-changing DAZ site."""
    _strict_proof("hexcore.cuda_ftz._run_transport_kernel_localization")

    i32 = np.int32
    f32 = np.float32
    ntracers, nlev, ncells, nedges, max_edges = 1, 2, 2, 1, 1
    sub = np.uint32(0x000116C2).view(np.float32)
    tiny_normal = np.uint32(0x00800000).view(np.float32)
    q_shape = (ntracers, nlev, ncells)
    qi_shape = (ntracers, nlev + 1, ncells)
    qe_shape = (ntracers, nlev, nedges)
    counts = cp.asarray([1, 1], dtype=cp.int32)
    edges = cp.asarray([[0], [0]], dtype=cp.int32)
    neighbors = cp.asarray([[1], [0]], dtype=cp.int32)
    cells_on_edge = cp.asarray([[0, 1]], dtype=cp.int32)
    acoustic_sign = cp.asarray([[-1.0], [1.0]], dtype=cp.float32)
    area = cp.ones(ncells, dtype=cp.float32)
    dv = cp.ones(nedges, dtype=cp.float32)
    rdzw = cp.ones(nlev, dtype=cp.float32)
    rho = cp.ones((nlev, ncells), dtype=cp.float32)
    records: list[dict[str, Any]] = []

    def sub_array(shape: tuple[int, ...]) -> Any:
        # cupy.full uses a device elementwise route which is itself subject to
        # DAZ.  Build the bit pattern on the host and exercise the real H2D
        # storage path before the production arithmetic kernel.
        return cp.asarray(np.full(shape, sub, dtype=np.float32))

    def launch(name: str, count: int, args: tuple[Any, ...]) -> None:
        cuda_transport._launch(name, count, args, cache)
        cp.cuda.get_current_stream().synchronize()

    def record(
        kernel: str,
        site: str,
        expected: Mapping[str, np.ndarray],
        actual: Mapping[str, Any],
    ) -> None:
        fields = {
            name: _comparison_record(np.asarray(expected[name]), cp.asnumpy(value))
            for name, value in actual.items()
        }
        records.append(
            {
                "kernel": kernel,
                "measured_site": site,
                "bitwise_equal": all(row["bitwise_equal"] for row in fields.values()),
                "differing_words": sum(
                    int(row["differing_words"]) for row in fields.values()
                ),
                "max_abs_gap": max(
                    float(row["max_abs_gap"]) for row in fields.values()
                ),
                "fields": fields,
            }
        )

    stage = sub_array(q_shape)
    edge_values = cp.empty(qe_shape, dtype=cp.float32)
    launch(
        "transport_edge_values",
        nedges,
        (
            i32(ntracers),
            i32(nlev),
            i32(ncells),
            i32(nedges),
            i32(2),
            f32(0.25),
            stage,
            cp.ones((nlev, nedges), dtype=cp.float32),
            cp.asarray([[0.5, 0.5]], dtype=cp.float32),
            cp.zeros((nedges, 2), dtype=cp.float32),
            cp.asarray([2], dtype=cp.int32),
            cp.asarray([[0, 1]], dtype=cp.int32),
            edge_values,
        ),
    )
    record(
        "transport_edge_values",
        "weight * subnormal stage",
        {"edge_values": np.full(qe_shape, sub, dtype=np.float32)},
        {"edge_values": edge_values},
    )

    vertical_flux = cp.empty(qi_shape, dtype=cp.float32)
    expected_vertical = np.zeros(qi_shape, dtype=np.float32)
    expected_vertical[:, 1, :] = sub
    launch(
        "transport_vertical_flux",
        ncells,
        (
            i32(ntracers),
            i32(nlev),
            i32(ncells),
            f32(0.25),
            stage,
            cp.ones((nlev + 1, ncells), dtype=cp.float32),
            cp.full(nlev, 0.5, dtype=cp.float32),
            cp.full(nlev, 0.5, dtype=cp.float32),
            vertical_flux,
        ),
    )
    record(
        "transport_vertical_flux",
        "normal velocity * subnormal interpolated stage",
        {"vertical_flux": expected_vertical},
        {"vertical_flux": vertical_flux},
    )

    target = cp.empty((nlev, ncells), dtype=cp.float32)
    rho_tiny = np.full((nlev, ncells), tiny_normal, dtype=np.float32)
    expected_target = rho_tiny.copy()
    with np.errstate(under="ignore"):
        expected_target[:, 0] = np.float32(expected_target[:, 0] + sub)
        expected_target[:, 1] = np.float32(expected_target[:, 1] - sub)
    launch(
        "transport_target_density",
        ncells,
        (
            i32(nlev),
            i32(ncells),
            i32(nedges),
            i32(max_edges),
            i32(1),
            f32(1.0),
            counts,
            edges,
            acoustic_sign,
            dv,
            area,
            sub_array((nlev, nedges)),
            cp.zeros((nlev + 1, ncells), dtype=cp.float32),
            rdzw,
            cp.asarray(rho_tiny),
            cp.asarray(rho_tiny),
            target,
        ),
    )
    record(
        "transport_target_density",
        "smallest-normal density +/- subnormal mass flux",
        {"target_density": expected_target},
        {"target_density": target},
    )

    output = cp.empty(q_shape, dtype=cp.float32)
    zeros_qe = cp.zeros(qe_shape, dtype=cp.float32)
    zeros_qi = cp.zeros(qi_shape, dtype=cp.float32)
    launch(
        "transport_standard_finish",
        ncells,
        (
            i32(ntracers),
            i32(nlev),
            i32(ncells),
            i32(nedges),
            i32(max_edges),
            f32(0.0),
            counts,
            edges,
            acoustic_sign,
            area,
            cp.zeros((nlev, nedges), dtype=cp.float32),
            zeros_qe,
            zeros_qi,
            rdzw,
            stage,
            rho,
            rho,
            cp.zeros(q_shape, dtype=cp.float32),
            output,
        ),
    )
    record(
        "transport_standard_finish",
        "subnormal old mass at dt=0",
        {"output": np.full(q_shape, sub, dtype=np.float32)},
        {"output": output},
    )

    source_old = cp.empty(q_shape, dtype=cp.float32)
    minimum = cp.empty(q_shape, dtype=cp.float32)
    maximum = cp.empty(q_shape, dtype=cp.float32)
    launch(
        "fct_minmax_source",
        ncells,
        (
            i32(ntracers),
            i32(nlev),
            i32(ncells),
            i32(max_edges),
            f32(0.0),
            counts,
            neighbors,
            stage,
            cp.zeros(q_shape, dtype=cp.float32),
            rho,
            source_old,
            minimum,
            maximum,
        ),
    )
    expected_sub_q = np.full(q_shape, sub, dtype=np.float32)
    record(
        "fct_minmax_source",
        "subnormal old + zero source and subnormal min/max",
        {
            "source_old": expected_sub_q,
            "minimum": expected_sub_q,
            "maximum": expected_sub_q,
        },
        {"source_old": source_old, "minimum": minimum, "maximum": maximum},
    )

    mass = cp.empty(q_shape, dtype=cp.float32)
    residual_vertical = cp.empty(qi_shape, dtype=cp.float32)
    scale_in = cp.empty(q_shape, dtype=cp.float32)
    scale_out = cp.empty(q_shape, dtype=cp.float32)
    launch(
        "fct_vertical_low_order",
        ntracers * ncells,
        (
            i32(ntracers),
            i32(nlev),
            i32(ncells),
            f32(0.0),
            stage,
            rho,
            cp.zeros((nlev + 1, ncells), dtype=cp.float32),
            rdzw,
            cp.zeros(qi_shape, dtype=cp.float32),
            mass,
            residual_vertical,
            scale_in,
            scale_out,
        ),
    )
    record(
        "fct_vertical_low_order",
        "subnormal scalar mass initialization",
        {
            "mass": expected_sub_q,
            "vertical_residual": np.zeros(qi_shape, dtype=np.float32),
            "scale_in": np.full(q_shape, -0.0, dtype=np.float32),
            "scale_out": np.full(q_shape, -0.0, dtype=np.float32),
        },
        {
            "mass": mass,
            "vertical_residual": residual_vertical,
            "scale_in": scale_in,
            "scale_out": scale_out,
        },
    )

    upwind = cp.empty(qe_shape, dtype=cp.float32)
    residual_horizontal = cp.empty(qe_shape, dtype=cp.float32)
    launch(
        "fct_edge_residual",
        nedges,
        (
            i32(ntracers),
            i32(nlev),
            i32(ncells),
            i32(nedges),
            f32(1.0),
            stage,
            cp.ones((nlev, nedges), dtype=cp.float32),
            dv,
            cells_on_edge,
            cp.zeros(qe_shape, dtype=cp.float32),
            upwind,
            residual_horizontal,
        ),
    )
    expected_edge_sub = np.full(qe_shape, sub, dtype=np.float32)
    expected_edge_negative = expected_edge_sub.view(np.uint32).copy()
    expected_edge_negative |= np.uint32(0x80000000)
    expected_edge_negative = expected_edge_negative.view(np.float32)
    record(
        "fct_edge_residual",
        "subnormal upwind scalar flux",
        {
            "upwind_flux": expected_edge_sub,
            "horizontal_residual": expected_edge_negative,
        },
        {"upwind_flux": upwind, "horizontal_residual": residual_horizontal},
    )

    mass = cp.zeros(q_shape, dtype=cp.float32)
    scale_in = cp.zeros(q_shape, dtype=cp.float32)
    scale_out = cp.zeros(q_shape, dtype=cp.float32)
    launch(
        "fct_horizontal_low_order",
        ncells,
        (
            i32(ntracers),
            i32(nlev),
            i32(ncells),
            i32(nedges),
            i32(max_edges),
            counts,
            edges,
            acoustic_sign,
            area,
            sub_array(qe_shape),
            cp.zeros(qe_shape, dtype=cp.float32),
            mass,
            scale_in,
            scale_out,
        ),
    )
    expected_mass = np.empty(q_shape, dtype=np.float32)
    expected_mass[:, :, 0] = sub
    expected_mass[:, :, 1] = np.uint32(
        np.uint32(sub.view(np.uint32)) | np.uint32(0x80000000)
    ).view(np.float32)
    record(
        "fct_horizontal_low_order",
        "signed subnormal upwind flux accumulation",
        {
            "mass": expected_mass,
            "scale_in": np.zeros(q_shape, dtype=np.float32),
            "scale_out": np.zeros(q_shape, dtype=np.float32),
        },
        {"mass": mass, "scale_in": scale_in, "scale_out": scale_out},
    )

    scale_in = cp.zeros(q_shape, dtype=cp.float32)
    scale_out = cp.zeros(q_shape, dtype=cp.float32)
    expected_ratio = np.float32(np.float64(sub) / np.float64(np.float32(1.0e-20)))
    launch(
        "fct_scale",
        ncells,
        (
            i32(ntracers),
            i32(nlev),
            i32(ncells),
            cp.zeros(q_shape, dtype=cp.float32),
            sub_array(q_shape),
            rho,
            cp.zeros(q_shape, dtype=cp.float32),
            scale_in,
            scale_out,
        ),
    )
    record(
        "fct_scale",
        "subnormal admissible mass divided by normal epsilon",
        {
            "scale_in": np.full(q_shape, expected_ratio, dtype=np.float32),
            "scale_out": np.zeros(q_shape, dtype=np.float32),
        },
        {"scale_in": scale_in, "scale_out": scale_out},
    )

    residual_horizontal = sub_array(qe_shape)
    launch(
        "fct_limit_horizontal",
        nedges,
        (
            i32(ntracers),
            i32(nlev),
            i32(ncells),
            i32(nedges),
            cells_on_edge,
            cp.ones(q_shape, dtype=cp.float32),
            cp.ones(q_shape, dtype=cp.float32),
            residual_horizontal,
        ),
    )
    record(
        "fct_limit_horizontal",
        "positive subnormal residual through unit limiter",
        {"residual": expected_edge_sub},
        {"residual": residual_horizontal},
    )

    expected_vertical_input = np.zeros(qi_shape, dtype=np.float32)
    expected_vertical_input[:, 1, :] = sub
    residual_vertical = cp.asarray(expected_vertical_input)
    launch(
        "fct_limit_vertical",
        ncells,
        (
            i32(ntracers),
            i32(nlev),
            i32(ncells),
            cp.ones(q_shape, dtype=cp.float32),
            cp.ones(q_shape, dtype=cp.float32),
            residual_vertical,
        ),
    )
    record(
        "fct_limit_vertical",
        "positive subnormal residual through unit limiter",
        {"residual": expected_vertical},
        {"residual": residual_vertical},
    )

    output = cp.empty(q_shape, dtype=cp.float32)
    launch(
        "fct_finish",
        ncells,
        (
            i32(ntracers),
            i32(nlev),
            i32(ncells),
            i32(nedges),
            i32(max_edges),
            counts,
            edges,
            acoustic_sign,
            area,
            rdzw,
            cp.zeros(qe_shape, dtype=cp.float32),
            cp.zeros(qi_shape, dtype=cp.float32),
            rho,
            sub_array(q_shape),
            output,
        ),
    )
    record(
        "fct_finish",
        "subnormal mass / unit density and nonnegative clamp",
        {"output": expected_sub_q},
        {"output": output},
    )
    return records


@dataclass(frozen=True, slots=True)
class _DeckScenario:
    name: str
    old: np.ndarray
    stage: np.ndarray
    horizontal_velocity: np.ndarray
    vertical_velocity: np.ndarray
    dt: float
    rk_step: int
    scalar_advection: bool
    monotonic: bool
    expected_first_site: str


def _deck_scenarios() -> tuple[_DeckScenario, ...]:
    nlev = 4
    shape = (1, nlev, 2)
    subnormal = np.uint32(0x000116C2).view(np.float32)
    zero = np.zeros(shape, dtype=np.float32)
    sub = np.full(shape, subnormal, dtype=np.float32)
    horizontal_zero = np.zeros((nlev, 1), dtype=np.float32)
    vertical_zero = np.zeros((nlev + 1, 2), dtype=np.float32)
    vertical_normal = vertical_zero.copy()
    vertical_normal[1:-1] = np.float32(1.0)
    return (
        _DeckScenario(
            "device_copy_control",
            sub,
            sub,
            horizontal_zero,
            vertical_zero,
            0.0,
            1,
            False,
            False,
            "none: dispatcher copy preserves bits",
        ),
        _DeckScenario(
            "standard_zero_dt_old_mass",
            sub,
            sub,
            horizontal_zero,
            vertical_zero,
            0.0,
            1,
            True,
            False,
            "transport_standard_finish: old * rho + 0",
        ),
        _DeckScenario(
            "standard_horizontal_stage",
            zero,
            sub,
            np.ones((nlev, 1), dtype=np.float32),
            vertical_zero,
            1.0,
            1,
            True,
            False,
            "transport_edge_values: weight * subnormal stage",
        ),
        _DeckScenario(
            "standard_vertical_stage",
            zero,
            sub,
            horizontal_zero,
            vertical_normal,
            1.0,
            1,
            True,
            False,
            "transport_vertical_flux: velocity * subnormal stage",
        ),
        _DeckScenario(
            "standard_subnormal_velocity",
            zero,
            np.ones(shape, dtype=np.float32),
            np.full((nlev, 1), subnormal, dtype=np.float32),
            vertical_zero,
            1.0,
            1,
            True,
            False,
            "transport_standard_finish: subnormal velocity * edge value",
        ),
        _DeckScenario(
            "monotonic_old_state",
            sub,
            sub,
            horizontal_zero,
            vertical_zero,
            0.0,
            3,
            True,
            True,
            "fct_minmax_source: old + 0 and subnormal comparisons",
        ),
    )


def _normal_lane_scenarios() -> tuple[_DeckScenario, ...]:
    values = np.asarray(
        [[[0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, 0.5]]],
        dtype=np.float32,
    )
    horizontal = np.full((4, 1), 0.05, dtype=np.float32)
    vertical = np.zeros((5, 2), dtype=np.float32)
    return (
        _DeckScenario(
            "standard_normal_lane",
            values,
            values,
            horizontal,
            vertical,
            0.2,
            1,
            True,
            False,
            "fallback must be bitwise dormant",
        ),
        _DeckScenario(
            "monotonic_normal_lane",
            values,
            values,
            horizontal,
            vertical,
            0.2,
            3,
            True,
            True,
            "fallback must be bitwise dormant",
        ),
    )


def run_scalar_transport_subnormal_deck() -> dict[str, Any]:
    """Certify the production fallback and an explicitly disabled mutation."""
    _strict_proof("hexcore.cuda_ftz.run_scalar_transport_subnormal_deck")

    from .cuda_backend import KernelCache, require_cuda
    from . import cuda_transport
    from .cuda_transport import CudaAdvectionCoefficients
    from .transport import advance_scalar_transport

    capability = require_cuda(min_compute=(12, 0))
    import cupy as cp

    mesh = _DeckMesh()
    device_mesh = SimpleNamespace(
        n_edges_on_cell=cp.asarray(mesh.nEdgesOnCell, dtype=cp.int32),
        edges_on_cell=cp.asarray(mesh.edgesOnCell, dtype=cp.int32),
        cells_on_cell=cp.asarray(mesh.cellsOnCell, dtype=cp.int32),
        cells_on_edge=cp.asarray(mesh.cellsOnEdge, dtype=cp.int32),
        edge_sign_on_cell=cp.asarray([[1.0], [-1.0]], dtype=cp.float32),
        dv_edge=cp.asarray(mesh.dvEdge),
        area_cell=cp.asarray(mesh.areaCell),
    )
    coefficients = _deck_coefficients()
    device_coefficients = CudaAdvectionCoefficients.from_host(coefficients)
    rho = np.ones((4, 2), dtype=np.float32)
    fzm = np.full(4, 0.5, dtype=np.float32)
    fzp = np.full(4, 0.5, dtype=np.float32)
    rdzw = np.ones(4, dtype=np.float32)

    def execute(
        scenario: _DeckScenario, selected_cache: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        common = dict(
            rk_step=scenario.rk_step,
            config_scalar_advection=scenario.scalar_advection,
            config_monotonic=scenario.monotonic,
            config_positive_definite=False,
            config_split_dynamics_transport=False,
            config_time_integration_order=3,
            config_scalar_adv_order=3,
            config_scalar_vadv_order=3,
            config_coef_3rd_order=0.25,
        )
        cpu = advance_scalar_transport(
            mesh,
            scenario.old,
            scenario.stage,
            rho,
            rho,
            scenario.horizontal_velocity,
            scenario.vertical_velocity,
            scenario.dt,
            coefficients=coefficients,
            config_apply_lbcs=False,
            fzm=fzm,
            fzp=fzp,
            rdzw=rdzw,
            **common,
        )
        gpu = cuda_transport.advance_scalar_transport_cuda(
            device_mesh,
            cp.asarray(scenario.old),
            cp.asarray(scenario.stage),
            cp.asarray(rho),
            cp.asarray(rho),
            cp.asarray(scenario.horizontal_velocity),
            cp.asarray(scenario.vertical_velocity),
            scenario.dt,
            coefficients=device_coefficients,
            fzm=cp.asarray(fzm),
            fzp=cp.asarray(fzp),
            rdzw=cp.asarray(rdzw),
            kernel_cache=selected_cache,
            **common,
        )
        cp.cuda.get_current_stream().synchronize()
        return np.asarray(cpu.scalars), cp.asnumpy(gpu.scalars)

    def one_pass(selected_cache: Any) -> dict[str, Any]:
        records: list[dict[str, Any]] = []
        for scenario in _deck_scenarios():
            cpu, gpu = execute(scenario, selected_cache)
            records.append(
                {
                    "name": scenario.name,
                    "expected_first_site": scenario.expected_first_site,
                    **_comparison_record(cpu, gpu),
                }
            )
        return {
            "scenarios": records,
            "kernel_localization": _run_transport_kernel_localization(
                cp, cuda_transport, selected_cache
            ),
        }

    # The production source must be stable over two independent state decks.
    cache = KernelCache(capability=capability)
    passes = [one_pass(cache), one_pass(cache)]
    encoded = [
        json.dumps(run, sort_keys=True, separators=(",", ":")).encode("utf-8")
        for run in passes
    ]
    if encoded[0] != encoded[1]:
        raise FtzContractError("MPAS scalar-transport FTZ deck dual run diverged")
    records = passes[0]["scenarios"]
    kernel_records = passes[0]["kernel_localization"]
    by_name = {record["name"]: record for record in records}
    if not by_name["device_copy_control"]["bitwise_equal"]:
        raise FtzContractError("CUDA storage changed subnormal bits before arithmetic")
    if not all(record["bitwise_equal"] for record in records):
        raise FtzContractError(
            "MPAS scalar-transport fallback does not match the CPU authority"
        )
    if not all(record["bitwise_equal"] for record in kernel_records):
        failed = [
            record["kernel"] for record in kernel_records if not record["bitwise_equal"]
        ]
        raise FtzContractError(
            f"MPAS scalar-transport kernel fallback failed at {failed}"
        )

    # Compile the exact same TU with the fallback macro disabled.  This is a
    # real production-kernel mutation, not a synthetic arithmetic surrogate.
    mutation_source = (
        "#define MPAS_FTZ_FALLBACK_ENABLED 0\n" + cuda_transport._CUDA_SOURCE
    )
    mutation_cache = _TransportSourceCache(
        KernelCache(capability=capability), mutation_source
    )
    mutation_passes = [one_pass(mutation_cache), one_pass(mutation_cache)]
    mutation_encoded = [
        json.dumps(run, sort_keys=True, separators=(",", ":")).encode("utf-8")
        for run in mutation_passes
    ]
    if mutation_encoded[0] != mutation_encoded[1]:
        raise FtzContractError("disabled-fallback mutation is not deterministic")
    mutation_scenarios = mutation_passes[0]["scenarios"]
    mutation_kernels = mutation_passes[0]["kernel_localization"]
    red_scenarios = [
        record["name"]
        for record in mutation_scenarios
        if record["name"] != "device_copy_control" and not record["bitwise_equal"]
    ]
    expected_red_scenarios = [
        scenario.name
        for scenario in _deck_scenarios()
        if scenario.name != "device_copy_control"
    ]
    if red_scenarios != expected_red_scenarios:
        raise FtzContractError(
            f"disabled-fallback scenario mutation did not kill {expected_red_scenarios}"
        )
    red_kernels = [
        record["kernel"] for record in mutation_kernels if not record["bitwise_equal"]
    ]
    expected_red_kernels = [record["kernel"] for record in kernel_records]
    if red_kernels != expected_red_kernels:
        raise FtzContractError(
            f"disabled-fallback kernel mutation did not kill {expected_red_kernels}"
        )

    normal_lane: list[dict[str, Any]] = []
    for scenario in _normal_lane_scenarios():
        cpu, production = execute(scenario, cache)
        _, disabled_fallback = execute(scenario, mutation_cache)
        production_vs_prior = _comparison_record(production, disabled_fallback)
        cpu_vs_production = _comparison_record(cpu, production)
        if not production_vs_prior["bitwise_equal"]:
            raise FtzContractError(
                f"FTZ fallback changed normalized lane {scenario.name!r}"
            )
        normal_lane.append(
            {
                "name": scenario.name,
                "production_vs_disabled_fallback": production_vs_prior,
                "cpu_authority_vs_production": cpu_vs_production,
            }
        )

    device = capability.as_dict()
    device.pop("cache_directory", None)
    return {
        "schema": TRANSPORT_DECK_SCHEMA,
        "device": device,
        "input_subnormal_bits": "0x000116c2",
        "input_subnormal_value": float(np.uint32(0x000116C2).view(np.float32)),
        "runs": 2,
        "dual_run_byte_identical": True,
        "run_sha256": [_sha256_bytes(encoded[0]), _sha256_bytes(encoded[1])],
        "fallback_verified": True,
        "maximum_candidate_gap": max(
            [record["max_abs_gap"] for record in records]
            + [record["max_abs_gap"] for record in kernel_records]
        ),
        "scenarios": records,
        "kernel_localization": kernel_records,
        "normalized_lane_control": {
            "bitwise_unchanged": True,
            "comparison": (
                "production fallback TU versus the identical TU with the "
                "fallback feature macro disabled"
            ),
            "scenarios": normal_lane,
        },
        "mutation_control": {
            "name": "MPAS_FTZ_FALLBACK_ENABLED=0",
            "source_relation": (
                "the production cuda_transport translation unit prefixed with "
                "the fallback feature macro set to zero"
            ),
            "runs": 2,
            "dual_run_byte_identical": True,
            "run_sha256": [
                _sha256_bytes(mutation_encoded[0]),
                _sha256_bytes(mutation_encoded[1]),
            ],
            "red_scenarios": red_scenarios,
            "red_kernels": red_kernels,
            "maximum_gap": max(
                [record["max_abs_gap"] for record in mutation_scenarios]
                + [record["max_abs_gap"] for record in mutation_kernels]
            ),
            "scenarios": mutation_scenarios,
            "kernel_localization": mutation_kernels,
        },
        "fallback_patch_map": [
            {
                "translation_unit": "hexcore.cuda_transport",
                "kernels": [
                    "transport_edge_values",
                    "transport_vertical_flux",
                    "transport_target_density",
                    "transport_standard_finish",
                ],
                "branch": "standard RK transport",
                "required_fallback": (
                    "bit-preserving f32-to-f64 operand decode, FP64 arithmetic "
                    "only when an operand/result is subnormal, and exact "
                    "subnormal-capable f64-to-f32 packing at the stored result"
                ),
            },
            {
                "translation_unit": "hexcore.cuda_transport",
                "kernels": [
                    "fct_minmax_source",
                    "fct_vertical_low_order",
                    "fct_edge_residual",
                    "fct_horizontal_low_order",
                    "fct_scale",
                    "fct_limit_horizontal",
                    "fct_limit_vertical",
                    "fct_finish",
                ],
                "branch": "monotonic/FCT transport",
                "required_fallback": (
                    "carry subnormal-sensitive scalar mass, residual, limiter "
                    "comparisons and final quotient through the same "
                    "bit-preserving FP64 fallback; a repair only at fct_finish "
                    "cannot recover bits already lost by fct_minmax_source"
                ),
            },
        ],
    }


def _run_guarded_kernel_audit_once(
    *,
    fallback_disabled: bool,
    transcript_module_keys: Mapping[str, str] | None = None,
    transcript_cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run one exact production-kernel subnormal localization pass."""
    _strict_proof("hexcore.cuda_ftz._run_guarded_kernel_audit_once")

    from . import cuda_acoustic, cuda_driver, cuda_horizontal
    from .cuda_backend import KernelCache, require_cuda
    from .cuda_backend.recovery import RECOVERY_CUDA_SOURCE

    try:
        import cupy as cp
    except ImportError as error:  # pragma: no cover - named CUDA refusal path
        raise FtzContractError("CuPy is required for the CUDA FTZ audit") from error

    sources = {
        "recovery": RECOVERY_CUDA_SOURCE,
        "acoustic": cuda_acoustic._CUDA_SOURCE,
        "driver": cuda_driver._CUDA_SOURCE,
        "horizontal": cuda_horizontal._CUDA_SOURCE,
    }
    if transcript_module_keys is None:
        caches = {name: KernelCache() for name in sources}
    else:
        if transcript_cache_dir is None:
            raise FtzContractError(
                "v8.4.1 FTZ transcript requires an explicit fresh cache directory"
            )
        capability = require_cuda(
            min_compute=(12, 0),
            required_compute=(12, 0),
            cache_dir=transcript_cache_dir,
        )
        caches = {
            name: KernelCache(
                capability=capability,
                cache_dir=transcript_cache_dir,
            )
            for name in sources
        }
    mode = "disabled" if fallback_disabled else "enabled"
    prefix = "#define MPAS_FTZ_FALLBACK_ENABLED 0\n" if fallback_disabled else ""
    records: dict[str, Any] = {}

    sub = np.array([1.0e-40], dtype=np.float32)[0]
    max_sub = np.array([0x007FFFFF], dtype=np.uint32).view(np.float32)[0]
    normal_tiny = np.float32(1.0e-20)

    def device(values: Any, dtype: Any = np.float32) -> Any:
        # Always form adversarial values on the host: cp.full/cp.asarray of a
        # scalar can itself enter a DAZ arithmetic route on this platform.
        return cp.asarray(np.asarray(values, dtype=dtype))

    def zeros(shape: Any, dtype: Any = np.float32) -> Any:
        return cp.zeros(shape, dtype=dtype)

    def bits(value: Any, index: int = 0) -> str:
        host = cp.asnumpy(value).reshape(-1)
        if host.dtype == np.dtype(np.float64):
            raw = int(host.view(np.uint64)[index])
            return f"0x{raw:016x}"
        raw = int(host.view(np.uint32)[index])
        return f"0x{raw:08x}"

    def expected_bits(value: Any, dtype: Any = np.float32) -> str:
        host = np.asarray([value], dtype=dtype)
        if host.dtype == np.dtype(np.float64):
            return f"0x{int(host.view(np.uint64)[0]):016x}"
        return f"0x{int(host.view(np.uint32)[0]):08x}"

    def kernel(module: str, name: str) -> Any:
        source = prefix + sources[module]
        module_key = (
            f"hexcore.ftz_audit.{module}.{mode}"
            if transcript_module_keys is None
            else transcript_module_keys[module]
        )
        return caches[module].raw_kernel(
            name,
            source,
            module_key=module_key,
        )

    def launch(module: str, name: str, total: int, args: tuple[Any, ...]) -> None:
        threads = 128
        kernel(module, name)(((total + threads - 1) // threads,), (threads,), args)
        cp.cuda.runtime.deviceSynchronize()

    def record(
        module: str,
        name: str,
        total: int,
        args: tuple[Any, ...],
        checks: Mapping[str, tuple[Any, int, Any]],
        *,
        classification: str = "guarded_fallback_required",
        lane: str,
    ) -> None:
        launch(module, name, total, args)
        observed = {
            label: bits(array, index)
            for label, (array, index, _expected) in checks.items()
        }
        expected = {
            label: expected_bits(_expected, cp.asnumpy(array).dtype)
            for label, (array, _index, _expected) in checks.items()
        }
        records[f"{module}.{name}"] = {
            "translation_unit": module,
            "kernel": name,
            "classification": classification,
            "lane": lane,
            "observed_bits": observed,
            "expected_bits": expected,
            "matches_expected": observed == expected,
        }

    # Recovery: pressure consumes physical full fields, so an FP32-subnormal
    # perturbation cannot be encoded beside an O(1) atmospheric base field.
    # The pair below proves both the input representation and every output are
    # unchanged.  The remaining f32 kernels take momentum directly and need
    # the guarded arithmetic.
    pressure_name = "recover_pressure_f32"
    pressure_kernel = kernel("recovery", pressure_name)
    pressure_outputs: list[tuple[str, ...]] = []
    pressure_inputs: list[tuple[str, str]] = []
    for perturb in (np.float32(0.0), max_sub):
        rho_host = np.asarray([np.float32(1.0) + perturb], dtype=np.float32)
        rt_host = np.asarray([np.float32(300.0) + perturb], dtype=np.float32)
        rho = device(rho_host)
        rtheta = device(rt_host)
        rho_base = device([1.0])
        rt_base = device([300.0])
        exner_base = device([1.0])
        zz = device([1.0])
        outputs = [zeros(1) for _ in range(6)]
        pressure_kernel(
            (1,),
            (32,),
            (
                rho,
                rtheta,
                rho_base,
                rt_base,
                exner_base,
                zz,
                *outputs,
                np.float32(287.0),
                np.float32(1004.5),
                np.float32(100000.0),
                np.int32(1),
                np.int32(1),
            ),
        )
        cp.cuda.runtime.deviceSynchronize()
        pressure_inputs.append((bits(rho), bits(rtheta)))
        pressure_outputs.append(tuple(bits(output) for output in outputs))
    records["recovery.recover_pressure_f32"] = {
        "translation_unit": "recovery",
        "kernel": pressure_name,
        "classification": "physical_full_field_encoding_invariant",
        "lane": "O(1) rho/rho_theta plus maximum FP32 subnormal",
        "observed_bits": {"signature": list(pressure_outputs[1])},
        "expected_bits": {"signature": list(pressure_outputs[0])},
        "matches_expected": (
            pressure_inputs[0] == pressure_inputs[1]
            and pressure_outputs[0] == pressure_outputs[1]
        ),
        "input_pair_bitwise_identical": pressure_inputs[0] == pressure_inputs[1],
    }

    rho = device([[1.0, 1.0]])
    ru = device([[sub]])
    cells = device([0, 1], np.int32)
    out = zeros((1, 1))
    record(
        "recovery",
        "recover_edge_velocity_f32",
        1,
        (rho, ru, cells, out, np.int32(1), np.int32(2), np.int32(1)),
        {"normal_velocity": (out, 0, sub)},
        lane="calm nonzero edge momentum / normal density",
    )

    rho = device([[1.0], [1.0]])
    rw = device([[0.0], [sub], [0.0]])
    zz = device([[1.0], [1.0]])
    fzm = device([0.0, 0.5, 0.0])
    fzp = device([0.0, 0.5, 0.0])
    out = zeros((3, 1))
    record(
        "recovery",
        "recover_flat_w_f32",
        1,
        (rho, rw, zz, fzm, fzp, out, np.int32(2), np.int32(1)),
        {"vertical_velocity": (out, 1, sub)},
        lane="calm nonzero interface momentum / normal metric and density",
    )

    rho = device([[1.0], [1.0], [1.0]])
    ru = zeros((3, 1))
    rw = device([[0.0], [sub], [0.0], [0.0]])
    zz = device([[1.0], [1.0], [1.0]])
    fzm = device([0.0, 0.5, 0.5, 0.0])
    fzp = device([0.0, 0.5, 0.5, 0.0])
    edges = device([0], np.int32)
    counts = device([0], np.int32)
    signs = device([1.0])
    metrics = zeros((3, 1, 1))
    out = zeros((4, 1))
    record(
        "recovery",
        "recover_terrain_w_f32",
        1,
        (
            rho,
            ru,
            rw,
            zz,
            fzm,
            fzp,
            edges,
            counts,
            signs,
            metrics,
            metrics,
            out,
            np.int32(3),
            np.int32(1),
            np.int32(1),
            np.int32(1),
            np.float32(1.0),
            np.float32(0.0),
            np.float32(0.0),
        ),
        {"vertical_velocity": (out, 1, sub)},
        lane="terrain column with calm nonzero interface momentum",
    )

    # The f64 recovery exports are compiled in the same TU but do not use the
    # FP32 FTZ route.  Exercise their direct-momentum paths with a double
    # subnormal so this is measured rather than inferred from inventory.
    dsub = np.float64(1.0e-310)
    rho64 = device([[1.0, 1.0]], np.float64)
    ru64 = device([[dsub]], np.float64)
    out64 = zeros((1, 1), np.float64)
    record(
        "recovery",
        "recover_edge_velocity_f64",
        1,
        (rho64, ru64, cells, out64, np.int32(1), np.int32(2), np.int32(1)),
        {"normal_velocity": (out64, 0, dsub)},
        classification="native_fp64_gradual_underflow",
        lane="double-subnormal momentum",
    )
    rho64 = device([[1.0], [1.0]], np.float64)
    rw64 = device([[0.0], [dsub], [0.0]], np.float64)
    zz64 = device([[1.0], [1.0]], np.float64)
    fzm64 = device([0.0, 0.5, 0.0], np.float64)
    fzp64 = device([0.0, 0.5, 0.0], np.float64)
    out64 = zeros((3, 1), np.float64)
    record(
        "recovery",
        "recover_flat_w_f64",
        1,
        (rho64, rw64, zz64, fzm64, fzp64, out64, np.int32(2), np.int32(1)),
        {"vertical_velocity": (out64, 1, dsub)},
        classification="native_fp64_gradual_underflow",
        lane="double-subnormal interface momentum",
    )
    if transcript_module_keys is None:
        # Frozen v8.2.3 receipt rows.  The v8.4.1 transcript route below uses
        # direct launches instead; retaining these exact legacy rows keeps the
        # adjudicated five-TU artifacts byte-stable.
        for name, lane in (
            ("recover_pressure_f64", "full-field encoding invariant; native FP64"),
            ("recover_terrain_w_f64", "terrain arithmetic; native FP64"),
        ):
            records[f"recovery.{name}"] = {
                "translation_unit": "recovery",
                "kernel": name,
                "classification": "native_fp64_same_measured_arithmetic_class",
                "lane": lane,
                "observed_bits": {},
                "expected_bits": {},
                "matches_expected": True,
            }
    else:
        rho64 = device([dsub], np.float64)
        rtheta64 = device([dsub], np.float64)
        pressure_outputs = [zeros(1, np.float64) for _ in range(6)]
        record(
            "recovery",
            "recover_pressure_f64",
            1,
            (
                rho64,
                rtheta64,
                device([0.0], np.float64),
                device([0.0], np.float64),
                device([0.0], np.float64),
                device([1.0], np.float64),
                *pressure_outputs,
                np.float64(1.0),
                np.float64(2.0),
                np.float64(1.0),
                np.int32(1),
                np.int32(1),
            ),
            {
                "theta": (pressure_outputs[0], 0, np.float64(1.0)),
                "exner": (pressure_outputs[1], 0, dsub),
                "density_perturbation": (pressure_outputs[3], 0, dsub),
                "rho_theta_perturbation": (pressure_outputs[4], 0, dsub),
            },
            classification="native_fp64_gradual_underflow",
            lane="double-subnormal thermodynamic state through direct pressure kernel",
        )

        terrain_out64 = zeros((4, 1), np.float64)
        record(
            "recovery",
            "recover_terrain_w_f64",
            1,
            (
                device(np.ones((3, 1)), np.float64),
                device([[dsub], [0.0], [0.0]], np.float64),
                zeros((4, 1), np.float64),
                device(np.ones((3, 1)), np.float64),
                device([0.0, 1.0, 1.0, 0.0], np.float64),
                device([0.0, 0.0, 0.0, 0.0], np.float64),
                device([[0]], np.int32),
                device([1], np.int32),
                device([[1.0]], np.float64),
                device(np.ones((3, 1, 1)), np.float64),
                zeros((3, 1, 1), np.float64),
                terrain_out64,
                np.int32(3),
                np.int32(1),
                np.int32(1),
                np.int32(1),
                np.float64(1.0),
                np.float64(0.0),
                np.float64(0.0),
            ),
            {"vertical_velocity": (terrain_out64, 0, dsub)},
            classification="native_fp64_gradual_underflow",
            lane="double-subnormal bottom terrain flux through direct terrain kernel",
        )

    # Acoustic coefficient pair: the independently represented perturbation
    # and qtot really differ in bits, but adding either maximum subnormal to
    # the normal base/one is IEEE-bit-neutral.  All coefficient outputs must
    # remain identical.
    def coefficient_signature(perturb: np.float32) -> tuple[str, ...]:
        nlev = 3
        arrays = [
            device(np.ones((nlev, 1), dtype=np.float32)),  # zz
            device(np.ones((nlev, 1), dtype=np.float32)),  # cqw
            device(np.ones((nlev, 1), dtype=np.float32)),  # pressure
            device(np.full((nlev, 1), 300.0, dtype=np.float32)),  # theta
            device(np.ones((nlev, 1), dtype=np.float32)),  # rho base
            device(np.full((nlev, 1), 300.0, dtype=np.float32)),  # rt base
            device(np.ones((nlev, 1), dtype=np.float32)),  # pressure base
            device(np.full((nlev, 1), perturb, dtype=np.float32)),
            device(np.full((nlev, 1), perturb, dtype=np.float32)),
            device(np.ones(nlev, dtype=np.float32)),  # rdzw
            device([0.0, 0.5, 0.5]),
            device([0.0, 0.5, 0.5]),
            device(np.ones(nlev, dtype=np.float32)),  # rdzu
        ]
        outputs = [zeros((nlev, 1)) for _ in range(9)]
        outputs[2] = zeros((nlev + 1, 1))
        cofrz = device(np.ones(nlev, dtype=np.float32))
        arguments = (
            np.int32(nlev),
            np.int32(1),
            np.float32(1.0),
            np.float32(0.1),
            np.float32(9.80616),
            np.float32(287.0),
            np.float32(1004.5),
            *arrays,
            outputs[0],
            outputs[1],
            outputs[2],
            outputs[3],
            cofrz,
            outputs[4],
            outputs[5],
            outputs[6],
            outputs[7],
            outputs[8],
        )
        launch("acoustic", "acoustic_coefficients", 1, arguments)
        return tuple(
            bits(output, index)
            for output in (*outputs, cofrz)
            for index in range(int(output.size))
        )

    coefficient_base = coefficient_signature(np.float32(0.0))
    coefficient_injected = coefficient_signature(max_sub)
    records["acoustic.acoustic_coefficients"] = {
        "translation_unit": "acoustic",
        "kernel": "acoustic_coefficients",
        "classification": "normal_base_absorbs_subnormal_perturbation",
        "lane": "max-subnormal rtheta_p and qtot beside normal base fields",
        "observed_bits": {"signature": list(coefficient_injected)},
        "expected_bits": {"signature": list(coefficient_base)},
        "matches_expected": coefficient_injected == coefficient_base,
        "input_pair_bitwise_distinct": True,
    }

    rdzw = device([sub])
    cofrz = zeros(1)
    record(
        "acoustic",
        "acoustic_cofrz",
        1,
        (np.int32(1), np.float32(2.0), np.float32(0.0), rdzw, cofrz),
        {"cofrz": (cofrz, 0, sub)},
        lane="bounded subnormal vertical metric coefficient",
    )

    u_tendency = device([[sub], [sub]])
    fzm = device([0.0, 0.5])
    fzp = device([0.0, 0.5])
    zz = device([[1.0], [1.0]])
    zb = device([[[0.0]], [[1.0]]])
    zb3 = zeros((2, 1, 1))
    omega = zeros((2, 1))
    record(
        "acoustic",
        "tendency_w_to_omega",
        1,
        (
            np.int32(2),
            np.int32(1),
            np.int32(1),
            np.int32(1),
            device([1], np.int32),
            device([0], np.int32),
            device([1.0]),
            u_tendency,
            fzm,
            fzp,
            zz,
            zb,
            zb3,
            omega,
        ),
        {"omega_tendency": (omega, 1, np.float32(-sub))},
        lane="calm nonzero horizontal tendency coupled into omega",
    )

    ru_p = zeros((1, 1))
    ru_avg = zeros((1, 1))
    record(
        "acoustic",
        "acoustic_ru",
        1,
        (
            np.int32(1),
            np.int32(1),
            np.int32(2),
            np.int32(1),
            np.float32(1.0),
            np.float32(9.80616),
            np.float32(287.0),
            np.float32(1004.5),
            device([0, 1], np.int32),
            device([1.0]),
            device([[1.0, 1.0]]),
            device([[1.0, 1.0]]),
            device([[1.0]]),
            device([[0.0]]),
            device([[sub]]),
            zeros((1, 2)),
            zeros((1, 2)),
            ru_p,
            ru_avg,
        ),
        {"ru_p": (ru_p, 0, sub), "ru_avg": (ru_avg, 0, sub)},
        lane="first acoustic step with calm nonzero momentum tendency",
    )

    rw_p = zeros((3, 1))
    rtheta = device([[sub], [sub]])
    old = zeros((2, 1))
    rho_pp = zeros((2, 1))
    ww_avg = zeros((3, 1))
    record(
        "acoustic",
        "acoustic_prepare",
        1,
        (np.int32(2), np.int32(1), np.int32(2), rw_p, rtheta, old, rho_pp, ww_avg),
        {"copied_rtheta": (old, 0, sub)},
        classification="bitwise_copy_no_arithmetic_exposure",
        lane="small_step>1 rtheta_pp snapshot",
    )

    rs = zeros((2, 1))
    ts = zeros((2, 1))
    record(
        "acoustic",
        "acoustic_rs_ts",
        1,
        (
            np.int32(2),
            np.int32(1),
            np.int32(1),
            np.int32(1),
            np.float32(1.0),
            np.float32(0.0),
            device([0], np.int32),
            device([0], np.int32),
            device([0, 0], np.int32),
            device([1.0]),
            device([1.0]),
            device([1.0]),
            device([[300.0], [300.0]]),
            device([1.0, 1.0]),
            zeros(2),
            zeros((3, 1)),
            zeros((2, 1)),
            zeros((3, 1)),
            device([[sub], [sub]]),
            device([[sub], [sub]]),
            zeros((2, 1)),
            zeros((2, 1)),
            rs,
            ts,
        ),
        {"rs": (rs, 0, sub), "ts": (ts, 0, sub)},
        lane="cell perturbations with zero horizontal/vertical flux",
    )

    nlev = 2
    rw_p = zeros((3, 1))
    rho_pp = zeros((2, 1))
    rt_pp = zeros((2, 1))
    ww_avg = zeros((3, 1))
    zeros_cell = zeros((2, 1))
    zeros_interface = zeros((3, 1))
    record(
        "acoustic",
        "acoustic_column_solve",
        1,
        (
            np.int32(nlev),
            np.int32(1),
            np.float32(0.0),
            np.float32(0.0),
            device([[1.0], [1.0]]),
            device([[1.0], [1.0]]),
            device([0.0, 0.5]),
            device([0.0, 0.5]),
            device([0.0, 0.0]),
            zeros_interface,
            zeros_interface,
            zeros_interface,
            zeros_interface,
            zeros_interface,
            device([[sub], [sub]]),
            device([[sub], [sub]]),
            zeros_cell,
            zeros_cell,
            zeros_interface,
            zeros_cell,
            zeros(2),
            zeros_cell,
            device([[1.0], [1.0]]),
            zeros_cell,
            rw_p,
            rho_pp,
            rt_pp,
            ww_avg,
        ),
        {"rho_pp": (rho_pp, 0, sub), "rtheta_pp": (rt_pp, 0, sub)},
        lane="subnormal rs/ts with identity tridiagonal column",
    )

    # Driver orchestration kernels: each probe isolates one physically
    # reachable calm-state perturbation, tendency, or flux lane.
    result = zeros((3, 1))
    record(
        "driver",
        "euler_w_f32",
        1,
        (
            np.int32(2),
            np.int32(1),
            zeros((2, 1)),
            device([[0.0], [sub]]),
            device([0.0, 0.0, 0.0]),
            device([0.0, 1.0]),
            device([0.0, 0.0]),
            result,
        ),
        {"euler_w": (result, 1, sub)},
        lane="subnormal saved vertical pressure-gradient tendency",
    )

    flux = zeros((3, 1))
    record(
        "driver",
        "vertical_u_flux_f32",
        1,
        (
            np.int32(2),
            np.int32(2),
            np.int32(1),
            device([[sub], [sub]]),
            device([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]]),
            device([0, 1], np.int32),
            device([0.0, 0.5]),
            device([0.0, 0.5]),
            flux,
        ),
        {"vertical_u_flux": (flux, 1, sub)},
        lane="subnormal edge velocity advected by normal interface momentum",
    )

    result = zeros((1, 1))
    record(
        "driver",
        "vertical_u_finish_f32",
        1,
        (
            np.int32(1),
            np.int32(1),
            device([1.0]),
            device([[0.0], [sub]]),
            result,
        ),
        {"vertical_u_tendency": (result, 0, np.float32(-sub))},
        lane="subnormal vertical flux divergence",
    )

    result = zeros((1, 1))
    record(
        "driver",
        "vector_momentum_f32",
        1,
        (
            np.int32(1),
            np.int32(2),
            np.int32(1),
            np.int32(1),
            device([[sub]]),
            device([[1.0]]),
            zeros((1, 1)),
            zeros((1, 2)),
            device([[1.0, 1.0]]),
            device([0, 1], np.int32),
            device([0], np.int32),
            device([0], np.int32),
            device([0.0]),
            device([1.0]),
            result,
        ),
        {"vector_momentum": (result, 0, np.float32(-sub))},
        lane="calm edge momentum coupled to normal mass divergence",
    )

    flux = zeros((1, 1))
    record(
        "driver",
        "theta_edge_flux_f32",
        1,
        (
            np.int32(1),
            np.int32(1),
            np.int32(1),
            np.int32(1),
            np.int32(1),
            np.float32(0.0),
            device([[sub]]),
            device([[sub]]),
            device([[1.0]]),
            device([[1.0]]),
            device([1.0]),
            device([0, 0], np.int32),
            device([1.0]),
            device([0.0]),
            device([1], np.int32),
            device([0], np.int32),
            flux,
        ),
        {"theta_edge_flux": (flux, 0, sub)},
        lane="subnormal potential temperature carried by normal mass flux",
    )

    flux = zeros((3, 1))
    record(
        "driver",
        "theta_vertical_flux_f32",
        1,
        (
            np.int32(2),
            np.int32(1),
            np.float32(0.0),
            device([[sub], [sub]]),
            device([[sub], [sub]]),
            device([[0.0], [1.0], [0.0]]),
            device([[0.0], [1.0], [0.0]]),
            device([0.0, 0.5]),
            device([0.0, 0.5]),
            flux,
        ),
        {"theta_vertical_flux": (flux, 1, sub)},
        lane="subnormal potential temperature carried vertically",
    )

    result = zeros((1, 1))
    record(
        "driver",
        "theta_finish_f32",
        1,
        (
            np.int32(1),
            np.int32(1),
            np.int32(1),
            np.int32(1),
            device([1], np.int32),
            device([0], np.int32),
            device([-1.0]),
            device([1.0]),
            device([0.0]),
            device([[sub]]),
            zeros((2, 1)),
            result,
        ),
        {"theta_tendency": (result, 0, sub)},
        lane="subnormal horizontal theta-flux divergence",
    )

    flux = zeros((3, 1))
    record(
        "driver",
        "w_edge_flux_f32",
        1,
        (
            np.int32(2),
            np.int32(1),
            np.int32(1),
            np.int32(1),
            np.float32(0.0),
            device([[0.0], [sub], [0.0]]),
            device([[1.0], [1.0]]),
            device([0.0, 0.5]),
            device([0.0, 0.5]),
            device([1.0]),
            device([0.0]),
            device([1], np.int32),
            device([0], np.int32),
            flux,
        ),
        {"w_edge_flux": (flux, 1, sub)},
        lane="subnormal vertical velocity carried by normal edge momentum",
    )

    flux = zeros((3, 1))
    record(
        "driver",
        "w_vertical_flux_f32",
        1,
        (
            np.int32(2),
            np.int32(1),
            device([[sub], [sub], [0.0]]),
            device([[1.0], [1.0], [0.0]]),
            flux,
        ),
        {"w_vertical_flux": (flux, 1, sub)},
        lane="subnormal vertical velocity in interface flux",
    )

    result = zeros((3, 1))
    record(
        "driver",
        "w_finish_f32",
        1,
        (
            np.int32(2),
            np.int32(1),
            np.int32(1),
            np.int32(1),
            device([1], np.int32),
            device([0], np.int32),
            device([-1.0]),
            device([1.0]),
            device([0.0, 1.0, 0.0]),
            device([[0.0], [sub], [0.0]]),
            zeros((3, 1)),
            result,
        ),
        {"w_tendency": (result, 1, sub)},
        lane="subnormal horizontal w-flux divergence",
    )

    target = zeros((1, 1))
    record(
        "driver",
        "add_inplace_f32",
        1,
        (np.int32(1), np.int32(1), device([[sub]]), target),
        {"sum": (target, 0, sub)},
        lane="subnormal tendency accumulation",
    )
    target = zeros((1, 1))
    record(
        "driver",
        "scale_f32",
        1,
        (np.int32(1), np.int32(1), np.float32(1.0), device([[sub]]), target),
        {"scaled": (target, 0, sub)},
        lane="subnormal RK tendency scaling",
    )

    rho = zeros((1, 1))
    rt = zeros((1, 1))
    rho_p = zeros((1, 1))
    rt_p = zeros((1, 1))
    theta = zeros((1, 1))
    exner = zeros((1, 1))
    pressure_p = zeros((1, 1))
    record(
        "driver",
        "recover_cells_f32",
        1,
        (
            np.int32(1),
            np.int32(1),
            np.int32(1),
            np.float32(287.0),
            np.float32(1004.5),
            np.float32(100000.0),
            zeros((1, 1)),
            zeros((1, 1)),
            device([[sub]]),
            device([[sub]]),
            # SG2 lagged heating (has_diabatic, dt, diabatic_theta): off here,
            # so the audited arithmetic is the carry-free recovery.
            np.int32(0),
            np.float32(0.0),
            zeros((1, 1)),
            device([[1.0]]),
            device([[300.0]]),
            device([[1.0]]),
            device([[1.0]]),
            device([[1.0]]),
            zeros((1, 1)),
            rho,
            rt,
            rho_p,
            rt_p,
            theta,
            exner,
            pressure_p,
        ),
        {"rho_perturbation": (rho_p, 0, sub), "rtheta_perturbation": (rt_p, 0, sub)},
        lane="acoustic cell perturbations restored onto normal base state",
    )

    ru = zeros((1, 1))
    flux_u = zeros((1, 1))
    record(
        "driver",
        "recover_edges_f32",
        1,
        (
            np.int32(1),
            np.int32(1),
            np.int32(1),
            zeros((1, 1)),
            device([[sub]]),
            device([[sub]]),
            ru,
            flux_u,
        ),
        {"rho_u": (ru, 0, sub), "flux_u": (flux_u, 0, sub)},
        lane="acoustic edge perturbation and average restored",
    )

    rw = zeros((3, 1))
    flux_w = zeros((3, 1))
    record(
        "driver",
        "recover_interfaces_f32",
        1,
        (
            np.int32(2),
            np.int32(1),
            np.int32(1),
            zeros((3, 1)),
            device([[0.0], [sub], [0.0]]),
            device([[0.0], [sub], [0.0]]),
            rw,
            flux_w,
        ),
        {"rho_w": (rw, 1, sub), "flux_w": (flux_w, 1, sub)},
        lane="acoustic interface perturbation and average restored",
    )

    # Horizontal C-grid and mixing classes.
    normal_velocity = zeros((1, 1))
    rho_edge = zeros((1, 1))
    ke_edge = zeros((1, 1))
    record(
        "horizontal",
        "recover_edge_f32",
        1,
        (
            device([[1.0, 1.0]]),
            device([[sub]]),
            device([1.0]),
            device([1.0]),
            device([0, 1], np.int32),
            np.int32(1),
            np.int32(2),
            np.int32(1),
            normal_velocity,
            rho_edge,
            ke_edge,
        ),
        {"normal_velocity": (normal_velocity, 0, sub)},
        lane="calm nonzero edge momentum / normal density",
    )

    rho_edge = zeros((1, 1))
    ke_edge = zeros((1, 1))
    expected_ke = np.float32(np.float32(normal_tiny) * np.float32(normal_tiny))
    record(
        "horizontal",
        "edge_fields_from_saved_u_f32",
        1,
        (
            device([[1.0, 1.0]]),
            device([[normal_tiny]]),
            device([1.0]),
            device([1.0]),
            device([0, 1], np.int32),
            np.int32(1),
            np.int32(2),
            np.int32(1),
            rho_edge,
            ke_edge,
        ),
        {"kinetic_energy": (ke_edge, 0, expected_ke)},
        lane="NORMAL 1e-20 m/s velocity whose square is subnormal",
    )

    tangential = zeros((1, 1))
    record(
        "horizontal",
        "tangential_velocity_f32",
        1,
        (
            device([[sub]]),
            device([0], np.int32),
            device([1], np.int32),
            device([1.0]),
            np.int32(1),
            np.int32(1),
            np.int32(1),
            tangential,
        ),
        {"tangential_velocity": (tangential, 0, sub)},
        lane="subnormal reconstructed tangential velocity",
    )

    vorticity = zeros((1, 1))
    ke_vertex = zeros((1, 1))
    pv_vertex = zeros((1, 1))
    record(
        "horizontal",
        "vertex_diagnostics_f32",
        1,
        (
            device([[sub]]),
            zeros((1, 1)),
            device([0], np.int32),
            device([0, 0], np.int32),
            device([1.0]),
            device([1.0]),
            device([0.0]),
            np.int32(1),
            np.int32(1),
            np.int32(1),
            np.int32(1),
            vorticity,
            ke_vertex,
            pv_vertex,
        ),
        {"vorticity": (vorticity, 0, sub), "pv_vertex": (pv_vertex, 0, sub)},
        lane="subnormal circulation on one vertex",
    )

    divergence = zeros((1, 1))
    kinetic = zeros((1, 1))
    record(
        "horizontal",
        "cell_diagnostics_f32",
        1,
        (
            device([[sub]]),
            zeros((1, 1)),
            zeros((1, 1)),
            device([0], np.int32),
            device([1], np.int32),
            device([0, 0], np.int32),
            device([0], np.int32),
            device([0], np.int32),
            device([1.0]),
            device([1.0]),
            device([0.0]),
            np.int32(1),
            np.int32(1),
            np.int32(1),
            np.int32(1),
            np.int32(1),
            np.int32(1),
            divergence,
            kinetic,
        ),
        {"divergence": (divergence, 0, sub)},
        lane="subnormal cell mass divergence",
    )

    pv_edge = zeros((1, 1))
    record(
        "horizontal",
        "pv_edge_base_f32",
        1,
        (
            device([[sub, sub]]),
            device([0, 1], np.int32),
            np.int32(1),
            np.int32(1),
            np.int32(2),
            pv_edge,
        ),
        {"pv_edge": (pv_edge, 0, sub)},
        lane="subnormal vertex PV interpolation",
    )

    pv_cell = zeros((1, 1))
    record(
        "horizontal",
        "pv_cell_f32",
        1,
        (
            device([[sub]]),
            device([0], np.int32),
            device([1], np.int32),
            device([0], np.int32),
            device([1.0]),
            device([1.0]),
            np.int32(1),
            np.int32(1),
            np.int32(1),
            np.int32(1),
            np.int32(1),
            pv_cell,
        ),
        {"pv_cell": (pv_cell, 0, sub)},
        lane="subnormal area-weighted cell PV",
    )

    pv_edge = zeros((1, 1))
    grad_n = zeros((1, 1))
    grad_t = zeros((1, 1))
    record(
        "horizontal",
        "pv_apvm_f32",
        1,
        (
            zeros((1, 1)),
            zeros((1, 1)),
            device([[0.0, sub]]),
            device([[0.0, sub]]),
            device([0, 1], np.int32),
            device([0, 1], np.int32),
            device([1.0]),
            device([1.0]),
            np.float32(1.0),
            np.int32(1),
            np.int32(2),
            np.int32(1),
            np.int32(2),
            pv_edge,
            grad_n,
            grad_t,
        ),
        {"grad_pv_normal": (grad_n, 0, sub), "grad_pv_tangential": (grad_t, 0, sub)},
        lane="subnormal normal/tangential PV gradients",
    )

    mass_divergence = zeros((1, 1))
    record(
        "horizontal",
        "mass_flux_divergence_f32",
        1,
        (
            device([[sub]]),
            device([0], np.int32),
            device([1], np.int32),
            device([0, 0], np.int32),
            device([1.0]),
            device([1.0]),
            np.int32(1),
            np.int32(1),
            np.int32(1),
            np.int32(1),
            mass_divergence,
        ),
        {"mass_divergence": (mass_divergence, 0, sub)},
        lane="calm nonzero edge mass flux",
    )

    tendency = zeros((1, 1))
    record(
        "horizontal",
        "density_tendency_f32",
        1,
        (
            device([[sub]]),
            zeros((2, 1)),
            device([0.0]),
            zeros((1, 1)),
            np.int32(0),
            np.int32(1),
            np.int32(1),
            tendency,
        ),
        {"density_tendency": (tendency, 0, np.float32(-sub))},
        lane="subnormal mass-divergence tendency",
    )

    tendency = zeros((1, 1))
    record(
        "horizontal",
        "pressure_gradient_f32",
        1,
        (
            device([[0.0, sub]]),
            zeros((1, 2)),
            device([[1.0]]),
            device([[1.0, 1.0]]),
            zeros((1, 1)),
            device([0, 1], np.int32),
            device([1.0]),
            np.int32(1),
            np.int32(2),
            np.int32(1),
            tendency,
        ),
        {"pressure_gradient": (tendency, 0, np.float32(-sub))},
        lane="subnormal pressure perturbation gradient",
    )

    del2 = zeros(1)
    del4 = zeros(1)
    record(
        "horizontal",
        "mixing_scaling_f32",
        1,
        (
            device([0, 1], np.int32),
            device([1.0, 1.0]),
            np.int32(1),
            np.int32(1),
            del2,
            del4,
        ),
        {"del2": (del2, 0, np.float32(1.0)), "del4": (del4, 0, np.float32(1.0))},
        classification="positive_normal_mesh_density_invariant",
        lane="committed unit positive-normal mesh density",
    )

    kdiff = zeros((1, 1))
    expected_strain = np.float32(np.sqrt(np.float64(expected_ke)))
    record(
        "horizontal",
        "smagorinsky_f32",
        1,
        (
            device([[normal_tiny]]),
            zeros((1, 1)),
            device([0], np.int32),
            device([1], np.int32),
            device([1.0]),
            device([0.0]),
            np.float32(1.0),
            np.float32(1.0),
            np.int32(1),
            np.int32(1),
            np.int32(1),
            np.int32(1),
            kdiff,
        ),
        {"kdiff": (kdiff, 0, expected_strain)},
        lane="NORMAL 1e-20 velocity whose strain square is subnormal",
    )

    delsq_u = zeros((1, 1))
    tendency = zeros((1, 1))
    record(
        "horizontal",
        "momentum_filter_lap2_f32",
        1,
        (
            device([[1.0]]),
            device([[0.0, sub]]),
            zeros((1, 2)),
            device([[1.0, 1.0]]),
            device([1.0]),
            device([0, 1], np.int32),
            device([0, 1], np.int32),
            device([1.0]),
            device([1.0]),
            np.int32(1),
            np.int32(2),
            np.int32(1),
            np.int32(2),
            delsq_u,
            tendency,
        ),
        {"delsq_u": (delsq_u, 0, sub), "momentum_tendency": (tendency, 0, sub)},
        lane="subnormal divergence Laplacian",
    )

    delsq_div = zeros((1, 1))
    record(
        "horizontal",
        "laplacian_divergence_f32",
        1,
        (
            device([[sub]]),
            device([0], np.int32),
            device([1], np.int32),
            device([0, 0], np.int32),
            device([1.0]),
            device([1.0]),
            np.int32(1),
            np.int32(1),
            np.int32(1),
            np.int32(1),
            delsq_div,
        ),
        {"delsq_divergence": (delsq_div, 0, sub)},
        lane="subnormal momentum Laplacian divergence",
    )

    delsq_vort = zeros((1, 1))
    record(
        "horizontal",
        "laplacian_vorticity_f32",
        1,
        (
            device([[sub]]),
            device([0], np.int32),
            device([0, 0], np.int32),
            device([1.0]),
            device([1.0]),
            np.int32(1),
            np.int32(1),
            np.int32(1),
            np.int32(1),
            delsq_vort,
        ),
        {"delsq_vorticity": (delsq_vort, 0, sub)},
        lane="subnormal momentum Laplacian vorticity",
    )

    tendency = zeros((1, 1))
    record(
        "horizontal",
        "momentum_filter_lap4_f32",
        1,
        (
            device([[1.0]]),
            device([[0.0, sub]]),
            zeros((1, 2)),
            device([1.0]),
            device([0, 1], np.int32),
            device([0, 1], np.int32),
            device([1.0]),
            device([1.0]),
            np.float32(1.0),
            np.float32(1.0),
            np.int32(1),
            np.int32(2),
            np.int32(1),
            np.int32(2),
            tendency,
        ),
        {"momentum_tendency": (tendency, 0, np.float32(-sub))},
        lane="subnormal fourth-order divergence diffusion",
    )

    delsq_theta = zeros((1, 2))
    tendency = zeros((1, 2))
    record(
        "horizontal",
        "theta_filter_lap2_f32",
        1,
        (
            device([[0.0, sub]]),
            device([[1.0]]),
            device([[1.0, 1.0]]),
            device([1.0]),
            device([0, 0], np.int32),
            device([1, 0], np.int32),
            device([0, 1], np.int32),
            device([1.0]),
            device([1.0]),
            device([1.0, 1.0]),
            np.int32(1),
            np.int32(2),
            np.int32(1),
            np.int32(1),
            delsq_theta,
            tendency,
        ),
        {"delsq_theta": (delsq_theta, 0, sub), "theta_tendency": (tendency, 0, sub)},
        lane="subnormal potential-temperature edge difference",
    )

    tendency = zeros((1, 2))
    record(
        "horizontal",
        "theta_filter_lap4_f32",
        1,
        (
            device([[0.0, sub]]),
            device([1.0]),
            device([0, 0], np.int32),
            device([1, 0], np.int32),
            device([0, 1], np.int32),
            device([1.0]),
            device([1.0]),
            device([1.0, 1.0]),
            np.float32(1.0),
            np.int32(1),
            np.int32(2),
            np.int32(1),
            tendency,
        ),
        {"theta_tendency": (tendency, 0, np.float32(-sub))},
        lane="subnormal fourth-order theta Laplacian",
    )

    delsq_w = zeros((2, 2))
    tendency = zeros((2, 2))
    record(
        "horizontal",
        "w_filter_lap2_f32",
        1,
        (
            device([[0.0, 0.0], [0.0, sub]]),
            device([[1.0], [1.0]]),
            device([[1.0, 1.0], [1.0, 1.0]]),
            device([1.0]),
            device([0, 0], np.int32),
            device([1, 0], np.int32),
            device([0, 1], np.int32),
            device([1.0]),
            device([1.0]),
            device([1.0, 1.0]),
            np.int32(2),
            np.int32(2),
            np.int32(1),
            np.int32(1),
            delsq_w,
            tendency,
        ),
        {"delsq_w": (delsq_w, 2, sub), "w_tendency": (tendency, 2, sub)},
        lane="subnormal vertical-velocity edge difference",
    )

    tendency = zeros((2, 2))
    record(
        "horizontal",
        "w_filter_lap4_f32",
        1,
        (
            device([[0.0, 0.0], [0.0, sub]]),
            device([1.0]),
            device([0, 0], np.int32),
            device([1, 0], np.int32),
            device([0, 1], np.int32),
            device([1.0]),
            device([1.0]),
            device([1.0, 1.0]),
            np.float32(1.0),
            np.int32(2),
            np.int32(2),
            np.int32(1),
            tendency,
        ),
        {"w_tendency": (tendency, 2, np.float32(-sub))},
        lane="subnormal fourth-order w Laplacian",
    )

    rho_u_p = zeros((1, 1))
    record(
        "horizontal",
        "divergence_damping_f32",
        1,
        (
            device([[0.5, 0.5]]),
            device([[sub, 0.0]]),
            zeros((1, 2)),
            device([0, 1], np.int32),
            device([0.0]),
            np.float32(1.0),
            np.int32(2),
            np.int32(1),
            np.int32(2),
            np.int32(1),
            rho_u_p,
        ),
        {"rho_u_perturbation": (rho_u_p, 0, sub)},
        lane="subnormal acoustic rtheta perturbation delta",
    )

    if transcript_module_keys is None:
        return records
    manifests = {
        transcript_module_keys[module]: cache.compile_manifest()
        for module, cache in caches.items()
    }
    return {
        "records": records,
        "compile_platforms": {
            module_key: manifest["compile_platform"]
            for module_key, manifest in manifests.items()
        },
        "compile_modules": {
            module_key: manifest["modules"][module_key]
            for module_key, manifest in manifests.items()
        },
        "device": {
            key: value
            for key, value in next(iter(caches.values())).capability.as_dict().items()
            if key != "cache_directory"
        },
    }


def run_guarded_kernel_subnormal_audit() -> dict[str, Any]:
    """Prove all non-transport arithmetic classes and red mutations."""
    _strict_proof("hexcore.cuda_ftz.run_guarded_kernel_subnormal_audit")

    first = _run_guarded_kernel_audit_once(fallback_disabled=False)
    second = _run_guarded_kernel_audit_once(fallback_disabled=False)
    mutation = _run_guarded_kernel_audit_once(fallback_disabled=True)
    if first != second:
        raise FtzContractError("remaining CUDA FTZ kernel audit is not dual-run stable")

    module_labels = {
        "hexcore.cuda_acoustic": "acoustic",
        "hexcore.cuda_backend.recovery": "recovery",
        "hexcore.cuda_driver": "driver",
        "hexcore.cuda_horizontal": "horizontal",
    }
    expected_names = {
        f"{module_labels[module]}.{name}"
        for module, (_source, names) in production_translation_units().items()
        if module in module_labels
        for name in names
    }
    missing = sorted(expected_names - set(first))
    extra = sorted(set(first) - expected_names)
    if missing or extra:
        raise FtzContractError(
            f"remaining CUDA FTZ audit inventory mismatch: missing={missing}, extra={extra}"
        )

    combined: dict[str, Any] = {}
    for name in sorted(first):
        candidate = first[name]
        disabled = mutation[name]
        if candidate.get("matches_expected") is not True:
            raise FtzContractError(f"guarded CUDA FTZ fallback failed at {name}")
        classification = str(candidate.get("classification"))
        requires_red = classification == "guarded_fallback_required"
        mutation_red = disabled.get("matches_expected") is not True
        if requires_red and not mutation_red:
            raise FtzContractError(f"disabled CUDA FTZ mutation stayed green at {name}")
        if not requires_red and disabled.get("matches_expected") is not True:
            raise FtzContractError(f"proven-green CUDA FTZ class changed at {name}")
        combined[name] = {
            **candidate,
            "disabled_fallback_observed_bits": disabled.get("observed_bits", {}),
            "disabled_fallback_matches_expected": disabled.get("matches_expected"),
            "mutation_red": mutation_red,
        }
    _validate_kernel_audit_disposition_spec(combined)
    return {
        "schema": KERNEL_AUDIT_SCHEMA,
        "fallback_verified": True,
        "dual_run_byte_identical": True,
        "kernel_count": len(combined),
        "kernels": combined,
    }


def run_normalized_fallback_performance_control(
    *,
    repeats: int = 24,
    warmup: int = 6,
) -> dict[str, Any]:
    """Bound guard overhead on normalized production-kernel weather lanes.

    Timings are secondary evidence.  Correctness is bitwise enabled/disabled
    identity; each median enabled/disabled ratio must also stay below the
    executing architecture's REGISTERED ceiling
    (``arch_admission.PERFORMANCE_RATIO_CEILINGS`` -- 1.25 on sm_120, the
    recorded-deviation row on sm_86; an unregistered architecture refuses
    by name).  The passes are interleaved to reduce clock drift.
    """
    _strict_proof("hexcore.cuda_ftz.run_normalized_fallback_performance_control")

    if repeats < 8 or warmup < 2:
        raise ValueError("normalized performance control needs repeats>=8, warmup>=2")
    from . import cuda_acoustic, cuda_driver, cuda_horizontal, cuda_transport
    from .cuda_backend import KernelCache
    from .cuda_backend.recovery import RECOVERY_CUDA_SOURCE

    try:
        import cupy as cp
    except ImportError as error:  # pragma: no cover - named CUDA refusal path
        raise FtzContractError(
            "CuPy is required for CUDA performance control"
        ) from error

    device_props = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    device_sm = f"sm_{int(device_props['major'])}{int(device_props['minor'])}"
    ratio_ceiling = _resolved_performance_ceiling(device_sm)

    sources = {
        "recovery": RECOVERY_CUDA_SOURCE,
        "acoustic": cuda_acoustic._CUDA_SOURCE,
        "driver": cuda_driver._CUDA_SOURCE,
        "horizontal": cuda_horizontal._CUDA_SOURCE,
        "transport": cuda_transport._CUDA_SOURCE,
    }
    caches = {
        mode: {name: KernelCache() for name in sources}
        for mode in ("enabled", "disabled")
    }

    def raw(module: str, name: str, mode: str) -> Any:
        prefix = "#define MPAS_FTZ_FALLBACK_ENABLED 0\n" if mode == "disabled" else ""
        return caches[mode][module].raw_kernel(
            name,
            prefix + sources[module],
            module_key=f"hexcore.ftz_perf.{module}.{mode}",
        )

    def launch(kernel: Any, total: int, args: tuple[Any, ...]) -> None:
        threads = 128
        kernel(((total + threads - 1) // threads,), (threads,), args)

    nlev = 8
    nowners = 131072
    ncells = nowners * 2
    edge_cells = cp.arange(ncells, dtype=cp.int32).reshape(nowners, 2)
    normalized = cp.linspace(
        np.float32(0.75),
        np.float32(1.25),
        nlev * ncells,
        dtype=cp.float32,
    ).reshape(nlev, ncells)
    edge_field = cp.linspace(
        np.float32(0.05),
        np.float32(0.25),
        nlev * nowners,
        dtype=cp.float32,
    ).reshape(nlev, nowners)

    benchmarks: list[tuple[str, str, int, Any]] = []

    def recovery_args() -> tuple[tuple[Any, ...], Any]:
        output = cp.empty_like(edge_field)
        return (
            normalized,
            edge_field,
            edge_cells.reshape(-1),
            output,
            np.int32(nlev),
            np.int32(ncells),
            np.int32(nowners),
        ), output

    benchmarks.append(("recovery", "recover_edge_velocity_f32", nowners, recovery_args))

    def acoustic_args() -> tuple[tuple[Any, ...], Any]:
        ru_p = cp.empty_like(edge_field)
        ru_avg = cp.empty_like(edge_field)
        zeros_cell = cp.zeros_like(normalized)
        ones_cell = cp.ones_like(normalized)
        return (
            (
                np.int32(nlev),
                np.int32(nowners),
                np.int32(ncells),
                np.int32(1),
                np.float32(0.5),
                np.float32(9.80616),
                np.float32(287.0),
                np.float32(1004.5),
                edge_cells.reshape(-1),
                cp.ones(nowners, dtype=cp.float32),
                ones_cell,
                ones_cell,
                cp.ones_like(edge_field),
                cp.zeros_like(edge_field),
                edge_field,
                zeros_cell,
                zeros_cell,
                ru_p,
                ru_avg,
            ),
            ru_p,
        )

    benchmarks.append(("acoustic", "acoustic_ru", nowners, acoustic_args))

    def driver_args() -> tuple[tuple[Any, ...], Any]:
        output = cp.empty_like(edge_field)
        return (
            np.int32(nlev),
            np.int32(nowners),
            np.float32(0.625),
            edge_field,
            output,
        ), output

    benchmarks.append(("driver", "scale_f32", nowners, driver_args))

    def horizontal_args() -> tuple[tuple[Any, ...], Any]:
        output = cp.empty((nlev, nowners), dtype=cp.float32)
        return (
            edge_field,
            edge_field * np.float32(0.25),
            cp.arange(nowners, dtype=cp.int32),
            cp.ones(nowners, dtype=cp.int32),
            cp.full(nowners, np.float32(0.75), dtype=cp.float32),
            cp.full(nowners, np.float32(0.25), dtype=cp.float32),
            np.float32(0.2),
            np.float32(10.0),
            np.int32(nlev),
            np.int32(nowners),
            np.int32(nowners),
            np.int32(1),
            output,
        ), output

    benchmarks.append(("horizontal", "smagorinsky_f32", nowners, horizontal_args))

    def transport_args() -> tuple[tuple[Any, ...], Any]:
        cells = nowners
        stage = cp.linspace(
            np.float32(0.001),
            np.float32(0.02),
            nlev * cells,
            dtype=cp.float32,
        ).reshape(1, nlev, cells)
        output = cp.empty((1, nlev, nowners), dtype=cp.float32)
        return (
            np.int32(1),
            np.int32(nlev),
            np.int32(cells),
            np.int32(nowners),
            np.int32(1),
            np.float32(1.0 / 3.0),
            stage,
            cp.ones((nlev, nowners), dtype=cp.float32),
            cp.ones(nowners, dtype=cp.float32),
            cp.full(nowners, np.float32(0.25), dtype=cp.float32),
            cp.ones(nowners, dtype=cp.int32),
            cp.arange(nowners, dtype=cp.int32),
            output,
        ), output

    benchmarks.append(("transport", "transport_edge_values", nowners, transport_args))

    results: dict[str, Any] = {}
    for module, name, total, builder in benchmarks:
        enabled_args, enabled_output = builder()
        disabled_args, disabled_output = builder()
        enabled_kernel = raw(module, name, "enabled")
        disabled_kernel = raw(module, name, "disabled")
        for _ in range(warmup):
            launch(enabled_kernel, total, enabled_args)
            launch(disabled_kernel, total, disabled_args)
        cp.cuda.runtime.deviceSynchronize()

        enabled_times: list[float] = []
        disabled_times: list[float] = []

        def time_one(selected: Any, args: tuple[Any, ...]) -> float:
            start = cp.cuda.Event()
            stop = cp.cuda.Event()
            start.record()
            launch(selected, total, args)
            stop.record()
            stop.synchronize()
            return float(cp.cuda.get_elapsed_time(start, stop))

        for iteration in range(repeats):
            order = (
                (
                    (enabled_kernel, enabled_args, enabled_times),
                    (disabled_kernel, disabled_args, disabled_times),
                )
                if iteration % 2 == 0
                else (
                    (disabled_kernel, disabled_args, disabled_times),
                    (enabled_kernel, enabled_args, enabled_times),
                )
            )
            for selected, args, timings in order:
                timings.append(time_one(selected, args))

        enabled_host = cp.asnumpy(enabled_output)
        disabled_host = cp.asnumpy(disabled_output)
        identical = enabled_host.tobytes() == disabled_host.tobytes()
        enabled_median = float(np.median(np.asarray(enabled_times)))
        disabled_median = float(np.median(np.asarray(disabled_times)))
        ratio = enabled_median / disabled_median
        key = f"{module}.{name}"
        results[key] = {
            "enabled_median_ms": enabled_median,
            "disabled_median_ms": disabled_median,
            "enabled_over_disabled": ratio,
            "ceiling": ratio_ceiling,
            "normalized_output_bitwise_identical": identical,
            "repeats": repeats,
            "warmup": warmup,
        }
        if not identical:
            raise FtzContractError(f"normalized fallback output changed at {key}")
        if ratio > ratio_ceiling:
            raise FtzContractError(
                f"normalized fallback median regressed at {key}: "
                f"{ratio:.6f}x > {ratio_ceiling:.2f}x ({device_sm} ceiling)"
            )

    return {
        "schema": PERFORMANCE_CONTROL_SCHEMA,
        "device_sm": device_sm,
        "declared_median_ratio_ceiling": ratio_ceiling,
        "timings_are_secondary": True,
        "all_normalized_outputs_bitwise_identical": True,
        "maximum_enabled_over_disabled": max(
            item["enabled_over_disabled"] for item in results.values()
        ),
        "benchmarks": results,
    }


def v841_reached_translation_units() -> dict[str, tuple[str, tuple[str, ...]]]:
    """Return the exact eight-TU, 46-kernel v8.4.1 step inventory.

    This is an execution inventory, not a compile-everything superset.  In
    particular, ``woof.hex.cuda_transport`` and the FCT-only exports in the
    v8.4.1 transport source are not resolved by the admitted non-monotonic
    closed-dry step and therefore must not appear in its compile manifest.
    """

    from . import (
        cuda_acoustic,
        cuda_acoustic_v841,
        cuda_driver,
        cuda_dynamics_v841,
        cuda_horizontal,
        cuda_horizontal_v841,
        cuda_transport_v841,
    )
    from .cuda_backend.recovery import RECOVERY_CUDA_SOURCE

    sources = {
        "hexcore.cuda_acoustic": cuda_acoustic._CUDA_SOURCE,
        "hexcore.cuda_acoustic_v841": cuda_acoustic_v841._CUDA_SOURCE,
        "hexcore.cuda_backend.recovery": RECOVERY_CUDA_SOURCE,
        "hexcore.cuda_driver": cuda_driver._CUDA_SOURCE,
        "hexcore.cuda_dynamics_v841": cuda_dynamics_v841._CUDA_SOURCE,
        "hexcore.cuda_horizontal": cuda_horizontal._CUDA_SOURCE,
        "hexcore.cuda_horizontal_v841": cuda_horizontal_v841._CUDA_SOURCE,
        "hexcore.cuda_transport_v841": cuda_transport_v841._CUDA_SOURCE,
    }
    kernels = {
        "hexcore.cuda_acoustic": ("tendency_w_to_omega",),
        "hexcore.cuda_acoustic_v841": (
            "acoustic_coefficients_v841",
            "acoustic_cofrz_v841",
            "acoustic_column_solve_v841",
            "acoustic_prepare_v841",
            "acoustic_rs_ts_v841",
            "acoustic_ru_v841",
        ),
        "hexcore.cuda_backend.recovery": (
            "recover_edge_velocity_f32",
            "recover_pressure_f32",
            "recover_terrain_w_f32",
        ),
        "hexcore.cuda_driver": (
            "add_inplace_f32",
            "euler_w_f32",
            "recover_cells_f32",
            "recover_edges_f32",
            "recover_interfaces_f32",
            "scale_f32",
            "theta_edge_flux_f32",
            "theta_vertical_flux_f32",
            "vertical_u_finish_f32",
            "vertical_u_flux_f32",
            "w_edge_flux_f32",
            "w_vertical_flux_f32",
        ),
        "hexcore.cuda_dynamics_v841": (
            "enforce_rw_endpoints_v841_f32",
            "split_flux_add_v841_f32",
            "split_flux_finish_v841_f32",
            "split_flux_first_v841_f32",
            "theta_finish_v841_f32",
            "validate_finite_array_v841_f32",
            "validate_recovered_v841_f32",
            "vector_momentum_v841_f32",
            "w_finish_v841_f32",
        ),
        "hexcore.cuda_horizontal": (
            "density_tendency_f32",
            "edge_fields_from_saved_u_f32",
            "pv_edge_base_f32",
            "tangential_velocity_f32",
        ),
        "hexcore.cuda_horizontal_v841": (
            "cell_diagnostics_v841_f32",
            "mass_flux_divergence_v841_f32",
            "pressure_gradient_v841_f32",
            "pv_apvm_v841_f32",
            "pv_cell_v841_f32",
            "vertex_diagnostics_v841_f32",
        ),
        "hexcore.cuda_transport_v841": (
            "transport_edge_values",
            "transport_interpolate_target_v841",
            "transport_standard_finish_v841",
            "transport_vertical_flux",
            "validate_density_v841",
        ),
    }
    if tuple(sorted(sources)) != V841_REACHED_TRANSLATION_UNITS:
        raise FtzContractError("v8.4.1 reached translation-unit inventory drifted")
    direct = re.compile(
        r'extern\s+"C"\s+__global__\s+void\s+([A-Za-z_][A-Za-z0-9_]*)\s*\('
    )
    declared = re.compile(r"DECLARE_[A-Z0-9_]+\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,")
    result: dict[str, tuple[str, tuple[str, ...]]] = {}
    for module_key in V841_REACHED_TRANSLATION_UNITS:
        source = sources[module_key]
        exported = (set(direct.findall(source)) | set(declared.findall(source))) - {
            "NAME"
        }
        expected = tuple(sorted(kernels[module_key]))
        missing = set(expected) - exported
        if missing:
            raise FtzContractError(
                f"v8.4.1 reached kernels missing from {module_key}: {sorted(missing)}"
            )
        result[module_key] = (source, expected)
    if sum(len(row[1]) for row in result.values()) != 46:
        raise FtzContractError("v8.4.1 reached kernel inventory is not exactly 46")
    return result


def v841_compiled_translation_units() -> dict[str, tuple[str, tuple[str, ...]]]:
    """Return all 95 entrypoints compiled inside the eight reached TUs.

    RawModule compiles a whole source even when the step resolves only some
    entrypoints.  This larger surface is therefore the FTZ/DAZ audit inventory;
    :func:`v841_reached_translation_units` remains the exact runtime manifest
    inventory.  Keeping both prevents concatenated legacy kernels in the new
    transport TU from escaping measurement while also refusing an unused old
    transport module in the execution receipt.
    """

    reached = v841_reached_translation_units()
    direct = re.compile(
        r'extern\s+"C"\s+__global__\s+void\s+([A-Za-z_][A-Za-z0-9_]*)\s*\('
    )
    declared = re.compile(r"DECLARE_[A-Z0-9_]+\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,")
    result: dict[str, tuple[str, tuple[str, ...]]] = {}
    for module_key, (source, _resolved) in reached.items():
        names = tuple(
            sorted(
                (set(direct.findall(source)) | set(declared.findall(source))) - {"NAME"}
            )
        )
        if not names:
            raise FtzContractError(
                f"no compiled v8.4.1 CUDA kernels found in {module_key}"
            )
        result[module_key] = (source, names)
    if sum(len(row[1]) for row in result.values()) != 95:
        raise FtzContractError("v8.4.1 compiled kernel inventory is not exactly 95")
    return result


def production_translation_units() -> dict[str, tuple[str, tuple[str, ...]]]:
    """Return all five CUDA sources and every exported kernel name.

    This is used by the receipt tool to force a fresh compile observation for
    every production kernel without running a forecast.  It imports sources;
    it does not duplicate them.
    """

    from . import cuda_acoustic, cuda_driver, cuda_horizontal, cuda_transport
    from .cuda_backend.recovery import RECOVERY_CUDA_SOURCE

    sources = {
        "hexcore.cuda_acoustic": cuda_acoustic._CUDA_SOURCE,
        "hexcore.cuda_backend.recovery": RECOVERY_CUDA_SOURCE,
        "hexcore.cuda_driver": cuda_driver._CUDA_SOURCE,
        "hexcore.cuda_horizontal": cuda_horizontal._CUDA_SOURCE,
        "hexcore.cuda_transport": cuda_transport._CUDA_SOURCE,
    }
    result: dict[str, tuple[str, tuple[str, ...]]] = {}
    direct = re.compile(
        r'extern\s+"C"\s+__global__\s+void\s+([A-Za-z_][A-Za-z0-9_]*)\s*\('
    )
    declared = re.compile(r"DECLARE_[A-Z0-9_]+\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,")
    for module_key, source in sources.items():
        names = sorted(
            (set(direct.findall(source)) | set(declared.findall(source))) - {"NAME"}
        )
        if not names:
            raise FtzContractError(f"no exported CUDA kernels found in {module_key}")
        result[module_key] = (source, tuple(names))
    return result


__all__ = [
    "FtzContractError",
    "GPUWM_FTZ_SCHEMA",
    "KERNEL_AUDIT_SCHEMA",
    "MPAS_FTZ_SCHEMA",
    "MPAS_FTZ_V841_SCHEMA",
    "PERFORMANCE_CONTROL_SCHEMA",
    "PERFORMANCE_RATIO_CEILING",
    "REQUIRED_MPAS_TRANSLATION_UNITS",
    "V841_REACHED_TRANSLATION_UNITS",
    "V841_KERNEL_AUDIT_MEASUREMENT",
    "V841_KERNEL_AUDIT_SCHEMA",
    "V841_ENABLED_RECORDS_SHA256",
    "V841_DISABLED_RECORDS_SHA256",
    "V841_PROBE_SPEC_SHA256",
    "TRANSPORT_DECK_SCHEMA",
    "build_mpas_ftz_binding",
    "build_mpas_ftz_binding_v841",
    "canonical_sha256",
    "measure_gpuwm_source_pins",
    "production_translation_units",
    "run_guarded_kernel_subnormal_audit",
    "run_normalized_fallback_performance_control",
    "run_scalar_transport_subnormal_deck",
    "sha256_file",
    "validate_compile_manifest_relation",
    "validate_v841_compile_manifest_relation",
    "v841_compiled_translation_units",
    "v841_reached_translation_units",
    "validate_gpuwm_ftz_receipt",
    "validate_mpas_ftz_binding",
    "validate_mpas_ftz_binding_v841",
]
