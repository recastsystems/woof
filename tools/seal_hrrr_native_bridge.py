#!/usr/bin/env python3
"""Seal an atomically published native-HRRR bridge tree and external receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone
import re
import time

from woof.hrrr_forecast import validate_hrrr_source_forecast_hours


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _window_shape(value: str) -> str:
    if re.fullmatch(r"[1-9][0-9]*x[1-9][0-9]*", value) is None:
        raise argparse.ArgumentTypeError(
            "expected window shape must be positive NYxNX")
    return value


def _series_hours(path: Path) -> tuple[int, ...]:
    """Return and validate the exact contiguous horizon named by SERIES."""
    hours: list[int] = []
    for number, raw in enumerate(path.read_text().splitlines(), start=1):
        if not raw.strip():
            continue
        columns = raw.split("\t")
        if len(columns) < 3:
            raise ValueError(
                f"{path}:{number}: expected hour, atmosphere, and soil columns")
        try:
            hour = int(columns[0])
        except ValueError as error:
            raise ValueError(
                f"{path}:{number}: invalid forecast hour {columns[0]!r}") from error
        from woof.ingest.native_supplements import supplement_bindings
        supplement_bindings(columns[3:], base=path.parent)
        hours.append(hour)
    try:
        return validate_hrrr_source_forecast_hours(hours)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{path}: expected contiguous source forecast hours: {error}") from error


def expected_atmosphere_selected(gate: dict) -> int:
    """The native atmosphere records one time must hold, counted as woof.ingest.hrrr counts them.

    561 records, plus 50 for each optional hybrid field (the analysed aerosol pair) and each native
    extra record a caller requested, plus one for each optional surface field (the cycle's PBL height,
    776a6ab41b). Breakage this prevents (2.9.4 rolling test, 2026-10-21): the seal still
    expected 561 + 50 * hybrid after the bridge began selecting HPBL, so every native-HRRR preparation
    whose wrfnat carries HPBL stopped at "gate atmosphere_selected_per_time='562', expected '561'".
    """
    from woof.ingest.native_extras import gate_extra_fields
    from woof.ingest.native_supplements import (gate_optional_hybrid_fields,
                                                 gate_optional_surface_fields)
    return (561 + 50 * (len(gate_optional_hybrid_fields(gate)) + len(gate_extra_fields(gate)))
            + len(gate_optional_surface_fields(gate)))


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--series", type=Path, required=True)
    parser.add_argument("--time-evidence", type=Path, required=True)
    parser.add_argument(
        "--expected-window-shape", type=_window_shape, default="207x207")
    args = parser.parse_args()
    if not args.root.is_dir():
        raise FileNotFoundError(args.root)
    hours = _series_hours(args.series)
    gate = dict(
        line.split("\t", 1)
        for line in (args.root / "gate.txt").read_text().splitlines())
    from woof.ingest.native_extras import gate_extra_fields
    from woof.ingest.native_supplements import (gate_optional_hybrid_fields,
                                                 gate_optional_surface_fields)
    optional_hybrid = gate_optional_hybrid_fields(gate)
    optional_extras = gate_extra_fields(gate)
    optional_surface = gate_optional_surface_fields(gate)
    required = {
        "status": "PASS",
        "forecast_hours": ",".join(map(str, hours)),
        "series_count": str(len(hours)),
        "atmosphere_selected_per_time": str(expected_atmosphere_selected(gate)),
        "soil_selected_per_time": "18",
        "window_shape": args.expected_window_shape,
    }
    for key, expected in required.items():
        if gate.get(key) != expected:
            raise ValueError(f"gate {key}={gate.get(key)!r}, expected {expected!r}")
    from woof.ingest.native_supplements import (
        gate_supplement_fields, gate_soil_surface_fields, verify_supplement_receipt)
    from tools.hrrr_pipeline import _parse_series
    fields = gate_supplement_fields(gate)
    soil_surface_fields = gate_soil_surface_fields(gate)
    ny, nx = map(int, args.expected_window_shape.split("x"))
    series_rows = _parse_series(args.series)
    if any(bool(len(row) > 3) != bool(fields) for row in series_rows):
        raise ValueError("native supplement publication differs from declared series inputs")
    if fields:
        controller = json.loads(args.time_evidence.read_text())
        source_receipt = controller.get("source_hash_preflight", {})
        expected_bindings = {(row[0], "PMSL", str(path))
                             for row in series_rows for path in row[3:]}
        observed_bindings = {(row["forecast_hour"], row["field"], str(Path(row["path"]).resolve()))
                             for row in source_receipt.get("supplement_bindings", ())}
        if (source_receipt.get("manifest_sha256") != args.source_manifest_sha256
                or observed_bindings != expected_bindings):
            raise ValueError("supplement source receipt does not bind this series and manifest")
        verify_supplement_receipt(source_receipt)
        if not (args.root / "supplement-inventory.tsv").is_file():
            raise ValueError("native supplement publication lacks its GRIB selection receipt")
    for hour in hours:
        for role, expected_files in (("atmosphere", 22 + len(fields)
                                      + len(optional_hybrid) + len(optional_extras)
                                      + len(optional_surface)),
                                     ("soil", 2 + len(soil_surface_fields))):
            directory = args.root / f"{role}-f{hour:02d}"
            files = sorted(directory.glob("*.f32le"))
            if len(files) != expected_files:
                raise ValueError(
                    f"{directory} has {len(files)} payloads, expected {expected_files}")
        for name in optional_surface:
            payload = args.root / f"atmosphere-f{hour:02d}" / f"{name}.f32le"
            if not payload.is_file() or payload.stat().st_size != 4 * ny * nx:
                raise ValueError(
                    f"{payload} must carry one float32 {ny}x{nx} surface; "
                    "another payload shape would seed the wrong PBL height")
        for name in soil_surface_fields:
            payload = args.root / f"soil-f{hour:02d}" / f"{name}.f32le"
            if not payload.is_file() or payload.stat().st_size != 4 * ny * nx:
                raise ValueError(
                    f"{payload} must carry one float32 {ny}x{nx} surface; "
                    "another payload shape would read the wrong vegetated area")
    manifest = args.root / "SHA256SUMS"
    if manifest.exists():
        raise FileExistsError(f"refusing to overwrite {manifest}")
    payloads = sorted(
        path for path in args.root.rglob("*")
        if path.is_file() and path != manifest)
    # Hashed across threads (hashlib releases the GIL on large updates),
    # lines kept in sorted path order, so SHA256SUMS is byte-identical to
    # the serial pass.  Breakage it prevents: the serial pass hashed an
    # ~11 GB 24 h bridge tree on ONE core for 100+ s inside every HRRR
    # preparation (single-core sentinel, shared host, 2026-10-27).
    from concurrent.futures import ThreadPoolExecutor
    workers = max(1, min(32, len(os.sched_getaffinity(0))
                         if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        digests = list(pool.map(_sha256, payloads))
    lines = [
        f"{digest}  ./{path.relative_to(args.root).as_posix()}"
        for digest, path in zip(digests, payloads)]
    temporary = manifest.with_suffix(".tmp")
    temporary.write_text("\n".join(lines) + "\n")
    os.replace(temporary, manifest)
    manifest_sha = _sha256(manifest)
    receipt = {
        "schema": "gpuwm-native-hrrr-bridge-seal-v1",
        "status": "PASS",
        "sealed_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(args.root.resolve()),
        "manifest": str(manifest.resolve()),
        "manifest_sha256": manifest_sha,
        "payload_file_count": len(payloads),
        "payload_bytes": sum(path.stat().st_size for path in payloads),
        "source_manifest_sha256": args.source_manifest_sha256,
        "source_forecast_hours": list(hours),
        "model_forcing_hours": list(range(len(hours))),
        "decoder_sha256": _sha256(args.decoder),
        "series_sha256": _sha256(args.series),
        "time_evidence_sha256": _sha256(args.time_evidence),
        "gate": gate,
        "wall_seconds": time.perf_counter() - started,
    }
    if args.receipt.exists():
        raise FileExistsError(f"refusing to overwrite {args.receipt}")
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt_tmp = args.receipt.with_suffix(args.receipt.suffix + ".tmp")
    receipt_tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(receipt_tmp, args.receipt)
    print(json.dumps({
        "status": "PASS",
        "manifest_sha256": manifest_sha,
        "payload_file_count": len(payloads),
        "payload_bytes": receipt["payload_bytes"],
        "receipt": str(args.receipt),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
