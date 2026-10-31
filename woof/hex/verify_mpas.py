"""Whole-model MPAS verification door and ratchet policy.

Deck schema ``mpas-verification-deck-v1`` binds ``pin_set``, ``mesh``, ``date``,
``id``, three ``inputs`` (namelist, streams, init), and a ``native`` reference
to a SHA-pinned ``mpas-native-run-v1`` manifest. Every file reference has
``path`` and ``sha256``; paths are relative to its containing JSON file.
The native manifest repeats pin_set and input digests, declares output_root,
and lists ALL output files with stream, path, sha256, steps and times.
An optional ``optimized`` manifest has the same contract.
The native/optimized reference may instead pin f0's PUBLICATION.json
(``mpas.native-cache-publication/v1``). Its sealed forecast manifest, completion
log, XML stream templates and Rust-read xtime records supply that same contract.

``hex_command`` is an argv array beginning with woof hex; it must consume
{namelist}, {streams}, {init}, {out}. Output paths match native relative paths.
The actual model door determines suite/backends; this module never changes
the default suite. Every run gets a fresh output directory. No shell is used.
``diagnostics`` binds the Rust diagnostic names (see rw_mpas_verify).
Optional ``render`` binds converted wrfout frames and rw_wrfbatch products.
``auxiliary_inputs`` pins relative filenames for stream lists and input tables.
``native_field_inventory`` may pin f0's source-field PUBLICATION and a nested
``registry`` reference. Its sealed native publication, rank count, Registry
bytes and Rust-read field names bind the prognostic inventory used by spread.
Forecasts run in a fresh input directory; reference inputs are never the cwd.
``process_count`` must agree with each sealed native reference. The current
forecast launcher is a single process; it cannot certify another rank count.
RMS and one-step tiers are diagnostics. Landing requires BITWISE, or the
native-spread tier with a pinned ``maths_library_witness`` and its paired replay.

Ratchet schema ``mpas-ratchet-v1`` contains entries keyed by
(pin_set, mesh, date, deck, stream, field), counters and first differing step.
Unknown keys have a zero baseline. This door emits a candidate ratchet and
never mutates the integrator's baseline. The foundation's recorded-missing
policy (integrator ruling 2026-10-13 07:45Z) permits only existing "field missing
in hex" entries at the same process count, with no rising counters. All written
fields must have zero differing words. This scoped acceptance is named
BITWISE-WRITTEN-FIELDS; it never claims complete native BITWISE identity.
Corrupt data, changed shapes, absent times and new missing fields still fail.
When a later landing writes a deferred field, adopting its zero candidate
retires that missing allowance; the field cannot become absent again.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any
import xml.etree.ElementTree as ET

from woof.hex import site_runner

KEY = ("pin_set", "mesh", "date", "deck", "stream", "field")
ABI = "gpuwm-rw-mpas-verify-v1"


class VerificationRefusal(ValueError):
    """A receipt would otherwise certify an unmeasured or different run."""


class InstalledResources:
    """``{resource:KEY}`` in a deck's hex_command: this install's MPAS route resource behind KEY.

    THE BREAKAGE THIS PREVENTS: every MPAS physics deck named the native framework
    library, the pool reader, the MPAS-Model tree and the resources packet by a
    path on one lab host, so no deck could run anywhere else.  A portable deck
    names them by key; `woof hex physics-resources build` makes them, each pinned
    by SHA-256 in the install's record, and the receipt lists what each key bound.
    """

    def __init__(self, cache: Path | None = None):
        self.cache, self.used = cache, {}

    def __format__(self, key: str) -> str:
        from woof.hex.mpasphys.resources_build import PhysicsResourceError, read_installed, resolve_resource
        try:
            path = resolve_resource(key, self.cache)
        except PhysicsResourceError as error:
            raise VerificationRefusal(f"deck names {{resource:{key}}}: {error}") from error
        self.used[key] = dict(read_installed(self.cache)["resources"][key])
        return str(path)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise VerificationRefusal(f"{path}: expected a JSON object")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if len(encoded.encode()) >= 10 * 1024 * 1024:
        raise VerificationRefusal("receipt exceeds the 10 MB delivery limit; split the deck")
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".writing")
    staging.write_text(encoded, encoding="utf-8")
    staging.replace(path)


def file_ref(ref: dict[str, Any], parent: Path) -> Path:
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
        raise VerificationRefusal("file reference requires exactly path and sha256")
    path = (parent / ref["path"]).resolve()
    if not path.is_file() or digest(path) != ref["sha256"]:
        raise VerificationRefusal(f"missing or changed pinned input: {path}")
    return path


def identity(deck: dict[str, Any]) -> dict[str, str]:
    result = {name: deck[name] for name in ("pin_set", "mesh", "date")}
    result["deck"] = deck["id"]
    if any(not isinstance(v, str) or not v for v in result.values()):
        raise VerificationRefusal("pin-set, mesh, date and deck identities must be nonempty strings")
    return result


def process_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise VerificationRefusal("proof requires an explicit positive process_count")
    return value


def native_manifest(ref: dict[str, Any], parent: Path, deck: dict[str, Any], binary: Path) -> tuple[Path, dict[str, Any]]:
    path = file_ref(ref, parent)
    data = read_json(path)
    if data.get("schema") == "mpas.native-cache-publication/v1":
        data = published_native_manifest(path, data, parent, deck, binary)
    if data.get("schema") != "mpas-native-run-v1":
        raise VerificationRefusal(f"{path}: unsupported native manifest schema")
    if data.get("pin_set") != deck["pin_set"]:
        raise VerificationRefusal("native run used a different pin set")
    if process_count(data.get("process_count")) != process_count(deck.get("process_count")):
        raise VerificationRefusal("native and hex process counts differ")
    expected = {k: v["sha256"] for k, v in deck["inputs"].items()}
    if data.get("inputs") != expected:
        raise VerificationRefusal("native and hex namelist, streams or init bytes differ")
    auxiliary = {k: v["sha256"] for k, v in deck.get("auxiliary_inputs", {}).items()}
    if data.get("auxiliary_inputs", {}) != auxiliary:
        raise VerificationRefusal("native and hex auxiliary input bytes differ")
    if data.get("complete") is not True:
        raise VerificationRefusal("native forecast completion has not been witnessed")
    if not data.get("outputs"):
        raise VerificationRefusal("native manifest has no output streams")
    root = (path.parent / data["output_root"]).resolve()
    listed: set[Path] = set()
    labels: set[tuple[str, int]] = set()
    for output in data["outputs"]:
        file = file_ref({k: output[k] for k in ("path", "sha256")}, root)
        if not file.is_relative_to(root) or file in listed:
            raise VerificationRefusal("native output paths escape the run or repeat")
        listed.add(file)
        if not output.get("steps") or len(output["steps"]) != len(output.get("times", [])):
            raise VerificationRefusal("native output must declare every record's step and time")
        for step, stamp in zip(output["steps"], output["times"]):
            if isinstance(step, bool) or not isinstance(step, int) or step < 0 or not isinstance(stamp, str):
                raise VerificationRefusal("invalid native output step/time")
            label = (output["stream"], step)
            if label in labels:
                raise VerificationRefusal("native output repeats a stream/step")
            labels.add(label)
    # A truncated manifest cannot make omitted reference output disappear.
    excluded = {file_ref(ref, parent) for ref in
                (*deck["inputs"].values(), *deck.get("auxiliary_inputs", {}).values())}
    actual = stream_inventory(binary, root) - excluded
    if actual != listed:
        raise VerificationRefusal(f"native manifest omits or invents output files: {sorted(map(str, actual ^ listed))}")
    data["resolved_output_root"] = root
    return path, data


def published_native_manifest(path: Path, publication: dict[str, Any], parent: Path,
                              deck: dict[str, Any], binary: Path) -> dict[str, Any]:
    """Consume f0's sealed publication without a second hand-written manifest.

    Python binds hashes, XML stream declarations and forecast clock metadata;
    Rust reads every netCDF record's xtime and storage inventory.
    """
    root = path.parent
    files = publication["files"]
    if "MANIFEST.json" not in files:
        raise VerificationRefusal("native publication has no forecast manifest")
    for name, sha in files.items():
        target = (root / name).resolve()
        if not target.is_relative_to(root) or not target.is_file() or digest(target) != sha:
            raise VerificationRefusal(f"native publication file missing or changed: {name}")
    source = read_json(root / "MANIFEST.json")
    if source.get("schema") != "mpas.native-forecast-cache/v1":
        raise VerificationRefusal("native publication is not a completed forecast cache")
    logs = [name for name in files if name.startswith("log.atmosphere.") and name.endswith(".out")]
    if not logs or not any("Finished running the atmosphere core" in (root / name).read_text()
                           for name in logs):
        raise VerificationRefusal("native publication has no pinned forecast completion witness")
    inputs = {key: value["sha256"] for key, value in deck["inputs"].items()}
    init_name = file_ref(deck["inputs"]["init"], parent).name
    if (source["namelist_sha256"] != inputs["namelist"] or
            source["streams_sha256"] != inputs["streams"] or
            source["inputs"].get(init_name) != inputs["init"]):
        raise VerificationRefusal("native and hex namelist, streams or init bytes differ")
    streams = ET.parse(file_ref(deck["inputs"]["streams"], parent)).getroot()
    auxiliary = deck.get("auxiliary_inputs", {})
    for entry in streams.iter("file"):
        name = entry.get("name")
        if name not in auxiliary or auxiliary[name]["sha256"] != source["inputs"].get(name):
            raise VerificationRefusal(f"native stream field list is not pinned for hex: {name}")
    for name, ref in auxiliary.items():
        file_ref(ref, parent)
        if source["inputs"].get(name) != ref["sha256"]:
            raise VerificationRefusal(f"native and hex auxiliary input bytes differ: {name}")
    output_streams = [s for s in streams if "output" in s.get("type", "")]
    tokens = {"Y": r"\d{4}", "M": r"\d{2}", "D": r"\d{2}",
              "h": r"\d{2}", "m": r"\d{2}", "s": r"\d{2}"}
    patterns = []
    for stream in output_streams:
        template = stream.get("filename_template", "")
        pattern = re.escape(template)
        for key, value in tokens.items():
            pattern = pattern.replace(re.escape("$" + key), f"(?P<{key}>{value})")
        patterns.append((stream, re.compile(pattern)))
    runtime = source["runtime"]
    start = datetime.strptime(runtime["start"], "%Y-%m-%d_%H:%M:%S")
    dt = Decimal(str(runtime["dt"]))
    if dt <= 0:
        raise VerificationRefusal("native timestep must be positive")
    outputs = []
    for name, sha in source["outputs"].items():
        if files.get(name) != sha:
            raise VerificationRefusal(f"native output is not sealed by publication: {name}")
        matches = [(stream, pattern.fullmatch(name)) for stream, pattern in patterns if pattern.fullmatch(name)]
        if len(matches) != 1:
            raise VerificationRefusal(f"native output has ambiguous or absent stream declaration: {name}")
        completed = subprocess.run([*comparator_argv(binary), "--describe", str(root / name)],
                                   capture_output=True, text=True, check=False)
        if completed.returncode:
            raise VerificationRefusal(f"Rust native stream metadata failed: {completed.stderr.strip()}")
        metadata = json.loads(completed.stdout)
        stream, match = matches[0]
        times = metadata["times"]
        clock = None
        if times is None:
            times = stream_filename_times(stream, match.groupdict(), metadata["records"])
            clock = {"filename_template": stream.get("filename_template")}
        steps = []
        for stamp in times:
            elapsed = datetime.strptime(stamp, "%Y-%m-%d_%H:%M:%S") - start
            seconds = Decimal(elapsed.days * 86400 + elapsed.seconds)
            if seconds < 0 or seconds % dt:
                raise VerificationRefusal(f"native output time is off its timestep clock: {stamp}")
            steps.append(int(seconds / dt))
        if steps != sorted(set(steps)):
            raise VerificationRefusal("native stream times repeat or move backwards")
        output = {"stream": stream.get("name"), "path": name, "sha256": sha,
                  "times": times, "steps": steps, "time_source": "STREAM-FILENAME+RECORD-COUNT" if clock else metadata["time_source"],
                  "records": metadata["records"]}
        if clock:
            output["clock_witness"] = clock
        outputs.append(output)
    return {"schema": "mpas-native-run-v1", "pin_set": source["pin_set_hash"],
            "process_count": process_count(source.get("ranks")),
            "inputs": inputs, "complete": True, "output_root": ".", "outputs": outputs,
            "auxiliary_inputs": {k: v["sha256"] for k, v in auxiliary.items()},
            "cache_manifest_sha256": files["MANIFEST.json"],
            "native_input_inventory": source["inputs"],
            "prognostic_fields": source.get("prognostic_fields", {})}


def stream_filename_times(stream: ET.Element, tokens: dict[str, str], records: int) -> list[str]:
    """Bind missing xtime to sealed XML filename/record cadence, never drop fields."""
    if set(tokens) != set("YMDhms") or not isinstance(records, int) or records < 1:
        raise VerificationRefusal("stream without xtime needs a complete filename timestamp and actual record count")
    first = datetime.strptime("{Y}-{M}-{D}_{h}:{m}:{s}".format_map(tokens), "%Y-%m-%d_%H:%M:%S")
    interval = stream.get("output_interval", "")
    match = re.fullmatch(r"(?:(\d+)_)?(\d+):(\d{2}):(\d{2})", interval)
    if not match:
        raise VerificationRefusal("stream without xtime needs a fixed output_interval clock")
    days, hours, minutes, seconds = (int(v or 0) for v in match.groups())
    cadence = timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
    if cadence.total_seconds() <= 0 or minutes >= 60 or seconds >= 60:
        raise VerificationRefusal("invalid stream output_interval clock")
    if records > 1 and stream.get("filename_interval") == "output_interval":
        raise VerificationRefusal("stream filename_interval declares one record per file but native has more")
    return [(first + r * cadence).strftime("%Y-%m-%d_%H:%M:%S") for r in range(records)]


def bind_native_field_inventory(ref: dict[str, Any], parent: Path, native_path: Path,
                                native: dict[str, Any], binary: Path) -> dict[str, Any]:
    """Authenticate f0's separate inventory without modifying its forecast cache.

    Registry.xml:1683-1748 declares the atmospheric state/scalars inventoried
    by f0. This is metadata orchestration; Rust still reads every stored word.
    An incomplete native water inventory remains explicit, never a zero budget.
    """
    if set(ref) != {"path", "sha256", "registry"}:
        raise VerificationRefusal("source-field inventory requires path, sha256 and pinned registry")
    publication_path = file_ref({key: ref[key] for key in ("path", "sha256")}, parent)
    publication = read_json(publication_path)
    if publication.get("schema") != "mpas.native-cache-publication/v1":
        raise VerificationRefusal("source-field inventory needs a sealed native publication")
    root = publication_path.parent
    members = publication.get("files", {})
    if "MANIFEST.json" not in members:
        raise VerificationRefusal("source-field inventory publication lacks MANIFEST.json")
    for name, sha in members.items():
        target = (root / name).resolve()
        if not target.is_relative_to(root) or not target.is_file() or digest(target) != sha:
            raise VerificationRefusal(f"source-field inventory member missing or changed: {name}")
    inventory = read_json(root / "MANIFEST.json")
    if inventory.get("schema") != "mpas.native-source-field-inventory/v1":
        raise VerificationRefusal("unsupported source-field inventory schema")
    if (inventory.get("native_publication_sha256") != digest(native_path) or
            inventory.get("native_manifest_sha256") != native.get("cache_manifest_sha256") or
            process_count(inventory.get("ranks")) != native["process_count"]):
        raise VerificationRefusal("source-field inventory belongs to another native forecast or process count")
    registry = file_ref(ref["registry"], parent)
    if inventory.get("source_sha256", {}).get("src/core_atmosphere/Registry.xml") != digest(registry):
        raise VerificationRefusal("source-field inventory Registry bytes differ from the pinned source")
    streams = inventory.get("streams", {})
    prognostics = inventory.get("prognostic_fields", {})
    examples = inventory.get("examples", {})
    actual_streams = {output["stream"] for output in native["outputs"]}
    if set(streams) != actual_streams or set(prognostics) != actual_streams or set(examples) != actual_streams:
        raise VerificationRefusal("source-field inventory omits or invents a native stream")
    outputs = {output["path"]: output for output in native["outputs"]}
    for stream, fields in streams.items():
        names = [field["field"] for field in fields]
        declared = prognostics[stream]
        if (len(set(names)) != len(names) or not isinstance(declared, list) or
                any(not isinstance(name, str) for name in declared) or
                len(set(declared)) != len(declared) or not set(declared).issubset(names)):
            raise VerificationRefusal(f"invalid source-field prognostic inventory: {stream}")
        example = examples[stream]
        output = outputs.get(example["file"])
        if not output or output["stream"] != stream or output["sha256"] != example["sha256"]:
            raise VerificationRefusal(f"source-field inventory example is not sealed native output: {stream}")
        completed = subprocess.run([*comparator_argv(binary), "--describe",
            str(native["resolved_output_root"] / output["path"])], capture_output=True, text=True, check=False)
        if completed.returncode:
            raise VerificationRefusal(f"Rust source-field metadata failed: {completed.stderr.strip()}")
        if set(json.loads(completed.stdout)["fields"]) != set(names):
            raise VerificationRefusal(f"source-field inventory differs from stored native fields: {stream}")
    if native.get("prognostic_fields") and native["prognostic_fields"] != prognostics:
        raise VerificationRefusal("native cache and source-field prognostic inventories differ")
    native["prognostic_fields"] = prognostics
    return {"publication": str(publication_path), "publication_sha256": digest(publication_path),
            "manifest_sha256": members["MANIFEST.json"], "registry_sha256": digest(registry),
            "process_count": native["process_count"], "prognostic_fields": prognostics,
            "water_integral_complete": inventory.get("water_integral_complete") is True,
            "water_integral_limitation": inventory.get("water_integral_limitation"),
            "water_bindings": inventory.get("water_bindings", {})}


def aggregate(rows: list[dict[str, Any]], context: dict[str, str]) -> dict[str, Any]:
    entries: dict[tuple[str, ...], dict[str, Any]] = {}
    seen: set[tuple[str, str, int]] = set()
    for row in rows:
        sample = (row["stream"], row["field"], row["step"])
        if sample in seen:
            raise VerificationRefusal("comparator repeated a stream/field/step")
        seen.add(sample)
        key = tuple(context[k] for k in ("pin_set", "mesh", "date", "deck")) + (row["stream"], row["field"])
        value = entries.setdefault(key, {**context, "stream": row["stream"], "field": row["field"],
            "differing_words": 0, "compared_words": 0, "max_ulp": 0,
            "first_differing_step": None, "missing_samples": 0, "statuses": []})
        words = row["differing_words"]
        if isinstance(words, bool) or not isinstance(words, int) or words < 0:
            raise VerificationRefusal("invalid differing-word counter")
        value["differing_words"] += words
        value["compared_words"] += row["compared_words"]
        value["max_ulp"] = max(value["max_ulp"], row["max_ulp"] or 0)
        if words:
            old = value["first_differing_step"]
            value["first_differing_step"] = row["step"] if old is None else min(old, row["step"])
        if row["status"] != "compared":
            value["missing_samples"] += 1
            if row["status"] not in value["statuses"]:
                value["statuses"].append(row["status"])
    if not entries:
        raise VerificationRefusal("no native fields were compared")
    return {"schema": "mpas-ratchet-v1", "entries": [entries[k] for k in sorted(entries)]}


def ratchet_failures(candidate: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    if candidate.get("schema") != "mpas-ratchet-v1" or baseline.get("schema") != "mpas-ratchet-v1":
        raise VerificationRefusal("unsupported ratchet schema")
    old = {}
    for row in baseline["entries"]:
        key = tuple(row[k] for k in KEY)
        if key in old:
            raise VerificationRefusal("baseline ratchet contains duplicate keys")
        for count in ("differing_words", "max_ulp", "missing_samples"):
            if isinstance(row[count], bool) or not isinstance(row[count], int) or row[count] < 0:
                raise VerificationRefusal(f"invalid baseline {count}")
        old[key] = row
    failures = []
    keys = set()
    contexts = set()
    for row in candidate["entries"]:
        key = tuple(row[k] for k in KEY)
        keys.add(key)
        contexts.add(key[:4])
        previous = old.get(key, {})
        label = "/".join(key)
        if key in old and row.get("process_count") != previous.get("process_count"):
            failures.append(f"{label}: ratchet process count differs or is unwitnessed")
        for count in ("differing_words", "max_ulp", "missing_samples"):
            if row[count] > previous.get(count, 0):
                failures.append(f"{label}: {count} {previous.get(count, 0)} -> {row[count]}")
        before, after = previous.get("first_differing_step"), row["first_differing_step"]
        if after is not None and before is not None and after < before:
            failures.append(f"{label}: first differing step moved earlier ({before} -> {after})")
        if key in old and row["compared_words"] < previous["compared_words"]:
            failures.append(f"{label}: fewer native words compared")
    for key in old:
        if key[:4] in contexts and key not in keys:
            failures.append(f"{'/'.join(key)}: previously inventoried field absent from comparison")
    return failures



def recorded_missing_failures(result: dict[str, Any], context: dict[str, Any],
                              baseline: dict[str, Any]) -> list[str]:
    """Enforce the recorded-missing foundation rule without enrolling new gaps.

    Integrator ruling 2026-10-13 07:45Z: every written word is equal; each
    deferred field remains in the ratchet, until its owner's zero landing.
    Structural errors other than that exact missing-field status cannot use
    the allowance, even if their word counters happen to fit the baseline.
    """
    rows = result.get("fields", [])
    if not rows or not any(r["status"] == "compared" and r["compared_words"] > 0 for r in rows):
        return ["recorded-missing proof requires written native words"]
    candidate = aggregate(rows, context)
    failures = ratchet_failures(candidate, baseline)
    old = {tuple(r[k] for k in KEY): r for r in baseline["entries"]}
    for row in rows:
        label = f"{row['stream']}/{row['field']} step {row['step']}"
        if row["status"] == "compared":
            if row["differing_words"]:
                failures.append(f"{label}: computed words differ")
            continue
        key = tuple(context[k] for k in KEY[:4]) + (row["stream"], row["field"])
        previous = old.get(key)
        if (row["status"] != "field missing in hex" or previous is None or
                previous.get("statuses") != ["field missing in hex"] or
                previous.get("process_count") != context["process_count"] or
                previous.get("missing_samples", 0) <= 0 or previous.get("differing_words", 0) <= 0):
            failures.append(f"{label}: missing field is not an existing recorded allowance ({row['status']})")
    return failures


def tier(result: dict[str, Any], residue: list[str]) -> tuple[str | None, list[str]]:
    """Report diagnostic tiers; this function alone never authorizes landing."""
    rows = result["fields"]
    if not rows:
        return None, ["no output fields compared"]
    structural = [f"{r['stream']}/{r['field']} step {r['step']}: {r['status']}"
                  for r in rows if r["status"] != "compared"]
    if structural:
        return None, structural
    if all(r["differing_words"] == 0 for r in rows):
        return "BITWISE", []
    spread = result.get("spread_samples", [])
    failure = result.get("spread_error")
    if spread and not failure and all(r.get("passed") is True for r in spread) and residue:
        # The one-step witness is useful even when a later forecast has a
        # declared libm residue. It never excuses a later spread violation.
        first_steps = {r["step"] for r in rows if r["step"] > 0}
        first = min(first_steps) if first_steps else None
        if first == 1 and all(r["differing_words"] == 0 for r in rows if r["step"] <= 1):
            return "ONE-STEP-IDENTICAL", []
        return "INSIDE-NATIVE-SPREAD", []
    reasons = [failure] if failure else []
    reasons.extend(f"spread exceeded at {r['stream']} step {r['step']}" for r in spread if r.get("passed") is not True)
    if not residue:
        reasons.append("differing words have no named DECLARED LIBM RESIDUE")
    if not spread and not failure:
        reasons.append("native spread samples absent")
    return None, reasons


def maths_library_witness(deck: dict[str, Any], parent: Path, native: Path,
                          optimized: Path | None) -> dict[str, Any] | None:
    """Bind spread acceptance to actual pinned math artifacts and their replay."""
    ref = deck.get("maths_library_witness")
    if ref is None:
        return None
    path = file_ref(ref, parent)
    witness = read_json(path)
    if optimized is None or witness.get("schema") != "mpas-maths-library-witness-v1":
        raise VerificationRefusal("spread maths witness requires the paired optimized native reference")
    bindings = {"pin_set": deck["pin_set"], "process_count": process_count(deck.get("process_count")),
                "strict_native_manifest_sha256": digest(native),
                "optimized_native_manifest_sha256": digest(optimized),
                "declared_libm_residue": deck.get("declared_libm_residue", [])}
    if any(witness.get(k) != v for k, v in bindings.items()) or not bindings["declared_libm_residue"]:
        raise VerificationRefusal("maths witness does not bind these native runs, process count and named residues")
    process_count(witness.get("process_count"))
    libraries = witness.get("libraries", {})
    if set(libraries) != {"native_strict", "native_optimized", "hex"}:
        raise VerificationRefusal("maths witness must pin strict, optimized and hex math artifacts")
    hashes = {k: digest(file_ref(v, path.parent)) for k, v in libraries.items()}
    replay_path = file_ref(witness.get("replay", {}), path.parent)
    replay = read_json(replay_path)
    if (replay.get("schema") != "mpas-maths-replay-v1" or replay.get("libraries") != hashes or
            replay.get("process_count") != bindings["process_count"] or
            replay.get("routines") != bindings["declared_libm_residue"] or
            isinstance(replay.get("tested_words"), bool) or not isinstance(replay.get("tested_words"), int) or
            replay["tested_words"] <= 0 or isinstance(replay.get("differing_words"), bool) or
            replay.get("differing_words") != 0):
        raise VerificationRefusal("maths library replay lacks a matching nonempty zero-word witness")
    process_count(replay.get("process_count"))
    return {"path": str(path), "sha256": digest(path), "libraries": hashes,
            "replay_sha256": digest(replay_path), **bindings}


def acceptance_tier(result: dict[str, Any], diagnostic_tier: str | None,
                    witness: dict[str, Any] | None,
                    field_inventory: dict[str, Any] | None = None) -> tuple[str | None, list[str]]:
    rows = result.get("fields", [])
    if rows and all(r["status"] == "compared" and r["differing_words"] == 0 for r in rows):
        return "BITWISE", []
    if field_inventory is not None and field_inventory.get("water_integral_complete") is not True:
        return None, ["native streams lack the complete water integral required for spread: " +
                      str(field_inventory.get("water_integral_limitation"))]
    if diagnostic_tier in ("ONE-STEP-IDENTICAL", "INSIDE-NATIVE-SPREAD") and witness is not None:
        return "INSIDE-NATIVE-SPREAD", []
    return None, ["nonzero words require a pinned maths-library witness; RMS and one-step grades are diagnostics only"]


def comparator_argv(binary: Path) -> list[str]:
    """The comparator's argv prefix: the multi-call ``rw_mpas_static`` runs it as ``verify``."""
    multicall = binary.stem == "rw_mpas_static" or binary.stem.startswith("rw_mpas_static-")
    return [str(binary), "verify"] if multicall else [str(binary)]


def installed_comparator() -> Path | None:
    """The stored-word comparator an install already carries, or None.

    THE BREAKAGE THIS PREVENTS: rw_mpas_verify ships only as the ``verify``
    tool of the multi-call ``rw_mpas_static`` (woof's bridges, and this
    wheel's libexec once its bridge is that build), never under its own name,
    so a PyPI install refused every deck with "rw_mpas_verify unavailable"
    although the comparator was installed.  The first copy whose ``verify
    --abi`` answers this contract is used; older static builders without the
    tool are skipped.
    """
    from woof.hex import engines
    candidates = [Path(__file__).resolve().parent / "libexec" / engines.executable_name("rw_mpas_static")]
    package = engines._gpuwm_package_dir()
    if package is not None:
        name = engines.executable_name("rw_mpas_static")
        candidates += [package / "libexec" / "bridges" / name, package.parent / "libexec" / "bridges" / name]
    for path in candidates:
        if not path.is_file() or not os.access(path, os.X_OK):
            continue
        probe = subprocess.run([*comparator_argv(path), "--abi"], capture_output=True, text=True, check=False)
        if probe.returncode == 0 and probe.stdout.strip() == ABI:
            return path.resolve()
    return None


def comparator_path(given: Path | None) -> Path:
    name = str(given) if given else os.environ.get("WOOF_HEX_RW_MPAS_VERIFY", "rw_mpas_verify")
    located = shutil.which(name)
    if not located and not given and "WOOF_HEX_RW_MPAS_VERIFY" not in os.environ:
        found = installed_comparator()
        located = str(found) if found else None
    if not located:
        raise VerificationRefusal("rw_mpas_verify unavailable: neither on PATH nor in this install's "
                                  "rw_mpas_static (woof libexec/bridges); build rw-mpas --bin rw_mpas_static "
                                  "and pass --comparator or set WOOF_HEX_RW_MPAS_VERIFY")
    path = Path(located).resolve()
    probe = subprocess.run([*comparator_argv(path), "--abi"], capture_output=True, text=True, check=False)
    if probe.returncode:
        detail = probe.stderr.strip().splitlines()
        cause = detail[0] if detail else 'no stderr from probe'
        raise VerificationRefusal(f"Rust comparator ABI probe failed (exit {probe.returncode}): {cause}")
    if probe.stdout.strip() != ABI:
        raise VerificationRefusal("Rust comparator ABI differs from the whole-stream contract")
    return path


def source_provenance() -> dict[str, Any]:
    tree = Path(__file__).resolve().parents[2]
    result = {}
    for name, command in {"commit": ["rev-parse", "HEAD"],
                          "dirty": ["status", "--porcelain", "--", "tree/src", "tree/tools", "tree/tests"]}.items():
        try:
            completed = subprocess.run(["git", "-C", str(tree.parent), *command],
                                       capture_output=True, text=True, check=False)
            result[name] = completed.stdout.strip() if completed.returncode == 0 else None
        except OSError:
            result[name] = None  # Installed wheels need no git executable.
    payload = hashlib.sha256()
    package = Path(__file__).resolve().parent
    for path in sorted(package.rglob("*.py")):
        payload.update(str(path.relative_to(package)).encode() + b"\0")
        payload.update(bytes.fromhex(digest(path)))
    result["python_sources_sha256"] = payload.hexdigest()
    return result


def stream_inventory(binary: Path, root: Path) -> set[Path]:
    completed = subprocess.run([*comparator_argv(binary), "--inventory-root", str(root)],
                               capture_output=True, text=True, check=False)
    if completed.returncode:
        raise VerificationRefusal(f"Rust output inventory failed: {completed.stderr.strip()}")
    return {Path(p).resolve() for p in json.loads(completed.stdout)}


def render_evidence(config: dict[str, Any], root: Path, work: Path) -> dict[str, Any]:
    from woof import rustwx
    renderer = rustwx.find_renderer()
    if renderer is None:
        raise VerificationRefusal("rw_wrfbatch unavailable for requested weather difference maps")
    frames = lambda key: [file_ref(ref, root) for ref in config[key]]
    native, hex_frames = frames("native_frames"), frames("hex_frames")
    if any(not p.is_file() for p in native + hex_frames):
        raise VerificationRefusal("difference rendering requires Rust-converted wrfout frames")
    images, failures, skips, differences = rustwx.run_renderer_difference(
        renderer, hex_frames, native, store_root=work / "render-store", out_dir=work / "maps",
        products=config["products"], labels=("hex", "MPAS strict"), timeidx=config["time_index"])
    if not images or skips:
        failures.append("requested difference products did not all render; evidence maps are incomplete")
    return {"renderer": str(renderer), "renderer_sha256": digest(renderer),
            "frames_sha256": {str(p): digest(p) for p in native + hex_frames},
            "images": [str(p) for p in images],
            "failures": failures, "skips": skips, "differences": differences}


#: The mode a proof runs when its deck names none.  The door's default for an MPAS suite route is correct
#: (owner ruling 2026-10-13); every proof harness and gate passes exact explicitly (coordinator 2026-10-14), so the
#: gates keep running the numerics they ran before the switch existed.
PROOF_PHYSICS_MODE = "exact"


def command_physics_mode(command: list[str]) -> str | None:
    """The ``--physics-mode`` a deck's hex_command names, or None.

    Breakage the refusals prevent: a repeated or malformed flag would make the receipt record a mode the
    door did not run."""
    values = []
    for i, part in enumerate(command):
        if part == "--physics-mode":
            values.append(command[i + 1] if i + 1 < len(command) else None)
        elif part.startswith("--physics-mode="):
            values.append(part.split("=", 1)[1])
    if len(values) > 1:
        raise VerificationRefusal("model command names --physics-mode more than once")
    if values and values[0] not in ("exact", "correct"):
        raise VerificationRefusal(f"model command --physics-mode must be exact or correct, not {values[0]!r}")
    return values[0] if values else None


def proof_physics_mode(command: list[str], namelist: Path | None) -> tuple[list[str], dict[str, Any]]:
    """The argument vector a proof runs and the receipt's ``physics_mode`` record.

    A deck that names a mode (``--physics-mode`` in hex_command, or config_physics_mode in its namelist)
    runs it; a deck that names none runs ``--physics-mode exact``, appended here so no deck file changes."""
    named = command_physics_mode(command)
    if named is not None:
        return list(command), {"mode": named, "source": "hex_command", "appended": False}
    if namelist is not None:
        from .mpasphys.forecast_v842 import split_physics_mode
        _, namelist_mode, _ = split_physics_mode(namelist.read_text())
        if namelist_mode is not None:
            return list(command), {"mode": namelist_mode, "source": "namelist config_physics_mode",
                                   "appended": False}
    return [*command, "--physics-mode", PROOF_PHYSICS_MODE], {
        "mode": PROOF_PHYSICS_MODE, "source": "verify_mpas default (the deck names no mode)", "appended": True}


#: The backend a proof's MPAS-suite slots take when its deck names none.  Every deck was written while
#: native-cpu was the door default, so the gates keep running what they proved (the native-backend forecasts);
#: a deck that proves the GPU names its slots (the headline's overlay) or --physics-default-backend.
PROOF_DEFAULT_BACKEND = "native-cpu"


def proof_default_backend(command: list[str]) -> list[str]:
    """``command`` with ``--physics-default-backend native-cpu`` appended when the deck names no default
    backend, so no deck file changes (as :func:`proof_physics_mode` does for the mode)."""
    if any(part == "--physics-default-backend" or part.startswith("--physics-default-backend=")
           for part in command):
        return list(command)
    return [*command, "--physics-default-backend", PROOF_DEFAULT_BACKEND]


#: The kernel numerics a proof runs when its deck names none.  The door default is fast (owner ruling
#: 2026-10-16: strict numerics is opt-in, never the default); bitwise identity to native MPAS is a property of
#: strict numerics, so every proof and gate deck runs ``--numerics strict``, appended here so no deck changes.
PROOF_NUMERICS = "strict"


def proof_numerics(command: list[str]) -> tuple[list[str], dict[str, Any]]:
    """``command`` with ``--numerics strict`` appended when the deck names none, and the receipt record."""
    values = [command[i + 1] if i + 1 < len(command) else None
              for i, part in enumerate(command) if part == "--numerics"]
    values += [part.split("=", 1)[1] for part in command if part.startswith("--numerics=")]
    if len(values) > 1:
        raise VerificationRefusal("model command names --numerics more than once")
    if values:
        if values[0] not in ("fast", "strict"):
            raise VerificationRefusal(f"model command --numerics must be fast or strict, not {values[0]!r}")
        return list(command), {"mode": values[0], "source": "hex_command", "appended": False}
    return [*command, "--numerics", PROOF_NUMERICS], {
        "mode": PROOF_NUMERICS, "source": "verify_mpas default (the deck names no numerics)", "appended": True}


def validate_model_command(command: list[str]) -> None:
    """A placeholder mentioned as provenance is not a model input binding."""
    if len(command) < 2 or command[:2] != ["woof hex", "forecast"]:
        raise VerificationRefusal("deck must execute the real woof hex model door")
    command_physics_mode(command)
    flags = {"namelist": "--physics-namelist", "streams": "--streams",
             "init": "--init", "out": "--out"}
    for name, flag in flags.items():
        values = []
        for i, part in enumerate(command):
            if part == flag:
                values.append(command[i + 1] if i + 1 < len(command) else None)
            elif part.startswith(flag + "="):
                values.append(part.split("=", 1)[1])
        if values != ["{" + name + "}"]:
            raise VerificationRefusal(f"model command must bind exactly one {flag} to {{{name}}}")


def compare_sample_headers(samples: list[dict[str, Any]], *, scratch: Path) -> list[dict[str, Any]]:
    """Compare declarations and typed attributes except volatile global file_id.

    Full file hashes remain provenance. The Rust comparator separately checks
    stored data words, so a header-only file identity change cannot fail data.
    """
    from .stream_headers_v842 import compare_headers
    comparisons = []
    for sample in samples:
        if Path(sample['hex']).is_file():
            try:
                grade = compare_headers(sample['strict'], sample['hex'], scratch=scratch,
                                        ignored_global_attributes=('file_id',))
            except (OSError, ValueError, RuntimeError) as error:
                grade = dict(passed=False, error=str(error))
            comparisons.append(dict(stream=sample['stream'], steps=sample['steps'], **grade))
    return comparisons


def stream_header_failures(comparisons: list[dict[str, Any]]) -> list[str]:
    """Header acceptance is independent of container hashes; data words gate separately."""
    failures = []
    for grade in comparisons:
        prefix = f"{grade['stream']} steps {grade['steps']}"
        if not grade['passed']:
            failures.append(f"{prefix}: stream header differs from native")
    return failures


def verify(args: argparse.Namespace) -> dict[str, Any]:
    from . import verification_environment as environment
    deck_path = args.deck.resolve()
    deck = read_json(deck_path)
    if deck.get("schema") != "mpas-verification-deck-v1":
        raise VerificationRefusal("unsupported verification deck schema")
    context = identity(deck)
    ranks = process_count(deck.get("process_count"))
    context["process_count"] = ranks
    args._context = context
    if set(deck["inputs"]) != {"namelist", "streams", "init"}:
        raise VerificationRefusal("deck must pin exactly namelist, streams and init")
    inputs = {k: file_ref(v, deck_path.parent) for k, v in deck["inputs"].items()}
    auxiliary = {}
    for name, ref in deck.get("auxiliary_inputs", {}).items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or name in (
                "namelist.atmosphere", "streams.atmosphere", inputs["init"].name):
            raise VerificationRefusal(f"auxiliary input name escapes or replaces a primary input: {name}")
        auxiliary[name] = file_ref(ref, deck_path.parent)
    binary = comparator_path(args.comparator)
    native_path, native = native_manifest(deck["native"], deck_path.parent, deck, binary)
    field_inventory = (bind_native_field_inventory(deck["native_field_inventory"], deck_path.parent,
                       native_path, native, binary) if deck.get("native_field_inventory") else None)
    optimized_path, optimized = (None, None)
    if deck.get("optimized"):
        optimized_path, optimized = native_manifest(deck["optimized"], deck_path.parent, deck, binary)
    work = args.work_root.resolve()
    work.mkdir(parents=True, exist_ok=False)
    args._owns_work = True
    out = work / "hex-output"
    resources = InstalledResources()
    command = deck.get("hex_command", [])
    # Proofs name their mode: a deck without one runs exact (the gates' numerics of before the switch).
    run_command, physics_mode = proof_physics_mode(command, inputs["namelist"])
    run_command = proof_default_backend(run_command)
    run_command, numerics_record = proof_numerics(run_command)
    model_execution = args.hex_output is None
    reference_environment = (read_json(file_ref(deck['strict_environment'], deck_path.parent))
                             if deck.get('strict_environment') else None)
    as_run = (read_json(file_ref(deck['hex_environment'], deck_path.parent))
              if not model_execution and deck.get('hex_environment') else {})
    before_environment = (environment.capture(numerics_record['mode'], run_command)
                          if model_execution else as_run.get('before'))
    environment_check = environment.judge(reference_environment, before_environment, before_environment)
    args._environment = environment_check
    if model_execution and environment_check['failures']:
        # Do not spend a forecast on an arm which cannot be a landing verdict.
        return {'schema': 'mpas-verification-v1', **context, 'passed': False,
                'tier': None, 'verdict': 'ENVIRONMENT', 'model_execution': False,
                'evidence_kind': 'ENVIRONMENT', 'environment': environment_check,
                'failures': environment_check['failures']}
    failed_exit = getattr(args, "failed_model_exit", None)
    if failed_exit is not None and model_execution:
        raise VerificationRefusal("failed-model recovery requires existing --hex-output; it cannot launch or pass a forecast")
    model_error = None
    if model_execution:
        validate_model_command(command)
        sizes = {int(os.environ[name]) for name in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "MV2_COMM_WORLD_SIZE")
                 if name in os.environ}
        if ranks != 1 or (sizes and sizes != {1}):
            raise VerificationRefusal("single-process forecast launcher cannot witness this process count")
        for entry in ET.parse(inputs["streams"]).getroot().iter("file"):
            if entry.get("name") not in auxiliary:
                raise VerificationRefusal(f"stream field list is not pinned for hex: {entry.get('name')}")
        bindings = {k: str(v) for k, v in inputs.items()}
        bindings["out"] = str(out)
        # a relocatable deck names its data relative to itself: {deck_dir}/../data/...
        bindings["deck_dir"] = str(deck_path.parent)
        # ...and the MPAS route's own resources by portable key: {resource:native-library} and the
        # rest of resources_build.RESOURCE_KEYS, resolved from this install's physics-resources build.
        bindings["resource"] = resources
        argv = [part.format_map(bindings) for part in run_command]
        # Run this installation's CLI, even if PATH contains another checkout.
        argv[:1] = [sys.executable, "-m", "woof.hex.cli"]
        run_inputs = work / "run-inputs"
        run_inputs.mkdir()
        for key in ("namelist", "streams"):
            shutil.copyfile(inputs[key], run_inputs / (key + ".atmosphere"))
        (run_inputs / inputs["init"].name).symlink_to(inputs["init"])
        for name, source in auxiliary.items():
            target = run_inputs / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source)
        # The header proof compares every file's container with native MPAS's (compare_sample_headers), so the
        # proof run writes history and diagnostics in the deck's own io_type, not lossless NetCDF-4
        # (woof.hex.history_compression, the default of every other run).
        run_env = {**os.environ, "GPUWM_HISTORY_COMPRESSION": "none"}
        with (work / "hex.log").open("w") as log:
            completed = subprocess.run(argv, cwd=run_inputs, stdout=log, stderr=subprocess.STDOUT, check=False,
                                       env=run_env)
        if completed.returncode:
            model_error = f"hex model door failed ({completed.returncode}); see {work / 'hex.log'}"
        # A failed forecast still counts every absent stream/field/time. Never
        # let an early exit hide missing native words from its ratchet receipt.
        out.mkdir(parents=True, exist_ok=True)
    else:
        out = args.hex_output.resolve()
        if not out.is_dir():
            raise VerificationRefusal("comparison-only hex output directory does not exist")
        if failed_exit is not None:
            model_error = f"outer model job failed ({failed_exit}); recovered partial output cannot pass"
    after_environment = (environment.capture(numerics_record['mode'], run_command)
                         if model_execution else as_run.get('after'))
    environment_check = environment.judge(reference_environment, before_environment, after_environment)
    if not model_execution and (as_run.get('schema') != 'mpas-run-environment-v1'
            or (before_environment or {}).get('pins', {}).get('numerics') != numerics_record['mode']):
        environment_check['failures'].append('as-run environment schema or deck numerics does not match')
        environment_check['verdict'] = 'ENVIRONMENT'
    args._environment = environment_check
    # Recheck inputs after execution: a command cannot alter its reference.
    for k, path in inputs.items():
        if digest(path) != deck["inputs"][k]["sha256"]:
            raise VerificationRefusal(f"model execution changed {k} bytes")
    for name, path in auxiliary.items():
        if digest(path) != deck["auxiliary_inputs"][name]["sha256"]:
            raise VerificationRefusal(f"model execution changed auxiliary input bytes: {name}")
    optimized_outputs = {} if optimized is None else {
        (p["stream"], tuple(p["steps"]), tuple(p["times"])): p for p in optimized["outputs"]}
    if optimized is not None and set(optimized_outputs) != {
        (p["stream"], tuple(p["steps"]), tuple(p["times"])) for p in native["outputs"]}:
        raise VerificationRefusal("optimized native stream/time coverage differs from strict")
    samples = []
    expected_files = set()
    for p in native["outputs"]:
        relative = Path(p["path"])
        candidate = (out / relative).resolve()
        if not candidate.is_relative_to(out):
            raise VerificationRefusal("candidate output path escapes run directory")
        expected_files.add(candidate)
        pair = {"stream": p["stream"], "steps": p["steps"], "times": p["times"],
                "strict": str(native["resolved_output_root"] / relative), "hex": str(candidate)}
        if p.get("clock_witness"):
            pair["clock_witness"] = p["clock_witness"]
        if optimized is not None:
            op = optimized_outputs.get((p["stream"], tuple(p["steps"]), tuple(p["times"])))
            if op is None:
                raise VerificationRefusal("optimized native stream/time coverage differs from strict")
            pair["optimized"] = str(optimized["resolved_output_root"] / op["path"])
        samples.append(pair)
    samples.sort(key=lambda p: (p["steps"][0], p["stream"]))
    request = {"schema": "mpas-stream-compare-v1", "samples": samples}
    if deck.get("diagnostics"):
        diagnostics = dict(deck["diagnostics"])
        diagnostics["area"] = str(file_ref(diagnostics["area"], deck_path.parent))
        prognostics = native.get("prognostic_fields", {}).get(diagnostics["stream"])
        if not prognostics or set(diagnostics["prognostics"]) != set(prognostics):
            raise VerificationRefusal("spread must cover the native manifest's entire prognostic inventory")
        request["diagnostics"] = diagnostics
    write_json(work / "compare-request.json", request)
    raw = work / "compare.json"
    completed = subprocess.run([*comparator_argv(binary), "--request", str(work / "compare-request.json"), "--out", str(raw)],
                               capture_output=True, text=True, check=False)
    if completed.returncode:
        raise VerificationRefusal(f"Rust stream comparison failed: {completed.stderr.strip()}")
    result = read_json(raw)
    header_comparisons = compare_sample_headers(samples, scratch=work / 'headers')
    extras = stream_inventory(binary, out) - expected_files
    output_hashes = {str(p.relative_to(out)): digest(p)
                     for p in sorted(expected_files | extras) if p.is_file()}
    if not model_execution and as_run.get('output_sha256') != output_hashes:
        environment_check['failures'].append('as-run environment is not bound to these output bytes')
        environment_check['verdict'] = 'ENVIRONMENT'
    write_json(work / 'run-environment.json', {
        'schema': 'mpas-run-environment-v1', 'before': before_environment,
        'after': after_environment, 'output_sha256': output_hashes})
    for file in sorted(extras):
        result["fields"].append({"stream": "unlisted", "field": str(file.relative_to(out)), "step": 0,
            "time": "", "status": "extra output stream in hex", "compared_words": 0,
            "differing_words": 1, "max_ulp": None, "first_differing_index": None})
    candidate = aggregate(result["fields"], context)
    baseline = read_json(args.ratchet) if args.ratchet else {"schema": "mpas-ratchet-v1", "entries": []}
    regressions = ratchet_failures(candidate, baseline)
    environment_failures = environment_check['failures']
    # Words identical to native are the proof in any complete strict capture.
    # When the capture differs from the deck's reference, a difference is not
    # attributed to source: raw word counts stay, the regression list and the
    # ratchet do not take them.
    unattributed = environment_check['verdict'] == 'DIFFERS'
    if environment_failures:
        regressions = []
    residue = deck.get("declared_libm_residue", [])
    if not isinstance(residue, list) or any(not isinstance(r, str) or not r.strip() for r in residue):
        raise VerificationRefusal("DECLARED LIBM RESIDUE must name concrete routines")
    reached, failures = tier(result, residue)
    witness = maths_library_witness(deck, deck_path.parent, native_path, optimized_path)
    accepted, acceptance_failures = acceptance_tier(result, reached, witness, field_inventory)
    recorded_missing = any(r["status"] != "compared" for r in result["fields"])
    missing_policy_failures = (recorded_missing_failures(result, context, baseline)
                               if recorded_missing else [])
    if recorded_missing and not missing_policy_failures:
        accepted, acceptance_failures = "BITWISE-WRITTEN-FIELDS", []
    if model_error:
        acceptance_failures.append(model_error)
    acceptance_failures.extend(stream_header_failures(header_comparisons))
    green = accepted is not None and not regressions and not acceptance_failures and not environment_failures
    unattributed_regressions = []
    if unattributed and not green:
        unattributed_regressions, regressions = regressions, []
    receipt = {"schema": "mpas-verification-v1", **context, "tier": reached,
        "acceptance_tier": accepted, "maths_library_witness": witness,
        "all_native_fields_bitwise": reached == "BITWISE",
        "recorded_missing_policy": {"ruling": "integrator-2026-10-13T07:45Z",
            "applied": accepted == "BITWISE-WRITTEN-FIELDS",
            "failures": missing_policy_failures,
            "baseline_sha256": digest(args.ratchet) if args.ratchet else None,
            "process_count": ranks},
        "passed": green,
        "verdict": ('GREEN' if green else 'ENVIRONMENT' if environment_failures or unattributed else 'RED'),
        "environment": environment_check,
        "ratchet_eligible": not environment_failures and not unattributed,
        "unattributed_ratchet_regressions": unattributed_regressions,
        "model_execution": model_execution,
        # The physics mode the model door ran (comparison-only: the mode the deck names or the harness runs).
        "physics_mode": physics_mode,
        # The kernel numerics the model door ran (strict unless the deck names fast: woof.hex.numerics).
        "numerics": numerics_record,
        "evidence_kind": "FORECAST-FAILED" if model_error else ("FORECAST" if model_execution else "COMPARISON-ONLY"),
        "failed_model_exit": failed_exit,
        "declared_libm_residue": residue,
        "failures": ([model_error] if model_error else []) +
                    ([] if accepted == "BITWISE-WRITTEN-FIELDS" else failures),
        "diagnostic_failures": failures,
        "acceptance_failures": acceptance_failures, "ratchet_regressions": regressions,
        "candidate_ratchet": candidate, "comparison": result,
        "header_comparison": header_comparisons,
        "data_words_compared": sum(r["compared_words"] for r in result["fields"]
                                   if r["status"] == "compared"),
        "data_words_differing": sum(r["differing_words"] for r in result["fields"]
                                   if r["status"] == "compared"),
        "provenance": {"deck": str(deck_path), "deck_sha256": digest(deck_path),
            "hex_source": source_provenance(),
            "native_manifest_sha256": digest(native_path),
            "native_cache_manifest_sha256": native.get("cache_manifest_sha256"),
            "native_input_inventory": native.get("native_input_inventory"),
            "native_field_inventory": field_inventory,
            "native_stream_clocks": [{k: p[k] for k in ("stream", "path", "sha256", "steps", "times",
                                                       "time_source", "records", "clock_witness") if k in p}
                                     for p in native["outputs"]],
            "optimized_manifest_sha256": digest(optimized_path) if optimized_path else None,
            "inputs": {k: {"path": str(v), "sha256": digest(v)} for k, v in inputs.items()},
            "auxiliary_inputs": {k: {"path": str(v), "sha256": digest(v)} for k, v in auxiliary.items()},
            "comparator": str(binary), "comparator_sha256": digest(binary),
            "hex_command": command, "hex_command_run": run_command,
            **({"installed_resources": resources.used} if resources.used else {}),
            "hex_output_sha256": {str(p.relative_to(out)): digest(p)
                for p in sorted(expected_files | extras) if p.is_file()}}}
    # Retain small driver evidence before disposable forecast output is removed.
    # A missing sidecar supplies no proof; driver lanes validate its contents.
    sidecars = {}
    for name in ('native-physics-receipt.json', 'physics-alarm-trace.json'):
        source = out / name
        if source.is_file():
            if source.stat().st_size > 10_000_000:
                raise VerificationRefusal('driver evidence sidecar exceeds delivery limit')
            target = args.receipt.parent / 'driver-sidecars' / args.receipt.stem / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            checksum = digest(target)
            receipt['provenance']['hex_output_sha256'][name] = checksum
            sidecars[name] = {'path': str(target.resolve()), 'sha256': checksum}
    if sidecars:
        receipt['provenance']['driver_sidecars'] = sidecars
    runtime_path = out / 'native-physics-receipt.json'
    if runtime_path.is_file():
        runtime = read_json(runtime_path)
        if 'surface_service' in runtime:
            receipt['surface_service'] = runtime['surface_service']
            receipt['provenance']['surface_runtime_receipt_sha256'] = digest(runtime_path)
            receipt['provenance']['surface_input'] = runtime.get('surface_input')
    if args.render:
        if not deck.get("render"):
            raise VerificationRefusal("deck has no Rust-rendered evidence frame bindings")
        receipt["render"] = render_evidence(deck["render"], deck_path.parent, work)
        receipt["passed"] = receipt["passed"] and not receipt["render"]["failures"]
    return receipt


def run(args: argparse.Namespace) -> int:
    start = time.monotonic()
    args._owns_work = False
    args._context = {}
    args._environment = {'verdict': 'ENVIRONMENT', 'failures': ['capture not reached']}
    try:
        receipt = verify(args)
    except (VerificationRefusal, OSError, KeyError, TypeError, ValueError, IndexError,
            ArithmeticError, ET.ParseError) as error:
        receipt = {"schema": "mpas-verification-v1", "passed": False, "tier": None,
                   **args._context, "process_count": args._context.get("process_count"),
                   "failures": [str(error)], "evidence_kind": "REFUSAL"}
    receipt.setdefault('environment', args._environment)
    receipt["wall_seconds"] = time.monotonic() - start
    write_json(args.receipt, receipt)
    # Retire fresh output only after its complete word comparison is recorded.
    # A missing/corrupt reference can fail after a forecast completes; keep
    # that owned output for a retry instead of destroying its only evidence.
    # Existing
    # comparison-only directories and previously existing scratch stay owned
    # by their caller. Rendered maps, requests and logs remain reviewable.
    if args._owns_work and args.hex_output is None and "comparison" in receipt:
        output = args.work_root.resolve() / "hex-output"
        if output.exists():
            shutil.rmtree(output)
    status = (receipt.get("acceptance_tier") if receipt["passed"] else
              ('ENVIRONMENT' if receipt.get('verdict') == 'ENVIRONMENT' else
               'RED' if receipt.get('verdict') == 'RED' else 'REFUSED'))
    words = (f", {receipt['data_words_compared']:,} words compared, {receipt['data_words_differing']:,} differ"
             if "data_words_compared" in receipt else "")
    environment_verdict = receipt.get("environment", {}).get("verdict")
    print(f"verify-mpas: {'GREEN ' if receipt['passed'] else ''}{status}{words}; "
          f"environment {environment_verdict}; receipt {args.receipt}")
    return 0 if receipt["passed"] else 1


def add_verify_mpas_parser(commands: Any) -> None:
    parser = commands.add_parser("verify-mpas", help="compare every native MPAS stream word and enforce the ratchet")
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True, help="fresh owned scratch directory")
    parser.add_argument("--ratchet", type=Path, help="read-only integrator baseline; unknown entries start at zero")
    parser.add_argument("--comparator", type=Path, help="rw_mpas_verify stored-word comparator")
    parser.add_argument("--hex-output", type=Path, help="compare existing output; labels receipt COMPARISON-ONLY")
    parser.add_argument("--failed-model-exit", type=int, choices=range(1, 256),
                        help="count partial output after an interrupted outer job; always FORECAST-FAILED")
    parser.add_argument("--render", action="store_true", help="Rust-render native/hex difference maps bound by the deck")
    parser.set_defaults(handler=run)


def run_cached_dycore(args: argparse.Namespace) -> int:
    if args.cached_deck is None:
        return args.legacy_dycore_handler(args)
    if args.receipt is None:
        print("verify-dycore --cached-deck requires --receipt", file=sys.stderr)
        return 2
    deck = read_json(args.cached_deck)
    if deck.get("role") != "dycore" or deck.get("suite") != "none" or deck.get("physics") != "off":
        print("cached dycore verification requires the suite-none physics-off deck", file=sys.stderr)
        return 2
    args.deck, args.hex_output, args.render = args.cached_deck, None, False
    return run(args)


def gate(args: argparse.Namespace) -> int:
    """Keep disposable gate output under TMPDIR, including on failure."""
    tree = Path(__file__).resolve().parents[2]
    # Keep the fallback outside the Git checkout too. G0 tests create plain
    # engine directories; nesting them in this checkout changes Git discovery
    # and gives them a repository identity they never had (a second-host G0 witness).
    temporary_root = Path(os.environ.get("TMPDIR", str(tree.parents[1] / ".mpas-gate-tmp"))).resolve()
    temporary_root.mkdir(parents=True, exist_ok=True)
    old_tmpdir = os.environ.get("TMPDIR")
    try:
        with tempfile.TemporaryDirectory(prefix="mpas-gate-", dir=temporary_root) as work:
            os.environ["TMPDIR"] = work
            return _gate(args, Path(work))
    finally:
        if old_tmpdir is None:
            os.environ.pop("TMPDIR", None)
        else:
            os.environ["TMPDIR"] = old_tmpdir


def _gate(args: argparse.Namespace, scratch: Path) -> int:
    """Gate index is table work: no meshes, dates or suites are defaults here.

    ``mpas-gate-index-v1`` has gates G1/G2/G3 containing pinned deck refs and
    ``landed_decks`` listing all suite/option deck IDs which G2 must include.
    G1 must contain roles dycore and none-physics. Decks declare duration_steps
    and expected_samples (stream -> step list) to prevent truncated forecasts.
    Every forecast, including cached verify-dycore, runs through the site's GPU
    and build queues when the host names them (woof.hex.site_runner); a public
    checkout runs it directly.
    """
    started = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=False)
    stages = {}
    failures = []
    gate_inputs = {}
    tree = Path(__file__).resolve().parents[2]
    cli = [sys.executable, "-m", "woof.hex.cli"]
    stages_to_run = args.gates if getattr(args, "gates", None) else (
        ["G0", "G1", "G2"] if args.stage == "all" else [args.stage])
    if "G0" in stages_to_run:
        t0 = time.monotonic()
        # G0 has no device. CPU transaction fixtures must not initialize CUDA
        # merely because CuPy is installed and phase memory release defaults on.
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", WOOF_HEX_POOL_RELEASE="0",
                   WOOF_HEX_LMEM_RELEASE="0")
        # A lane may keep TMPDIR inside its checkout. Plain temporary fixtures
        # must not discover the enclosing repository, while fixture repositories
        # with their own .git still work normally.
        env["GIT_CEILING_DIRECTORIES"] = os.pathsep.join(filter(None, (
            env.get("GIT_CEILING_DIRECTORIES"), str(scratch))))
        # lane-env can provide an absolute --basetemp in PYTEST_ADDOPTS.
        # Strip only that setting so nested test runners cannot recreate
        # disposable dumps outside this invocation's owned TMPDIR.
        addopts = iter(shlex.split(env.get("PYTEST_ADDOPTS", "")))
        kept = []
        for option in addopts:
            if option == "--basetemp":
                next(addopts, None)
            elif not option.startswith("--basetemp="):
                kept.append(option)
        env["PYTEST_ADDOPTS"] = shlex.join(kept)
        commands = [[sys.executable, "-m", "pytest", "tests", "-q", "-m", "not gpu and not cuda",
                     "--basetemp", str(scratch / "pytest")],
                    [sys.executable, "tools/check_mpas_citations.py"]]
        outcomes = []
        for i, command in enumerate(commands):
            with (args.out / f"G0-{i}.log").open("w") as log:
                code = subprocess.run(command, cwd=tree, env=env, stdout=log,
                                      stderr=subprocess.STDOUT, check=False).returncode
            outcomes.append(code)
        stages["G0"] = {"exit_codes": outcomes, "wall_seconds": time.monotonic() - t0}
        if any(outcomes):
            failures.append("G0 CPU tests or citation audit failed")
    if not failures and any(s != "G0" for s in stages_to_run):
        try:
            if args.index is None or args.ratchet is None:
                raise VerificationRefusal("G1-G3 require the pinned deck index and integrator ratchet")
            index = read_json(args.index)
            if index.get("schema") != "mpas-gate-index-v1":
                raise VerificationRefusal("unsupported gate index schema")
            baseline = read_json(args.ratchet)
            if baseline.get("schema") != "mpas-ratchet-v1" or not isinstance(baseline.get("entries"), list):
                raise VerificationRefusal("unsupported integrator ratchet schema")
            gate_inputs = {"index_sha256": digest(args.index), "ratchet_sha256": digest(args.ratchet)}
            for stage in stages_to_run:
                if stage == "G0":
                    continue
                references = index["gates"][stage]
                if not references:
                    raise VerificationRefusal(f"{stage} has no decks; an empty gate cannot pass")
                loaded = [(file_ref(ref, args.index.parent), None) for ref in references]
                loaded = [(path, read_json(path)) for path, _ in loaded]
                ids = [deck["id"] for _, deck in loaded]
                if len(ids) != len(set(ids)):
                    raise VerificationRefusal("gate index repeats a deck")
                if stage == "G1" and not {"dycore", "none-physics"}.issubset({d.get("role") for _, d in loaded}):
                    raise VerificationRefusal("G1 lacks cached dycore or suite-none physics-on proof")
                if stage == "G2" and not set(index["landed_decks"]).issubset(ids):
                    raise VerificationRefusal("G2 omits a landed suite or option deck")
                begin = time.monotonic()
                results = []
                for number, (path, deck) in enumerate(loaded):
                    ranks = process_count(deck.get("process_count"))
                    if not deck.get("expected_samples") or not deck.get("duration_steps"):
                        raise VerificationRefusal("gate deck lacks forecast duration or complete output schedule")
                    if stage == "G1" and deck.get("role") == "none-physics" and (
                        deck.get("suite") != "none" or deck.get("physics") != "on"):
                        raise VerificationRefusal("G1 none-physics deck does not enable physics")
                    receipt_path = args.out / f"{stage}-{number}.json"
                    door = "verify-dycore" if deck.get("role") == "dycore" and stage == "G1" else "verify-mpas"
                    flag = "--cached-deck" if door == "verify-dycore" else "--deck"
                    command = cli + [door, flag, str(path), "--receipt", str(receipt_path),
                        "--work-root", str(scratch / f"{stage}-{number}-scratch"), "--ratchet", str(args.ratchet)]
                    if args.comparator:
                        command += ["--comparator", str(args.comparator)]
                    with (args.out / f"{stage}-{number}.log").open("w") as log:
                        code = subprocess.run(site_runner.gpu_command(command, args.minutes, proof=True, build=True),
                            stdout=log, stderr=subprocess.STDOUT, check=False).returncode
                    recovery_code = None
                    if code and not receipt_path.is_file():
                        # the GPU queue's timeout may kill the verifier itself. Count
                        # every absent native word on CPU afterwards; an outer
                        # failure must never disappear as an empty gate receipt.
                        partial = scratch / f"{stage}-{number}-scratch" / "hex-output"
                        partial.mkdir(parents=True, exist_ok=True)
                        recovery = cli + ["verify-mpas", "--deck", str(path),
                            "--receipt", str(receipt_path), "--hex-output", str(partial),
                            "--work-root", str(scratch / f"{stage}-{number}-recovery"),
                            "--failed-model-exit", str(code if 1 <= code <= 255 else min(255, 128 + abs(code))),
                            "--ratchet", str(args.ratchet)]
                        if args.comparator:
                            recovery += ["--comparator", str(args.comparator)]
                        with (args.out / f"{stage}-{number}-recovery.log").open("w") as log:
                            recovery_code = subprocess.run(recovery, stdout=log,
                                stderr=subprocess.STDOUT, check=False).returncode
                        if receipt_path.is_file():
                            shutil.rmtree(partial)
                    model_log = scratch / f"{stage}-{number}-scratch" / "hex.log"
                    if model_log.is_file():
                        shutil.copyfile(model_log, args.out / f"{stage}-{number}-model.log")
                    receipt = read_json(receipt_path) if receipt_path.is_file() else {}
                    errors = gate_receipt_failures(receipt, deck, stage, baseline)
                    results.append({"deck": deck["id"], "process_count": ranks,
                                    "exit_code": code, "recovery_exit_code": recovery_code,
                                    "receipt": str(receipt_path), "failures": errors})
                    if code or errors:
                        failures.append(f"{stage}: {deck['id']} refused")
                        break
                stages[stage] = {"wall_seconds": time.monotonic() - begin, "decks": results}
                if failures:
                    break
            if digest(args.index) != gate_inputs["index_sha256"] or digest(args.ratchet) != gate_inputs["ratchet_sha256"]:
                failures.append("gate index or integrator ratchet changed during the proof")
        except (VerificationRefusal, OSError, KeyError, TypeError, ValueError) as error:
            failures.append(str(error))
    write_json(args.out / "gate.json", {"schema": "mpas-gate-v1", "passed": not failures,
        "stages": stages, "failures": failures, "inputs": gate_inputs,
        "process_counts": sorted({r.get("process_count") for s in stages.values()
                                  for r in s.get("decks", []) if isinstance(r.get("process_count"), int)}),
        "wall_seconds": time.monotonic() - started})
    return 1 if failures else 0


def gate_receipt_failures(receipt: dict[str, Any], deck: dict[str, Any], stage: str,
                          baseline: dict[str, Any] | None = None) -> list[str]:
    failures = []
    if receipt.get("passed") is not True or receipt.get("evidence_kind") != "FORECAST":
        failures.append("gate requires a passing real-model forecast receipt")
    if receipt.get("failed_model_exit") is not None:
        failures.append("interrupted outer model job cannot satisfy the forecast gate")
    ranks = deck.get("process_count")
    if isinstance(ranks, bool) or not isinstance(ranks, int) or ranks < 1 or isinstance(receipt.get("process_count"), bool) or receipt.get("process_count") != ranks:
        failures.append("gate deck and receipt lack the same fixed process count")
    fields = receipt.get("comparison", {}).get("fields", [])
    bitwise = bool(fields) and all(r["status"] == "compared" and r["differing_words"] == 0 for r in fields)
    recorded = False
    if fields and baseline is not None:
        context = {**identity(deck), "process_count": ranks}
        if any(receipt.get(k) != context[k] for k in context):
            failures.append("forecast receipt context differs from the pinned gate deck")
        # Recompute from raw field rows: neither a claimed tier nor a forged
        # candidate ratchet can make an absent/perturbed field pass the gate.
        regressions = ratchet_failures(aggregate(fields, context), baseline)
        failures.extend(regressions)
        if stage in ("G1", "G2") and any(r["status"] != "compared" for r in fields):
            policy_errors = recorded_missing_failures(receipt["comparison"], context, baseline)
            recorded = not policy_errors and receipt.get("acceptance_tier") == "BITWISE-WRITTEN-FIELDS"
            failures.extend(policy_errors)
    if stage == "G1" and not recorded and receipt.get("tier") != "BITWISE":
        failures.append("G1 requires zero differing words")
    if not bitwise and not recorded:
        witness = receipt.get("maths_library_witness")
        if (stage == "G1" or receipt.get("acceptance_tier") != "INSIDE-NATIVE-SPREAD" or
                not witness or witness.get("process_count") != ranks):
            failures.append("gate requires zero words or witnessed maths-library spread; diagnostic grades cannot pass")
    observed: dict[str, set[int]] = {}
    for row in fields:
        observed.setdefault(row["stream"], set()).add(row["step"])
    expected = {name: set(steps) for name, steps in deck["expected_samples"].items()}
    if observed != expected:
        failures.append("stream output schedule differs from the complete gate deck")
    if not fields or max((r["step"] for r in fields), default=-1) != deck["duration_steps"]:
        failures.append("forecast did not reach the deck's final step")
    return failures


def gate_list(value: str) -> list[str]:
    stages = value.split(",")
    if not stages or len(set(stages)) != len(stages) or any(s not in ("G0", "G1", "G2", "G3") for s in stages):
        raise argparse.ArgumentTypeError("gates must be a nonempty comma-separated list of distinct G0,G1,G2,G3")
    return stages


def build_headline(args: argparse.Namespace) -> int:
    """Build a routed deck from an existing native-physics gate context.

    This copies orchestration metadata, never forecast values. Existing
    headline counters survive regeneration, including retired missing entries.
    """
    import copy
    from .physics_suites import load_table

    index, baseline = read_json(args.index), read_json(args.ratchet)
    parents = [(file_ref(ref, args.index.parent), ref)
               for ref in index["gates"]["G2"]]
    matches = [(path, read_json(path)) for path, _ in parents
               if read_json(path)["id"] == args.parent_deck]
    if len(matches) != 1 or not args.headline_id or args.headline_id == args.parent_deck:
        raise VerificationRefusal("headline requires one indexed parent and a distinct stable id")
    parent_path, parent = matches[0]
    if parent["schema"] != "mpas-verification-deck-v1" or parent["physics"] != "on":
        raise VerificationRefusal("headline parent must be a physics-on verification deck")
    if "--physics" in parent["hex_command"]:
        raise VerificationRefusal("headline parent must have no backend overlay to replace")
    # A parent's --physics-mode (exact, as every proof names it) is carried into the headline unchanged;
    # a parent that names none runs exact through verify (proof_physics_mode).  Malformed flags refuse.
    command_physics_mode(parent["hex_command"])
    process_count(parent["process_count"])
    # Parent data paths become absolute because the generated JSON may live
    # outside the original directory. Absolute index refs stay unchanged.
    deck = copy.deepcopy(parent)
    for group in ("inputs", "auxiliary_inputs", "hex_support_inputs"):
        for ref in deck.get(group, {}).values():
            ref["path"] = str(file_ref(ref, parent_path.parent))
    deck["native"]["path"] = str(file_ref(deck["native"], parent_path.parent))
    spec = load_table()["mpasphys_v842"]["slots"]
    selectors = {row["backend_selector"]: slot for slot, row in spec.items()}
    backends = {name: "native-cpu" for name in selectors}
    landed = {}
    slots_text = args.slots.read_bytes()
    for number, line in enumerate(slots_text.decode("utf-8").splitlines(), 1):
        value = line.split("#", 1)[0].strip()
        if not value:
            continue
        pair = value.split("=")
        if len(pair) != 2:
            raise VerificationRefusal(f"landed-slot line {number} requires slot=hex-cuda")
        name, backend = (part.strip() for part in pair)
        if name not in selectors or backend != "hex-cuda" or name in landed:
            raise VerificationRefusal(f"landed-slot line {number} has an unknown, repeated or non-CUDA slot")
        landed[name] = backend
    backends.update(landed)
    # A slot the parent's namelist leaves inactive runs nothing (convection_permitting: config_ngw_scheme and
    # config_oml1d are false, mpas_atmphys_control.F:310-319), so it takes the route default, not a native-cpu
    # label.  When no active slot is left on the CPU the route is the ported framework
    # (runtime_v842.framework_only), which loads no native library: the headline then names none.
    from .mpasphys.driver_v842 import resolve_namelist
    from .mpasphys.runtime_v842 import framework_only
    namelist = file_ref(parent["inputs"]["namelist"], parent_path.parent).read_text()
    def resolved():
        return resolve_namelist(namelist, backends={selectors[k]: v for k, v in backends.items()})
    config = resolved()
    backends.update({name: "hex-cuda" for name, slot in selectors.items() if not config.active(slot)})
    if framework_only(resolved()):
        command = deck["hex_command"]
        while "--native-physics-library" in command:
            at = command.index("--native-physics-library")
            del command[at:at + 2]
        deck.get("hex_support_inputs", {}).pop("native-library", None)
    slots_sha = hashlib.sha256(slots_text).hexdigest()
    if digest(args.slots) != slots_sha:
        raise VerificationRefusal("landed-slot file changed during deck generation")
    overlay = ",".join(f"{name}='{value}'" for name, value in sorted(backends.items()))
    deck.update(id=args.headline_id, role="headline", headline_backends={selectors[k]: v for k, v in backends.items()})
    deck["hex_command"] += ["--physics", overlay]
    resources = getattr(args, "physics_resources", None)
    if resources is None:
        previous = [read_json(path) for path, _ in parents
                    if read_json(path)["id"] == args.headline_id]
        if previous and "physics-resources" in previous[0].get("hex_support_inputs", {}):
            for name, ref in previous[0]["hex_support_inputs"].items():
                if name.startswith("physics-resource"):
                    file_ref(ref, args.index.parent)
            resources = file_ref(previous[0]["hex_support_inputs"]["physics-resources"],
                                 args.index.parent)
    resource_bytes = resources.read_bytes() if resources is not None else None
    resource_sha = hashlib.sha256(resource_bytes).hexdigest() if resource_bytes is not None else None
    resource_refs = {}
    if resources is not None:
        def authenticate_resource(value):
            if isinstance(value, dict):
                if "path" in value and "sha256" in value:
                    path = file_ref(value, resources.parent)
                    if not Path(value["path"]).is_absolute():
                        raise VerificationRefusal("physics resource paths must be absolute")
                    resource_refs[str(path)] = {"path": str(path), "sha256": digest(path)}
                for key, child in value.items():
                    if key == "table_directory":
                        directory = Path(child)
                        if not directory.is_absolute() or not directory.is_dir():
                            raise VerificationRefusal("physics resource table directory must be absolute and exist")
                        for path in sorted(directory.iterdir()):
                            if path.is_file():
                                resource_refs[str(path)] = {"path": str(path.resolve()), "sha256": digest(path)}
                    else:
                        authenticate_resource(child)
            elif isinstance(value, list):
                for child in value:
                    authenticate_resource(child)
        authenticate_resource(json.loads(resource_bytes))
        if digest(resources) != resource_sha:
            raise VerificationRefusal("physics resources changed during deck generation")
        deck.setdefault("hex_support_inputs", {}).update({
            f"physics-resource-{number}": ref for number, ref in enumerate(resource_refs.values())})
    deck.setdefault("hex_support_inputs", {})["headline-slots"] = {
        "path": str(args.slots.resolve()), "sha256": slots_sha}
    original_entries = copy.deepcopy(baseline["entries"])
    existing = [entry for entry in baseline["entries"] if entry["deck"] == args.headline_id]
    parent_entries = [entry for entry in baseline["entries"]
                      if all(entry[k] == identity(parent)[k] for k in KEY[:4])]
    if not parent_entries or any(entry.get("process_count") != parent["process_count"] for entry in parent_entries):
        raise VerificationRefusal("headline parent lacks its same-process authorized ratchet inventory")
    if existing:
        if any(any(entry[k] != identity(deck)[k] for k in KEY[:3])
               or entry.get("process_count") != deck["process_count"] for entry in existing):
            raise VerificationRefusal("headline id already belongs to a different native context")
        # Never fill absent entries or raise a retired allowance on regeneration.
    else:
        for entry in parent_entries:
            clone = copy.deepcopy(entry)
            clone["deck"] = args.headline_id
            baseline["entries"].append(clone)
    baseline["headline_authority"] = {
        "rule": "Integrator 2026-10-14 03:05Z: headline shares the original M0 native publication and recorded-missing inventory; no counter may grow.",
        "parent": {"path": str(parent_path), "sha256": digest(parent_path)},
        "native": copy.deepcopy(deck["native"]), "process_count": deck["process_count"],
        "slots": copy.deepcopy(deck["hex_support_inputs"]["headline-slots"]),
    }
    assert baseline["entries"][:len(original_entries)] == original_entries
    args.out.mkdir(parents=True, exist_ok=True)
    snapshot = args.out / f"headline-slots-{slots_sha}.txt"
    snapshot.write_bytes(slots_text)
    if resource_bytes is not None:
        resource_snapshot = args.out / f"physics-resources-{resource_sha}.json"
        resource_snapshot.write_bytes(resource_bytes)
        deck["hex_command"] += ["--physics-resources", str(resource_snapshot.resolve())]
        deck["hex_support_inputs"]["physics-resources"] = {
            "path": str(resource_snapshot.resolve()), "sha256": resource_sha}
    deck["hex_support_inputs"]["headline-slots"]["path"] = str(snapshot.resolve())
    baseline["headline_authority"]["slots"] = copy.deepcopy(deck["hex_support_inputs"]["headline-slots"])
    deck_path = args.out / "headline-deck.json"
    write_json(deck_path, deck)
    headline_ref = {"path": str(deck_path.resolve()), "sha256": digest(deck_path)}
    index["gates"]["G2"] = [ref for path, ref in parents if read_json(path)["id"] != args.headline_id] + [headline_ref]
    if args.headline_id not in index["landed_decks"]:
        index["landed_decks"].append(args.headline_id)
    index["scope"] = "Existing verification decks plus the landed-GPU headline; native publications and recorded inventory unchanged."
    for refs in index["gates"].values():
        for ref in refs:
            if not Path(ref["path"]).is_absolute():
                ref["path"] = str(file_ref(ref, args.index.parent))
    write_json(args.out / "mpas-gate-index.json", index)
    write_json(args.out / "ratchet.json", baseline)
    write_json(args.out / "builder.json", {"schema": "mpas-headline-builder-v1", "process_count": deck["process_count"],
        "parent_deck": parent["id"], "headline_deck": deck["id"], "backends": deck["headline_backends"],
        "slots": deck["hex_support_inputs"]["headline-slots"], "existing_headline_inventory_preserved": bool(existing)})
    return 0


def gate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MPAS landing gate")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--stage", choices=("G0", "G1", "G2", "G3", "all"))
    selection.add_argument("--gates", type=gate_list)
    selection.add_argument("--build-headline", action="store_true")
    receipts = Path(__file__).resolve().parents[3] / "receipts"
    parser.add_argument("--index", type=Path,
                        default=Path(os.environ.get("MPAS_GATE_INDEX", str(receipts / "mpas-gate-index.json"))))
    parser.add_argument("--ratchet", type=Path,
                        default=Path(os.environ.get("MPAS_RATCHET", str(receipts / "ratchet.json"))))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--comparator", type=Path)
    parser.add_argument("--slots", type=Path, help="integrator's landed slot=hex-cuda list")
    parser.add_argument("--physics-resources", type=Path,
                        help="certified CUDA resource packet; retained from the existing headline on regeneration")
    parser.add_argument("--parent-deck", help="existing G2 native-physics deck id")
    parser.add_argument("--headline-id", help="stable headline deck id, preserved across landing batches")
    parser.add_argument("--minutes", type=int, choices=range(1, 121), default=120,
                        help="whole-forecast proof cap passed to the site GPU queue (1-120 minutes)")
    return parser


if __name__ == "__main__":
    arguments = gate_parser().parse_args()
    if arguments.build_headline:
        if not all((arguments.slots, arguments.parent_deck, arguments.headline_id)):
            raise SystemExit("--build-headline requires --slots, --parent-deck and --headline-id")
        raise SystemExit(build_headline(arguments))
    raise SystemExit(gate(arguments))
