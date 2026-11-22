#!/usr/bin/env python3
"""Catalog an experimental v10 trace without asserting a complete census.

Physical driver allocations, named CuPy views and argument-word pointer
candidates are separate observations. Neither candidate pointer matches nor
sampled object names prove access mode, touched extents, or carried state. A
complete census additionally requires native replay and the poison tests.
"""
# SPDX-License-Identifier: Apache-2.0
import argparse
import bisect
import collections
import gzip
import json
from pathlib import Path


def rows(path, partial):
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                yield json.loads(line)
    except EOFError:
        if not partial:
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--observer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", choices=("hrrrv5", "convection_permitting"), required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    allocations = []
    live = {}
    bases = []
    boundaries = []
    step = 0
    counts = collections.Counter()
    upload_bytes = 0
    capture_errors = []
    last_generic = None

    def locate(ptr):
        if not isinstance(ptr, int):
            return None
        i = bisect.bisect_right(bases, ptr) - 1
        if i < 0:
            return None
        entry = live[bases[i]]
        if ptr - entry["recorded_base"] >= entry["bytes"]:
            return None
        return entry

    def candidate(ptr, label, size=0):
        entry = locate(ptr)
        if entry is None:
            counts[label + "_unmatched"] += 1
            return
        offset = ptr - entry["recorded_base"]
        if size and size > entry["bytes"] - offset:
            counts[label + "_out_of_bounds"] += 1
            return
        key = "candidate_argument_words" if label == "argument" else "explicit_transfer_ranges"
        entry[key] += 1
        entry["candidate_offset_min"] = min(entry.get("candidate_offset_min", offset), offset)
        entry["candidate_offset_max"] = max(entry.get("candidate_offset_max", offset), offset)
        if step and (not entry["observed_use_steps"] or entry["observed_use_steps"][-1] != step):
            entry["observed_use_steps"].append(step)
        counts[label + "_matched"] += 1

    for row in rows(args.driver, args.allow_incomplete):
        kind = row["type"]
        counts[kind] += 1
        if row.get("status", 0):
            capture_errors.append({"sequence": row.get("sequence"), "type": kind, "status": row["status"]})
        if kind in ("MemAlloc", "MemAllocAsync") and not row.get("status", 0):
            base, size = row["a"], row["count"]
            if base in live:
                capture_errors.append({"sequence": row["sequence"], "duplicate_live_base": base})
                continue
            entry = {"id": len(allocations) + 1, "recorded_base": base, "bytes": size,
                     "owning_field": None, "dtype": None, "shape": None, "stagger": None,
                     "memory": "device", "lifetime": None, "carried_across_steps": None,
                     "owner_status": "physical root only; may contain multiple unresolved owners",
                     "allocated_clock_ns": row["clock_ns"], "freed_clock_ns": None,
                     "initial_sha256": None, "boundary_sha256": {},
                     "boundary_live_steps": [], "observed_use_steps": [],
                     "candidate_argument_words": 0, "explicit_transfer_ranges": 0,
                     "named_view_candidates": [], "carried_classification": "unresolved"}
            allocations.append(entry)
            bisect.insort(bases, base)
            live[base] = entry
        elif kind in ("MemFree", "MemFreeAsync") and not row.get("status", 0):
            base = row["a"]
            if base == 0:
                # This is still emitted as a driver call by the replay plan;
                # it has no physical allocation lifetime to close.
                counts["null_free_calls"] += 1
            elif base not in live:
                capture_errors.append({"sequence": row["sequence"], "free_unknown_base": base})
            else:
                live.pop(base)["freed_clock_ns"] = row["clock_ns"]
                bases.pop(bisect.bisect_left(bases, base))
        elif kind == "InitialAllocation":
            entry = live.get(row["base"])
            if entry is None or row["bytes"] != entry["bytes"] or row["captured"] != row["bytes"]:
                capture_errors.append({"sequence": row["sequence"], "bad_initial_allocation": row["base"]})
            else:
                entry["initial_sha256"] = row["sha256"]
        elif kind == "BoundaryAllocationDigest":
            entry = live.get(row["base"])
            digest = row.get("sha256", "")
            digest_step = row.get("step")
            if (entry is None or row["bytes"] != entry["bytes"]
                    or not isinstance(digest_step, int) or digest_step < 1
                    or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
                    or str(digest_step) in entry["boundary_sha256"]):
                capture_errors.append({"sequence": row["sequence"], "bad_boundary_digest": row.get("base")})
            else:
                entry["boundary_sha256"][str(digest_step)] = digest
        elif kind == "Boundary":
            step = row["a"]
            boundaries.append({"step": step, "clock_ns": row["clock_ns"], "sequence": row["sequence"]})
            for entry in live.values():
                entry["boundary_live_steps"].append(step)
        elif kind == "LaunchKernel":
            for param in row.get("parameters", []):
                text = param.get("bytes_hex")
                if text is None:
                    counts["missing_parameter_bytes"] += 1
                    continue
                value = bytes.fromhex(text)
                for offset in range(0, len(value) - 7, 8):
                    candidate(int.from_bytes(value[offset:offset + 8], "little"), "argument")
        elif kind == "GenericCopyKind":
            last_generic = row
        elif kind in ("Memcpy", "MemcpyAsync"):
            if last_generic is None or last_generic["count"] != row["count"]:
                counts["unresolved_generic_copy"] += 1
            else:
                if last_generic["a"] == 2:
                    candidate(row["b"], "transfer", row["count"])
                if last_generic["b"] == 2:
                    candidate(row["a"], "transfer", row["count"])
            last_generic = None
        elif kind.startswith("MemcpyHtoD") or kind.startswith("MemsetD"):
            element = row.get("extra", 1) if kind.startswith("MemsetD") else 1
            candidate(row["a"], "transfer", row["count"] * element)
        elif kind.startswith("MemcpyDtoH"):
            candidate(row["b"], "transfer", row["count"])
        elif kind.startswith("MemcpyDtoD"):
            candidate(row["a"], "transfer", row["count"])
            candidate(row["b"], "transfer", row["count"])
        elif kind == "HostUploadBytes" and row.get("payload"):
            upload_bytes += row["bytes"]

    # Metadata samples can name subranges of a physical pool allocation. A
    # time-qualified containment match is still only an owner candidate.
    def named_samples():
        for row in rows(args.observer / "state.jsonl.gz", args.allow_incomplete):
            for group in ("fields", "saved_fields"):
                for name, view in row.get(group, {}).items():
                    yield row["clock_ns"], group + "." + name, view
        for row in rows(args.observer / "boundaries.jsonl.gz", args.allow_incomplete):
            outer = row.get("boundary") == "outer_after_step"
            for name, view in row.get("fields", {}).items():
                yield row["clock_ns"], ("outer_fields." if outer else "boundary_fields.") + name, view
            for name, view in row.get("saved_fields", {}).items():
                yield row["clock_ns"], "outer_saved_fields." + name, view
            for name, view in row.get("physics_pools", {}).items():
                yield row["clock_ns"], ("outer_physics_pools." if outer else "physics_pools.") + name, view

    # A physical address can be recycled. Keep the full time-qualified index,
    # rather than choosing the last allocation with a matching base. The
    # prefix maxima permit overlapping ranges without a quadratic full scan.
    by_base = sorted(allocations, key=lambda entry: entry["recorded_base"])
    sorted_bases = [entry["recorded_base"] for entry in by_base]
    prefix_end = []
    for entry in by_base:
        end = entry["recorded_base"] + entry["bytes"]
        prefix_end.append(max(prefix_end[-1], end) if prefix_end else end)

    for when, name, view in named_samples():
        ptr = view["base"] + view["offset"]
        matches = []
        i = bisect.bisect_right(sorted_bases, ptr) - 1
        while i >= 0 and prefix_end[i] > ptr:
            entry = by_base[i]
            if (ptr < entry["recorded_base"] + entry["bytes"]
                    and entry["allocated_clock_ns"] <= when
                    and (entry["freed_clock_ns"] is None or when < entry["freed_clock_ns"])):
                matches.append(entry)
            i -= 1
        if len(matches) != 1:
            counts["named_view_ambiguous"] += 1
            continue
        entry = matches[0]
        value = {"name": name,
                 "offset": ptr - entry["recorded_base"],
                 "dtype": view["dtype"], "shape": view["shape"], "strides": view["strides"],
                 "observed_lease_id": view.get("allocation_id"), "stagger": "unresolved"}
        if value not in entry["named_view_candidates"]:
            entry["named_view_candidates"].append(value)
        counts["named_view_samples"] += 1

    boundary_steps = [item["step"] for item in boundaries]
    result = {
        "schema": "ad.binder.census/v1", "config": args.config, "rank": 0, "ranks": 1,
        "complete": False,
        "draft_note": "Allocation entries are physical roots, not yet resolved owners. Null fields are unknown; this draft intentionally does not validate as a complete census.",
        "source_pin": "16251589f19aec0012fd164d1ea060126208aed1",
        "recorded_outer_boundaries": boundaries,
        "recorded_72_steps": boundary_steps == list(range(1, 74)),
        "poison_test": "not run", "native_replay": "not established",
        "classifications": "unresolved; pointer-word containment is only a candidate",
        "initial_state": "SHA256 names compressed initial allocation payloads in local capture scratch",
        "per_step_upload_payload_bytes": upload_bytes,
        "counts": dict(sorted(counts.items())), "capture_errors": capture_errors,
        "allocations": allocations,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps({"physical_allocations": len(allocations), "boundary_markers": len(boundaries),
                      "errors": len(capture_errors), "counts": dict(sorted(counts.items()))}, sort_keys=True))


if __name__ == "__main__":
    main()
