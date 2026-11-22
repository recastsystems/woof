# SPDX-License-Identifier: Apache-2.0
"""ad-record: the per-recording allocation map and scale-table slots of a sealed plan package.

Allocation numbers belong to a recording: hex's pool layout differs run to run, so pkg18 owns theta_m at 65/351 and
pkg18m2 at 67/368 (two recordings of the same 24 h window). The map is therefore made from the recording itself, never carried over:

* the observer's state.jsonl (the forward's own named dycore fields at the first boundary, with device base address,
  dtype, shape and strides) is joined to the plan's catalog by recorded base address -> the dycore allocation id;
* every other allocation with the same byte size and the same initial SHA-256 is a copy the forward made of it at
  start (the physics-pool copies: rho 13 -> 367, theta_m 67 -> 368, scalars 62 -> 378 on pkg18m2). A constant
  initial content (all zeros, e.g. w at a cold start) proves nothing, so its candidates are listed as unproven.

Output (in the package): alloc-map.json (schema ad-record.alloc-map/v1) and scale-table.json (one slot per mapped
allocation, power-of-two exponent 0; runner-lib's scale table fills the exponents). Both carry the plan sha256.

Usage: make_alloc_map.py PACKAGE_DIR OBSERVER_DIR
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

#: hex's scalars order on the RRFS hrrrv5 route (hex's default scalar names plus the RRFS extras)
SPECIES = ("qv", "qc", "qr", "qi", "qs", "qg", "nc", "ni", "nr", "ng", "nifa", "nwfa", "volg")


_PLAN: dict = {}


def _payload(pkg: Path, alloc: dict) -> bytes:
    entry = alloc["initial"]
    path = pkg / entry["path"]
    return gzip.open(path).read() if entry.get("encoding") == "gzip" else path.read_bytes()


def padded_copies(pkg: Path, root_id: int, offset: int, shape) -> dict:
    """Copies of a padded field: on a limited-area mesh the dycore's arrays carry MPAS's padding column (nCells + 1,
    nEdges + 1) and the physics pool's start copies hold the mesh's own columns, so their bytes and SHA-256 differ
    from the root's. A copy is an allocation whose initial payload begins with the root's float32 words with the
    last column of every row dropped. Returns {allocation id: the copy's shape}."""
    import numpy as np
    if pkg not in _PLAN:
        doc = json.loads((pkg / "plan.json").read_text())
        _PLAN[pkg] = {int(a["id"]): a for a in doc["allocations"]}
    allocs = _PLAN[pkg]
    root = allocs.get(root_id)
    if root is None or not root.get("initial") or len(shape) < 2:
        return {}
    count = int(np.prod(shape))
    arr = np.frombuffer(_payload(pkg, root), "<f4", count=count, offset=offset).reshape(shape)
    trim = np.ascontiguousarray(arr[..., :-1]).tobytes()
    limit = int(root.get("bytes") or 4 * count) + 4096
    return {i: [*shape[:-1], int(shape[-1]) - 1] for i, a in sorted(allocs.items())
            if i != root_id and a.get("initial") and len(trim) <= int(a.get("bytes") or 0) <= limit
            and _payload(pkg, a)[:len(trim)] == trim}


def main(argv) -> int:
    pkg, obs = Path(argv[1]), Path(argv[2])
    plan_sha = hashlib.sha256((pkg / "plan.json").read_bytes()).hexdigest()
    catalog = json.loads((pkg / "catalog-a.json").read_text())
    allocs = catalog["allocations"]
    by_base = {int(a["recorded_base"]): a for a in allocs if a.get("recorded_base") is not None}
    with gzip.open(obs / "state.jsonl.gz", "rt") as f:
        first = json.loads(f.readline())
    named = {}
    for group in ("fields", "saved_fields"):
        for name, view in first.get(group, {}).items():
            named[name] = (group, view)
    out = {"schema": "ad-record.alloc-map/v1", "plan_sha256": plan_sha, "boundary": int(first["step"]),
           "source": "observer state.jsonl first boundary joined to catalog-a.json by recorded base; copies by equal "
                     "bytes + initial_sha256", "species_order": list(SPECIES),
           "species_order_source": "hex's default scalar names plus the RRFS extras", "fields": {}}
    refused = []
    for name, (group, view) in sorted(named.items()):
        base = int(view["base"])
        root = by_base.get(base - int(view.get("offset", 0)))
        if root is None:
            refused.append(name)
            continue
        zero = hashlib.sha256(bytes(int(root["bytes"]))).hexdigest()
        copies = [a["id"] for a in allocs if a["id"] != root["id"] and a["bytes"] == root["bytes"]
                  and a.get("initial_sha256") == root.get("initial_sha256")]
        constant = root.get("initial_sha256") == zero
        copy_shapes = {}
        if not copies and not constant:
            padded = padded_copies(pkg, int(root["id"]), int(view.get("offset", 0)), view["shape"])
            copies = list(padded)
            copy_shapes = {str(i): s for i, s in padded.items()}
        out["fields"][name] = {
            "group": group, "dtype": view["dtype"], "shape": view["shape"], "strides": view["strides"],
            "offset": int(view.get("offset", 0)), "bytes": int(root["bytes"]), "allocation": int(root["id"]),
            "copies": [] if constant else copies, "copies_unproven": copies if constant else [],
            "initial_sha256": root.get("initial_sha256")}
        if copy_shapes:
            out["fields"][name]["copy_shapes"] = copy_shapes
    if refused:
        print(f"DEFECT: ad-record alloc-map: named fields {refused} have no catalog allocation at their base",
              file=sys.stderr)
        return 3
    # The dycore state moves to new allocations every step (time-level swaps and pool reuse), so the map carries the
    # owner of every named field at EVERY recorded boundary: the catalog root live at the boundary's clock that holds
    # the field's base address. The physics-pool copies are persistent allocations (ids above).
    live = sorted(allocs, key=lambda a: int(a.get("recorded_base") or 0))
    boundaries = {}
    with gzip.open(obs / "state.jsonl.gz", "rt") as f:
        for line in f:
            rec = json.loads(line)
            clock = int(rec["clock_ns"])
            row = {}
            for group in ("fields", "saved_fields"):
                for name, view in rec.get(group, {}).items():
                    base = int(view["base"])
                    hit = [a for a in live if a.get("recorded_base") is not None
                           and int(a["recorded_base"]) <= base < int(a["recorded_base"]) + int(a["bytes"])
                           and int(a.get("allocated_clock_ns") or 0) <= clock
                           and (a.get("freed_clock_ns") is None or int(a["freed_clock_ns"]) > clock)]
                    if len(hit) != 1:
                        refused.append(f"{name}@{rec['step']}({len(hit)} roots)")
                        continue
                    row[name] = {"allocation": int(hit[0]["id"]), "offset": base - int(hit[0]["recorded_base"])}
            boundaries[str(rec["step"])] = row
    if refused:
        print(f"DEFECT: ad-record alloc-map: per-boundary owners not unique: {refused[:12]}", file=sys.stderr)
        return 3
    out["boundaries"] = boundaries
    out["boundary_note"] = ("boundaries[s][field] = the allocation (and byte offset) holding the field at recorded boundary "
                            "s (state.jsonl step s; step 1 = t0). 'fields'[f].allocation is boundary 1's.")
    (pkg / "alloc-map.json").write_text(json.dumps(out, indent=1))
    slots = []
    for name, entry in out["fields"].items():
        for alloc in [entry["allocation"], *entry["copies"]]:
            slots.append({"allocation": alloc, "field": name, "exponent": 0})
    (pkg / "scale-table.json").write_text(json.dumps(
        {"schema": "ad-record.scale-table/v1", "plan_sha256": plan_sha, "base": 2, "slots": slots,
         "note": "exponents are runner-lib's (power-of-2 per allocation, applied in the T/T^T kernels); 0 = unscaled"},
        indent=1))
    print("alloc-map: " + ", ".join(f"{n} {e['allocation']}{'/' + '/'.join(map(str, e['copies'])) if e['copies'] else ''}"
                                    for n, e in out["fields"].items()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
