#!/usr/bin/env python3
"""Turn real forecasts' recorded CUDA compile requests into the shipped recipe catalog.

Input: one or more request records written by a forecast run with
``WOOF_HEX_AOT_RECORD=<file>`` (or ``woof.hex.kernel_prewarm`` manifests,
``<prep cache>/kernels/*.json``, which hold the CuPy route only).  A record
holds every request the run made: hexcore's own modules, the engine's physics
modules, CuPy's generated elementwise and reduction kernels, and the engine's
direct-NVRTC units (radiation, land, drag).

For a complete recorded run, pass ``--plan`` plus exactly one companion
prewarm manifest and ``--receipt``. The plan supplies final compiler options
and direct calls. Its recorded numerical environment is required; cached
requests must match their companion after that normalizer, including names
and header metadata. Unknown, ambiguous or corrupt requests fail the export.
This mode emits source recipes, never reuses an image from another compiler.

Output: ``src/hexcore/data/compiler-recipes.json``, which
``woof.hex.cuda_backend.aot_build.inventory`` reads and ``tools/build_hex_aot.py``
compiles for every shipped architecture.  The capturing venv's site-packages
prefix in CuPy's ``-I`` options becomes a token and ``extra_source`` becomes
its digest, so the catalog is the same whichever venv recorded it.

    python tools/record_hex_aot_recipes.py --manifest RECORD.json [--manifest B.json ...] \
        --output src/hexcore/data/compiler-recipes.json

Record every route under BOTH numerics (one ``--numerics fast`` and one
``--numerics strict`` run each): a unit recorded in one mode has no image in the
other, and a fresh box compiles it on one core at first use (547 s first strict
step on 0.4.9, VRFS 2026-10-28).  ``--merge`` keeps every row of the existing
catalog and adds the recorded rows it lacks (rows the declared inventory serves,
and shape modules specialised to the recording mesh, are not added); ``--routes direct-nvrtc`` adds only direct-NVRTC rows, for the
CUDA 12 catalog from a CUDA 13 recording (their sources carry no CuPy headers).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re

from woof.hex.cuda_backend.aot import (DIRECT_ROUTE, direct_metadata, portable_options, recipe,
                                      request_metadata)

_CUPY_INCLUDE = re.compile(r"^-I(.+)/cupy/_core/include$")
#: The engine's direct-NVRTC units include their own kernel folder
#: (``-I<site-packages>/woof/core/kernels``) and carry no CuPy include, so the
#: CuPy rule alone left the recording venv's absolute path in the catalog: a row
#: no other install can match, and one the pack builder cannot compile on
#: another host ("cannot open source file glibc_flt32.cuh", 2026-10-26).
_ENGINE_INCLUDE = re.compile(r"^-I(.+/site-packages)/woof/")
#: woof.hex.cuda_shape renames each specialised dimension parameter to this prefix.
_SHAPE_MARK = "hex_shape_arg_"


def portable_entry(arguments):
    roots = {match.group(1) for option in arguments.get("options") or ()
             if (match := _CUPY_INCLUDE.match(str(option)) or _ENGINE_INCLUDE.match(str(option)))}
    options = portable_options(arguments.get("options") or (), roots=tuple(roots))
    if arguments.get("route") == DIRECT_ROUTE:
        entry = {"source": arguments["source"], "options": options,
                 **direct_metadata(arguments.get("filename"), arguments.get("cache_in_memory", False))}
    else:
        entry = {"source": arguments["source"], "options": options, **request_metadata(arguments)}
    recipe(entry)  # refuses a route the pack cannot reproduce
    return entry



def plan_entries(plan, manifest, normalize):
    """Join final compiler arguments to cached-route metadata without loading images.

    A prewarm manifest can sit outside the engine numerical wrapper. The plan
    records the final ordered options, so accepting its companion requires an
    exact match after that recorded run's normalizer. Never infer a cached
    request's extra-source headers or silently drop unmatched requests.
    """
    from woof.hex import kernel_pack, kernel_plan
    if plan.get("schema") != kernel_plan.SCHEMA or manifest.get("schema") != kernel_pack.SCHEMA:
        raise ValueError("expected one kernel plan and its companion prewarm manifest")
    identity = manifest.get("compiler")
    if not isinstance(identity, dict) or not identity or any(
            plan.get("compiler", {}).get(k) != v for k, v in identity.items()):
        raise ValueError("plan and manifest compiler identities differ")
    records, cached = plan.get("entries"), manifest.get("entries")
    if not isinstance(records, list) or not isinstance(cached, list) or max(len(records), len(cached)) > 4000:
        raise ValueError("missing or oversized source request list")
    sources = {}
    for record in cached:
        a = record.get("arguments")
        if not isinstance(a, dict) or record.get("sha256") != kernel_pack.digest(a):
            raise ValueError("prewarm entry digest mismatch")
        sources.setdefault(a["source"], []).append(a)
    result, bindings = [], []
    for record in records:
        a = record.get("arguments")
        if not isinstance(a, dict) or record.get("sha256") != kernel_plan._key(a):
            raise ValueError("plan entry digest mismatch")
        if a.get("method") is not None:
            raise ValueError("plan uses an unsupported compiler method")
        is_cached = bool(re.fullmatch(r"[a-f0-9]+\.(?:cubin|ptx|ltoir)\.cu", a["filename"]))
        if is_cached:
            matches = {}
            for c in sources.get(a["source"], []):
                options = list(normalize(c["options"]))
                if (options + ["-ftz=true"] != a["options"]
                        or c.get("name_expressions") != a.get("name_expressions")):
                    continue
                entry = portable_entry({**c, "options": options})
                matches[kernel_pack.digest(entry)] = entry
            if len(matches) != 1:
                raise ValueError(f"{record['sha256']}: expected one exact companion request, found {len(matches)}")
            entry = next(iter(matches.values()))
        else:
            if a.get("name_expressions"):
                raise ValueError("direct plan request has name expressions")
            entry = portable_entry({**a, "route": DIRECT_ROUTE})
        result.append(entry)
        bindings.append({"request_sha256": record["sha256"],
                         "recipe_sha256": kernel_pack.digest(entry),
                         "route": "cached" if is_cached else DIRECT_ROUTE})
    return result, bindings


def plan_environment(plan):
    """Recover explicitly recorded numerical controls; never assume FAST."""
    try:
        environment = json.loads(plan["tag_text"])["environment"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("plan lacks a recorded numerical environment") from error
    names = ("WOOF_HEX_NUMERICS", "GPUWM_NUMERICS", "GPUWM_WRF_EXACT", "CUPY_CUDA_COMPILE_WITH_DEBUG")
    if any(name not in environment or not isinstance(environment[name], str) for name in names):
        raise ValueError("plan numerical environment is incomplete")
    if environment["CUPY_CUDA_COMPILE_WITH_DEBUG"] not in ("", "0"):
        raise ValueError("debug compilation is not supported by plan recipe export")
    return {name: environment[name] for name in names}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path,
                        help="Join final compiler arguments with one companion prewarm manifest")
    parser.add_argument("--receipt", type=Path,
                        help="Required with --plan: source hashes and every request-to-recipe binding")
    parser.add_argument("--merge", action="store_true",
                        help="keep every row of the existing --output catalog and add the recorded rows it lacks")
    parser.add_argument("--routes", default="cupy,direct-nvrtc",
                        help="comma list of recorded routes to export (cupy, direct-nvrtc; default both)")
    args = parser.parse_args()
    routes = {route.strip() for route in args.routes.split(",") if route.strip()}
    if not routes <= {"cupy", DIRECT_ROUTE}:
        parser.error("--routes takes cupy and/or direct-nvrtc")
    if args.plan:
        if len(args.manifest) != 1 or args.receipt is None:
            parser.error("--plan requires exactly one --manifest and --receipt")
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        environment = plan_environment(plan)
        for name, value in environment.items():
            os.environ[name] = value
        from woof.hex.cuda_backend.aot import effective_options
        rows, bindings = plan_entries(plan, json.loads(args.manifest[0].read_text(encoding="utf-8")),
                                      effective_options)
        args.output.write_text(json.dumps(rows, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        receipt = {"entries": len(rows), "bindings": bindings, "environment": environment,
                   "inputs": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                              for p in [args.plan, args.manifest[0]]],
                   "output": str(args.output), "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                   "scope": "Source recipes only. No compiled images or full-model qualification."}
        args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"entries": len(rows), "cached": sum(x["route"] == "cached" for x in bindings),
                          "direct": sum(x["route"] == DIRECT_ROUTE for x in bindings), "output": str(args.output)}))
        return
    catalog, skipped = {}, 0
    kept, indent = 0, 1
    if args.merge and args.output.is_file():
        text = args.output.read_text(encoding="utf-8")
        lines = text.splitlines()
        if len(lines) > 1 and lines[1].strip():
            indent = len(lines[1]) - len(lines[1].lstrip(" ")) or 1  # keep the file's own layout
        for entry in json.loads(text):
            catalog[hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()] = entry
        kept = len(catalog)
    declared, declared_selections = set(), set()
    if args.merge:
        from woof.hex.cuda_backend.aot_build import declared_requests, selection
        declared, declared_selections = declared_requests()
    added = served = other_route = mesh_specific = 0
    for path in args.manifest:
        document = json.loads(path.read_text(encoding="utf-8"))
        for record in document["entries"]:
            try:
                entry = portable_entry(record["arguments"])
            except ValueError:
                skipped += 1
                continue
            if (DIRECT_ROUTE if entry.get("route") == DIRECT_ROUTE else "cupy") not in routes:
                other_route += 1
                continue
            direct = entry.get("route") == DIRECT_ROUTE
            if ((not direct and hashlib.sha256(entry["source"].encode("utf-8")).hexdigest() in declared)
                    or (args.merge and selection(recipe(entry)) in declared_selections)):
                served += 1
                continue
            if args.merge and _SHAPE_MARK in entry["source"]:
                # woof.hex.cuda_shape bakes the recording mesh's dimensions into
                # these sources: no other mesh ever requests them.
                mesh_specific += 1
                continue
            digest = hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()
            added += digest not in catalog
            catalog[digest] = entry
    # A merge keeps the existing rows in their file order and appends the new
    # ones in digest order, so the catalog diff shows only what was added.
    order = list(catalog)[:kept] + sorted(list(catalog)[kept:]) if args.merge else sorted(catalog)
    rows = [catalog[key] for key in order]
    args.output.write_text(json.dumps(rows, indent=indent, sort_keys=True) + "\n", encoding="utf-8")
    from woof.hex.cuda_backend.aot_build import catalog_numerics_gaps
    gaps = catalog_numerics_gaps(rows)
    print(json.dumps({"entries": len(rows), "kept": kept, "added": added,
                      "declared_skipped": served, "route_skipped": other_route,
                      "mesh_specific_skipped": mesh_specific,
                      "skipped_linking_routes": skipped, "numerics_gaps": gaps,
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
