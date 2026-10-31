"""Replay: the Python RRFS stream tree against the fork's native ezxml + extract_stream_interval.

For every shipped RRFS streams file, ``woof.hex.mpasphys.rrfs.stream_xml_rrfs.stream_tree`` (Python,
no native library) must give the same resolved tree as ``rrfs_native_stream_tree.stream_tree`` run through the
RRFS fork's native library (ezxml_parse_file, ezxml_attr first-value semantics and
extract_stream_interval, MPAS-Model-rrfs 1d6e370e src/external/ezxml/ezxml.c:83-99 and
src/framework/xml_stream_parser.c:925-1023,1150-1209,1495-1560): every element, tag, attribute
name, value and order, and the resolved filename_interval; a file native refuses must be refused.
When the r11 artifact (rw_mpas_rrfs_clock, woof b3a0b0db) is supplied, the first-value XML filter
must also equal its ``--stream-xml`` output byte for byte.

    python tests/replay_rrfs_stream_tree.py --library LIB/libmpasphys_native_rrfs.so \
        --receipt OUT.json [--rust-clock rw_mpas_rrfs_clock] ROOT_OR_FILE ...

Each ROOT is searched for files named streams.atmosphere* (and *.streams.atmosphere).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def flatten(element, out=None, depth=0):
    """Every element in document order: (depth, tag, ordered attributes)."""
    out = [] if out is None else out
    out.append((depth, element.tag, list(element.attrib.items())))
    for child in element:
        flatten(child, out, depth + 1)
    return out


def discover(roots):
    files = []
    for root in roots:
        root = Path(root)
        if root.is_file():
            files.append(root.resolve())
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and (path.name.startswith("streams.atmosphere") or path.name.endswith(".streams.atmosphere")):
                files.append(path.resolve())
    return list(dict.fromkeys(files))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--rust-clock", type=Path, default=None)
    parser.add_argument("roots", nargs="+")
    args = parser.parse_args(argv)
    from rrfs_native_stream_tree import stream_tree as native_stream_tree
    from woof.hex.mpasphys.rrfs.stream_xml_rrfs import first_value_xml, stream_tree
    files = discover(args.roots)
    rows, differing, elements, attributes = [], 0, 0, 0
    python_seconds = native_seconds = 0.0
    for path in files:
        row = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        try:
            t0 = time.perf_counter()
            ours = flatten(stream_tree(path))
            python_seconds += time.perf_counter() - t0
            ours_error = None
        except Exception as error:  # noqa: BLE001 - a refusal is compared with native's
            ours, ours_error = None, f"{type(error).__name__}: {error}"
        try:
            t0 = time.perf_counter()
            theirs = flatten(native_stream_tree(path, args.library))
            native_seconds += time.perf_counter() - t0
            theirs_error = None
        except Exception as error:  # noqa: BLE001
            theirs, theirs_error = None, f"{type(error).__name__}: {error}"
        same = (ours == theirs) if (ours is not None and theirs is not None) else (ours is None and theirs is None)
        row.update(python_error=ours_error, native_error=theirs_error, identical=same)
        if ours is not None:
            row["elements"] = len(ours)
            row["attributes"] = sum(len(a) for _, _, a in ours)
            elements += len(ours)
            attributes += row["attributes"]
            row["streams"] = {dict(a).get("name"): dict(a).get("filename_interval")
                              for d, tag, a in ours if d == 1 and tag in ("stream", "immutable_stream")}
        if args.rust_clock is not None:
            rust = subprocess.run([str(args.rust_clock), "--stream-xml", str(path)], capture_output=True, text=True)
            try:
                mine = first_value_xml(path.read_text())
            except ValueError:
                mine = None
            row["rust_filter_identical"] = (rust.returncode != 0 and mine is None) or (rust.returncode == 0 and rust.stdout == mine)
            same = same and row["rust_filter_identical"]
        if not same:
            differing += 1
            row["python"], row["native"] = ours, theirs
        rows.append(row)
    library_sha = hashlib.sha256(args.library.read_bytes()).hexdigest()
    receipt = {"schema": "rrfs-stream-tree-replay/v1", "files": len(files), "differing_files": differing,
               "refused_by_both": sum(1 for r in rows if r["python_error"] and r["native_error"]),
               "elements_compared": elements, "attributes_compared": attributes,
               "python_seconds": round(python_seconds, 4), "native_seconds": round(native_seconds, 4),
               "native_library": str(args.library.resolve()), "native_library_sha256": library_sha,
               "rust_filter_compared": args.rust_clock is not None,
               "native_process_count": 1, "whole_model_proof": False, "rows": rows}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "rows"}))
    return 0 if differing == 0 and files else 1


if __name__ == "__main__":
    sys.exit(main())
