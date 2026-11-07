"""Record every kernel a run compiles, then prebuild them for other cards.

THE COST THIS REMOVES.  A fresh container whose kernel cache does not hold
this card's images compiles every kernel before (and during) step 1:
measured at one to three minutes on a 300 x 300 x 49 member.  ``woof
warm-kernels`` pays that on purpose, but only for the card it runs on and
only for the kernels its synthetic columns reach, so a cache baked on one
card says nothing for another architecture, and kernels only a real case
reaches (terrain, lateral boundaries, output diagnostics) still compile on
the first real run.

RECORD.  ``WOOF_KERNEL_RECORD=FILE`` makes a run append, once per distinct
compile, exactly what it handed the compiler: the translation unit, its
options, and the remaining arguments of CuPy's cached compile
(``cupy.cuda.compiler._compile_with_cache_cuda``, which every RawModule,
RawKernel and elementwise or reduction kernel reaches) and of the engine's
direct NVRTC route (:func:`woof.nvrtc_ptx_cache.compile_using_nvrtc`).
The sources are recorded, not only their digests: a prebuild needs the text.

BUILD.  ``python -m woof.kernel_prebuild build FILE --arch 86,89,120
--cache-dir DIR`` replays each record through the same functions with the
target architecture named, so each entry is written under the key a run on
that card computes -- the source, the options, the architecture, the NVRTC
build and CuPy's own cache key -- and the reader on that card checks the
entry's content digest before it uses it.  Building for an architecture
other than the local card's writes the entry and then fails to load it
here, which is expected and counted as built.  The direct route's key also
holds the CUDA driver version, so those entries serve hosts on the same
driver as the build host only; the receipt says how many such entries
were built.

A run that sets nothing records nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

ENV_RECORD = "WOOF_KERNEL_RECORD"
RECORD_SCHEMA = "gpuwm.kernel-record/v1"
RECEIPT_SCHEMA = "gpuwm.kernel-prebuild/v1"

_LOCK = threading.Lock()
_SEEN: set[str] = set()
_INSTALLED = False


def _append(path: Path, row: dict) -> None:
    digest = hashlib.sha256(json.dumps(row, sort_keys=True).encode("utf-8")).hexdigest()
    with _LOCK:
        if digest in _SEEN:
            return
        _SEEN.add(digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"schema": RECORD_SCHEMA, "id": digest, **row}) + "\n")


def install_recorder(path=None) -> bool:
    """Start recording compiles to ``path`` (default: ``WOOF_KERNEL_RECORD``)."""
    global _INSTALLED
    path = path or os.environ.get(ENV_RECORD, "").strip()
    if not path or _INSTALLED:
        return bool(path) and _INSTALLED
    target = Path(path)
    import cupy.cuda.compiler as compiler

    from woof import nvrtc_ptx_cache

    cached = compiler._compile_with_cache_cuda

    def recording_cached(source, options, arch, extra_source=None, backend="nvrtc",
                         enable_cooperative_groups=False, name_expressions=None,
                         log_stream=None, cache_in_memory=False, jitify=False,
                         to_ltoir=False):
        try:
            _append(target, {
                "route": "cupy", "source": source, "options": list(options),
                "extra_source": extra_source, "backend": backend,
                "enable_cooperative_groups": bool(enable_cooperative_groups),
                "name_expressions": (None if name_expressions is None
                                     else list(name_expressions)),
                "jitify": bool(jitify), "to_ltoir": bool(to_ltoir),
                "cache_in_memory": bool(cache_in_memory),
                "recorded_arch": None if arch is None else str(arch)})
        except Exception:  # noqa: BLE001 - a record must never fail a compile
            pass
        return cached(source, options, arch, extra_source, backend,
                      enable_cooperative_groups, name_expressions, log_stream,
                      cache_in_memory, jitify, to_ltoir)

    direct = nvrtc_ptx_cache.compile_using_nvrtc

    def recording_direct(source, options=(), arch=None, filename="kern.cu",
                         name_expressions=None, log_stream=None,
                         cache_in_memory=False, jitify=None, method=None):
        try:
            _append(target, {
                "route": "direct", "source": source, "options": list(options),
                "filename": filename,
                "name_expressions": (None if name_expressions is None
                                     else list(name_expressions)),
                "cache_in_memory": bool(cache_in_memory),
                "jitify": jitify, "method": method,
                "recorded_arch": None if arch is None else str(arch)})
        except Exception:  # noqa: BLE001
            pass
        return direct(source, options, arch, filename, name_expressions,
                      log_stream, cache_in_memory, jitify, method)

    compiler._compile_with_cache_cuda = recording_cached
    nvrtc_ptx_cache.compile_using_nvrtc = recording_direct
    _INSTALLED = True
    return True


def _records(path: Path):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("schema") != RECORD_SCHEMA:
                raise ValueError(f"{path}: not a {RECORD_SCHEMA} record")
            yield row


def build(record_paths, arches, *, cache_dir) -> dict:
    """Compile every recorded kernel for each of ``arches`` into ``cache_dir``.

    ``cache_dir`` must be chosen before CuPy is imported in this process
    (the CLI does), so it is set here only when CuPy is not yet loaded.
    """
    if "cupy" not in sys.modules:
        os.environ["CUPY_CACHE_DIR"] = str(cache_dir)
    import cupy.cuda.compiler as compiler

    from woof import nvrtc_ptx_cache
    from woof.kernel_compile_notice import cupy_kernel_cache_dir

    if Path(cupy_kernel_cache_dir()).resolve() != Path(cache_dir).resolve():
        raise ValueError(
            f"CuPy's cache directory is {cupy_kernel_cache_dir()}, not {cache_dir}: "
            "run the build in a fresh process")
    rows = []
    for path in record_paths:
        rows.extend(_records(path))
    unique = {row["id"]: row for row in rows}
    began = time.perf_counter()
    counts = {}
    failures = []
    for arch in arches:
        arch = str(arch)
        built = loaded_here = 0
        for row in unique.values():
            try:
                from woof.hip_backend import active,compile_aot
                if active():
                    compile_aot(row["source"],tuple(row["options"]),arch)
                elif row["route"] == "cupy":
                    compiler._compile_with_cache_cuda(
                        row["source"], tuple(row["options"]), arch,
                        row.get("extra_source"), row.get("backend", "nvrtc"),
                        row.get("enable_cooperative_groups", False),
                        row.get("name_expressions"), None,
                        False, row.get("jitify", False), row.get("to_ltoir", False))
                else:
                    nvrtc_ptx_cache.compile_using_nvrtc(
                        row["source"], tuple(row["options"]), arch,
                        row.get("filename", "kern.cu"), row.get("name_expressions"),
                        None, False, row.get("jitify"), row.get("method"))
                built += 1
                loaded_here += 1
            except Exception as error:  # noqa: BLE001
                # A foreign architecture's image is written to the cache and
                # then refused by this card's loader: built, not loadable here.
                text = f"{type(error).__name__}: {error}"
                if "NO_BINARY_FOR_GPU" in text or "no kernel image" in text.lower() \
                        or "INVALID_SOURCE" in text or "invalid device" in text.lower():
                    built += 1
                else:
                    failures.append({"arch": arch, "route": row["route"],
                                     "id": row["id"], "error": text[:300]})
        counts[arch] = {"built": built, "loadable_on_this_card": loaded_here}
    entries = sum(1 for p in Path(cache_dir).iterdir() if p.is_file())
    direct = Path(cache_dir) / "gpuwm-direct-nvrtc"
    return {"schema": RECEIPT_SCHEMA, "records": len(unique),
            "cupy_records": sum(1 for r in unique.values() if r["route"] == "cupy"),
            "direct_records": sum(1 for r in unique.values() if r["route"] == "direct"),
            "arches": counts, "failures": failures, "cache_dir": str(cache_dir),
            "cache_entries": entries,
            "direct_images": (sum(1 for _ in direct.glob("*.image")) if direct.is_dir() else 0),
            "seconds": round(time.perf_counter() - began, 2)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m woof.kernel_prebuild",
                                     description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    b = sub.add_parser("build", help="compile recorded kernels for the named architectures")
    b.add_argument("records", type=Path, nargs="+")
    b.add_argument("--arch", required=True,
                   help="comma-separated compute capabilities, e.g. 86,89,120")
    b.add_argument("--cache-dir", type=Path, required=True)
    b.add_argument("--receipt", type=Path, default=None)
    args = parser.parse_args(argv)
    arches = [a.strip() for a in args.arch.split(",") if a.strip()]
    if not arches or not all(a.isdigit() for a in arches):
        print("kernel-prebuild: --arch takes compute capabilities such as 86,89,120",
              file=sys.stderr)
        return 2
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    receipt = build(args.records, arches, cache_dir=args.cache_dir)
    text = json.dumps(receipt, indent=1)
    if args.receipt is not None:
        args.receipt.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if not receipt["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
