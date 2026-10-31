"""Pinned native compilation for explicit MPAS/RRFS referee measurements.

This source-checkout tool is not part of the installed forecast framework.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable, Iterable

from woof.hex.mpasphys.native_library_build import (
    BACKEND, LIBRARY, LICENCE_COMPONENTS, NativeLibraryError, PIN_SETS,
    RRFS_BACKEND, SCHEMA, _sha, library_licence, source_manifest,
    source_manifest_path, stage_library, stubbed_components, toolchain,
)

MPICH = {"version": "4.3.2", "url": "https://www.mpich.org/static/downloads/4.3.2/mpich-4.3.2.tar.gz",
         "sha256": "47d774587a7156a53752218c811c852e70ac44db9c502dc3f399b4cb817e3818"}

MPICH_CONFIGURE = ("CC=gcc", "CXX=g++", "FC=gfortran", "F77=gfortran", "--with-device=ch4:ofi",
                   "--with-libfabric=embedded", "--enable-fast=O2", "--disable-doc", "--enable-romio",
                   "FFLAGS=-fallow-argument-mismatch", "FCFLAGS=-fallow-argument-mismatch")

PNETCDF = {"version": "1.14.1", "url": "https://parallel-netcdf.github.io/Release/pnetcdf-1.14.1.tar.gz",
           "sha256": "6f0f7221006c211fce9ddd2c008796b8c69dd717b2ad1be0b4027fc328fd3220"}

PNETCDF_CONFIGURE = ("--disable-fortran", "--disable-cxx", "--disable-shared", "MPICC=mpicc",
                     "CFLAGS=-O2 -fPIC -std=gnu17")

TOOLS = ("gcc", "g++", "gfortran", "make", "cpp")

def _key(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]

def backend_directory(pin_set: str = "v8.4.2") -> Path:
    """Compiler sources are resolved only from this reference checkout."""
    row = PIN_SETS[pin_set]
    checkout = Path(__file__).resolve().parents[2] / row["checkout"]
    if all((checkout / name).is_file() for name in row["files"]):
        return checkout
    raise NativeLibraryError(f"native referee sources are absent from {checkout}")


def check_prerequisites() -> None:
    missing = [tool for tool in TOOLS if shutil.which(tool) is None]
    if missing:
        raise NativeLibraryError(
            "the MPAS native framework library is compiled on this host and " + ", ".join(missing)
            + " is not on PATH; install the GNU toolchain (gcc, g++, gfortran, make), or pass "
            "--native-library with an existing build (libmpasphys_native.so beside its abi.json and "
            "build-receipt.json)")
    if not Path("/usr/include/netcdf.mod").is_file():
        raise NativeLibraryError(
            "the MPAS native framework library needs the netCDF-Fortran development files "
            "(/usr/include/netcdf.mod and libnetcdff; e.g. apt install libnetcdff-dev); install them, or "
            "pass --native-library with an existing build")

def _run(command: list[str], *, cwd: Path, log: Path, env: dict[str, str] | None = None) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream:
        stream.write("$ " + " ".join(command) + "\n")
        stream.flush()
        completed = subprocess.run(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, env=env)
    if completed.returncode:
        raise NativeLibraryError(f"{command[0]} failed ({completed.returncode}) while building the native "
                                 f"library; the log is {log}")

def _unpack(fetch: Callable[..., Path], cache: Path, row: dict[str, str], name: str, offline: bool,
            directory: Path) -> Path:
    archive = fetch(name, row["url"], row["sha256"], cache / "native" / "downloads" / row["sha256"]
                    / row["url"].rsplit("/", 1)[1], offline=offline)
    source = directory / "src"
    if not (source / "configure").is_file():
        if source.exists():
            shutil.rmtree(source)
        source.mkdir(parents=True)
        subprocess.run(["tar", "--no-same-owner", "--strip-components=1", "-xzf", str(archive), "-C", str(source)],
                       check=True)
    return source

def build_mpich(cache: Path, fetch: Callable[..., Path], *, jobs: int, offline: bool,
                log: Callable[[str], None]) -> Path:
    tools = toolchain()
    directory = cache / "native" / f"mpich-{MPICH['version']}-{_key(MPICH, MPICH_CONFIGURE, tools)}"
    install = directory / "install"
    marker = install / ".gpuwm-hex-built.json"
    if marker.is_file() and (install / "bin/mpif90").is_file():
        return install
    log(f"building MPICH {MPICH['version']} (the native reference's MPI) under {directory}; a few minutes")
    source = _unpack(fetch, cache, MPICH, "MPICH", offline, directory)
    logs = directory / "logs"
    _run(["./configure", f"--prefix={install}", *MPICH_CONFIGURE], cwd=source, log=logs / "configure.log")
    _run(["make", f"-j{jobs}"], cwd=source, log=logs / "make.log")
    _run(["make", "install"], cwd=source, log=logs / "install.log")
    marker.write_text(json.dumps({"mpich": MPICH, "configure": MPICH_CONFIGURE, "toolchain": tools}, indent=2) + "\n")
    shutil.rmtree(source, ignore_errors=True)
    return install

def build_pnetcdf(cache: Path, fetch: Callable[..., Path], mpich: Path, *, jobs: int, offline: bool,
                  log: Callable[[str], None]) -> Path:
    directory = cache / "native" / f"pnetcdf-{PNETCDF['version']}-{_key(PNETCDF, PNETCDF_CONFIGURE, str(mpich))}"
    install = directory / "install"
    library = install / "lib/libpnetcdf.a"
    if (install / ".gpuwm-hex-built.json").is_file() and library.is_file():
        return install
    log(f"building PnetCDF {PNETCDF['version']} (static, PIC) under {directory}")
    source = _unpack(fetch, cache, PNETCDF, "PnetCDF", offline, directory)
    env = dict(os.environ, PATH=f"{mpich / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}")
    logs = directory / "logs"
    _run(["./configure", f"--prefix={install}", *PNETCDF_CONFIGURE], cwd=source, log=logs / "configure.log", env=env)
    _run(["make", f"-j{jobs}"], cwd=source, log=logs / "make.log", env=env)
    _run(["make", "install"], cwd=source, log=logs / "install.log", env=env)
    (install / ".gpuwm-hex-built.json").write_text(json.dumps(
        {"pnetcdf": PNETCDF, "configure": PNETCDF_CONFIGURE, "mpich": str(mpich),
         "libpnetcdf_sha256": _sha(library)}, indent=2) + "\n")
    shutil.rmtree(source, ignore_errors=True)
    return install

def fetch_mpas_tree(cache: Path, fetch_file: Callable[..., Path], *, mpas_source: Path | None, offline: bool,
                    log: Callable[[str], None], skip: Iterable[str] = (), pin_set: str = "v8.4.2") -> Path:
    """Every source file the build reads, at its MPAS-tree path, each verified by SHA-256.

    ``skip`` names MPAS-tree directories (a stubbed component's) that are neither fetched nor read.
    """
    manifest = source_manifest(pin_set)
    row = PIN_SETS[pin_set]
    tree = cache / "native" / row["tree"] / manifest["pins"][row["model"]][:12]
    prefixes = tuple(directory.rstrip("/") + "/" for directory in skip)
    rows = {relative: row for relative, row in manifest["files"].items()
            if not (prefixes and relative.startswith(prefixes))}
    pending = [(relative, row) for relative, row in rows.items()
               if not ((tree / relative).is_file() and _sha(tree / relative) == row["sha256"])]
    if pending:
        log(f"fetching {len(pending)} pinned MPAS source files into {tree}")

        def one(item):
            relative, row = item
            return fetch_file(relative, row["origin"], row["path"], row["sha256"], tree / relative,
                              mpas_source=mpas_source, offline=offline)

        with ThreadPoolExecutor(8) as pool:
            list(pool.map(one, pending))
    return tree

def build_native_library(cache: Path, *, fetch: Callable[..., Path], fetch_file: Callable[..., Path],
                         mpas_source: Path | None = None, mpich: Path | None = None, jobs: int | None = None,
                         offline: bool = False, licence_opt_in: Iterable[str] = (),
                         log: Callable[[str], None], pin_set: str = "v8.4.2",
                         static_gnu_runtime: bool = False) -> dict[str, Any]:
    """Build (or reuse) the pin set's native library; return its staged row (path, sha256, receipt, licence).

    ``licence_opt_in`` names the terms the caller opts in to (``ucar-noahmp``, ``noaa-gsl-unlicensed``);
    every component not named is compiled as its licence stub (the default: both stubbed).
    """
    check_prerequisites()
    unknown = sorted(set(licence_opt_in) - set(LICENCE_COMPONENTS))
    if unknown:
        raise NativeLibraryError(f"unknown licence opt-in {', '.join(unknown)}; the native library's are "
                                 + ", ".join(LICENCE_COMPONENTS))
    stubs = stubbed_components(licence_opt_in)
    jobs = jobs or max(1, min(16, os.cpu_count() or 4))
    if mpich is None:
        mpich = build_mpich(cache, fetch, jobs=jobs, offline=offline, log=log)
    else:
        mpich = Path(mpich).resolve()
        if not ((mpich / "bin/mpif90").is_file() and (mpich / "bin/mpicc").is_file()):
            raise NativeLibraryError(f"--mpich {mpich} has no bin/mpif90 and bin/mpicc")
    pnetcdf = build_pnetcdf(cache, fetch, mpich, jobs=jobs, offline=offline, log=log)
    skip = [directory for terms, (component, directory) in LICENCE_COMPONENTS.items() if component in stubs]
    tree = fetch_mpas_tree(cache, fetch_file, mpas_source=mpas_source, offline=offline, log=log, skip=skip,
                           pin_set=pin_set)
    backend = backend_directory(pin_set)
    manifest = source_manifest_path(pin_set)
    library = PIN_SETS[pin_set]["library"]
    recipe = _key(PIN_SETS[pin_set], {name: _sha(backend / name) for name in PIN_SETS[pin_set]["files"]}, _sha(manifest), str(mpich),
                  _sha(pnetcdf / "lib/libpnetcdf.a"), toolchain(), stubs, static_gnu_runtime)
    work = cache / "native" / f"build-{recipe}"
    receipt_path = work / "build-receipt.json"
    built = None
    if receipt_path.is_file() and (work / library).is_file():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("status") == "built" and receipt.get("library_sha256") == _sha(work / library):
            built = receipt
    if built is None:
        flavour = ("with licence stubs for " + " and ".join(stubs)) if stubs else "with every opted-in component"
        what = "the RRFS fork's native library" if pin_set == "rrfs" else "the MPAS native framework library"
        log(f"compiling {what} with {toolchain()['gfortran']} (strict recipe, {flavour}) in {work}")
        env = dict(os.environ, PATH=f"{mpich / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}")
        env["LD_LIBRARY_PATH"] = f"{mpich / 'lib'}{os.pathsep}{os.environ.get('LD_LIBRARY_PATH', '')}"
        work.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(backend / "build.py"), "--source", str(tree), "--work", str(work),
              "--pnetcdf", str(pnetcdf), "--source-manifest", str(manifest), "--no-oracle-transport",
              "--jobs", str(jobs), *PIN_SETS[pin_set].get("build_arguments", []), *(["--static-gnu-runtime"] if static_gnu_runtime else []),
              *[part for stub in stubs for part in ("--licence-stub", stub)]]
        build_log = cache / "native" / "logs" / f"build-{recipe}.log"
        # manylinux's GNU archive can contain non-PIC TLS. Keep the actual link
        # failure and retry the same strict source recipe with shared runtimes.
        # The changed compile receipt can rebuild objects in the same workdir.
        # Compilation errors and other linker failures still fail closed.
        def non_pic_runtime():
            text = build_log.read_text(errors="replace") if build_log.is_file() else ""
            return all(token in text for token in (
                "libgfortran.a(", "R_X86_64_TPOFF32", "can not be used when making a shared object", "recompile with -fPIC"))
        if static_gnu_runtime and non_pic_runtime():
            command.remove("--static-gnu-runtime")
        try:
            _run(command, cwd=work, log=build_log, env=env)
        except NativeLibraryError:
            if not static_gnu_runtime or "--static-gnu-runtime" not in command or not non_pic_runtime():
                raise
            command.remove("--static-gnu-runtime")
            _run(command, cwd=work, log=build_log, env=env)
        built = json.loads(receipt_path.read_text())
        if static_gnu_runtime and "--static-gnu-runtime" not in command:
            built["static_gnu_runtime_attempt"] = {
                "flags": ["-static-libgfortran", "-static-libgcc"],
                "result": "non-PIC libgfortran.a TLS relocation; shared runtime fallback",
                "build_log_sha256": _sha(build_log),
            }
            receipt_path.write_text(json.dumps(built, indent=2) + "\n")
    relocatable_runpath(cache, work / library, receipt_path, mpich)
    return stage_library(cache, work / library, licence_opt_in=licence_opt_in, pin_set=pin_set)

def _elf_runpath_entries(data: bytes):
    """(string offset, current value) of every DT_RUNPATH/DT_RPATH in a little-endian ELF64 file."""
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        raise NativeLibraryError("not a little-endian ELF64 file")
    u16 = lambda at: int.from_bytes(data[at:at + 2], "little")
    u32 = lambda at: int.from_bytes(data[at:at + 4], "little")
    u64 = lambda at: int.from_bytes(data[at:at + 8], "little")
    shoff, shentsize, shnum = u64(0x28), u16(0x3A), u16(0x3C)
    sections = [(u32(shoff + i * shentsize + 4), u64(shoff + i * shentsize + 0x18),
                 u64(shoff + i * shentsize + 0x20), u32(shoff + i * shentsize + 0x28)) for i in range(shnum)]
    found = []
    for kind, offset, size, link in sections:
        if kind != 6:  # SHT_DYNAMIC
            continue
        strtab = sections[link][1]
        for entry in range(offset, offset + size, 16):
            tag, value = u64(entry), u64(entry + 8)
            if tag == 0:
                break
            if tag in (15, 29):  # DT_RPATH, DT_RUNPATH
                start = strtab + value
                end = data.index(b"\0", start)
                found.append((start, data[start:end].decode()))
    return found

def set_runpath_in_place(path: Path, value: str) -> list[str]:
    """Rewrite a shared library's RUNPATH string in place (the new one no longer than the old); return the old."""
    data = bytearray(Path(path).read_bytes())
    entries = _elf_runpath_entries(bytes(data))
    encoded = value.encode()
    for start, old in entries:
        if old == value:
            continue
        if len(encoded) > len(old.encode()):
            raise NativeLibraryError(f"{path}: RUNPATH {value!r} is longer than the {old!r} it replaces")
        data[start:start + len(old.encode())] = encoded + b"\0" * (len(old.encode()) - len(encoded))
    if any(old != value for _, old in entries):
        handle, temporary = tempfile.mkstemp(dir=Path(path).parent, prefix=".partial-")
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
        os.chmod(temporary, os.stat(path).st_mode & 0o7777)
        os.replace(temporary, path)
    return [old for _, old in entries]

def relocatable_runpath(cache: Path, library: Path, receipt_path: Path, mpich: Path) -> None:
    """Make the native library (and the MPICH it loads) find MPICH relative to itself, not by absolute path.

    THE BREAKAGE THIS PREVENTS: the MPICH wrappers link with an absolute RUNPATH to the build folder's
    MPICH and PnetCDF, so a physics-resources cache copied elsewhere (an image's builder stage into its final
    stage) loaded MPICH from the original folder, or failed to load at all once it was gone (2026-10-17).
    The library's RUNPATH becomes ``$ORIGIN``-relative to the MPICH inside the cache, valid both from the
    build folder and from native/library/<sha256>/ where it is staged (PnetCDF is linked statically, so it
    needs no entry); MPICH's own libraries find each other by ``$ORIGIN``.  The build receipt's library
    SHA-256 is updated to the rewritten bytes.  An MPICH outside the cache (an adopted --mpich) is left
    absolute, as it is not part of what moves.
    """
    cache, mpich = Path(cache).resolve(), Path(mpich).resolve()
    lib = mpich / "lib"
    try:
        lib.relative_to(cache)
    except ValueError:
        return
    for shared in sorted(lib.glob("lib*.so*")):
        if shared.is_file() and not shared.is_symlink() and _elf_runpath_entries(shared.read_bytes()):
            set_runpath_in_place(shared, "$ORIGIN")
    library = Path(library).resolve()
    staged_folder = cache / "native" / "library" / ("0" * 64)
    value = ":".join(dict.fromkeys("$ORIGIN/" + os.path.relpath(lib, folder)
                                   for folder in (library.parent, staged_folder)))
    before = _sha(library)
    try:
        set_runpath_in_place(library, value)
    except NativeLibraryError:
        # A cache folder so short that the relative form is the longer string: patchelf can grow it.
        patchelf = shutil.which("patchelf")
        if patchelf is None:
            print(f"woof hex: {library} keeps its absolute RUNPATH (no patchelf to lengthen it); this cache "
                  "is not relocatable", file=sys.stderr, flush=True)
            return
        subprocess.run([patchelf, "--set-rpath", value, str(library)], check=True)
    after = _sha(library)
    if after != before and receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("library_sha256") == before:
            receipt["library_sha256"] = after
            receipt["runpath"] = value
            receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
