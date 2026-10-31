"""Native receipt metadata and an explicit source-checkout referee bridge.

The installed default MPAS route never builds or loads a native library.
Compilation lives in tools/native_dry_reference/native_library_build.py.
Receipt validation remains available for explicitly selected native referees.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterable

SCHEMA = "gpuwm-hex.native-library-build/v1"

LIBRARY = "libmpasphys_native.so"

BACKEND = ("build.py", "generate.py", "framework.py", "tables.py", "noahmp_io.py", "runtime.f90",
           "licence_stub_noahmp.F90")

RRFS_BACKEND = ("build.py", "generate.py", "framework.py", "sources.py", "tables.py", "noahmp_io.py", "runtime.f90",
                "licence_stub_noahmp.F90")

PIN_SETS = {
    "v8.4.2": {"library": LIBRARY, "backend": "mpasphys-native-backend", "files": BACKEND,
               "checkout": "tools/native_physics_reference/backend", "manifest": "mpas-source-v842.json",
               "model": "mpas-model", "tree": "mpas-src"},
    "rrfs": {"library": "libmpasphys_native_rrfs.so", "backend": "mpasphys-native-backend-rrfs",
             "files": RRFS_BACKEND, "checkout": "tools/native_physics_reference/backend/rrfs",
             "manifest": "mpas-source-rrfs.json", "model": "rrfs-model", "tree": "rrfs-src"},
    "rrfs-44c0849": {"library": "libmpasphys_native_rrfs.so", "backend": "mpasphys-native-backend-rrfs",
             "files": RRFS_BACKEND, "checkout": "tools/native_physics_reference/backend/rrfs",
             "manifest": "mpas-source-rrfs-44c0849.json", "model": "rrfs-model", "tree": "rrfs-src",
             "build_arguments": ["--pin-set", "rrfs-44c0849"]},
}

LICENCE_COMPONENTS = {
    "ucar-noahmp": ("noahmp", "src/core_atmosphere/physics/physics_noahmp"),
}

class NativeLibraryError(ValueError):
    """The native library cannot be built or adopted here, named with what closes it."""

def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def _data_root(name: str) -> Path:
    return Path(__file__).resolve().parents[1] / "data" / name

def source_manifest_path(pin_set: str = "v8.4.2") -> Path:
    return _data_root("mpasphys-native") / PIN_SETS[pin_set]["manifest"]

def source_manifest(pin_set: str = "v8.4.2") -> dict[str, Any]:
    return json.loads(source_manifest_path(pin_set).read_text(encoding="utf-8"))

def toolchain() -> dict[str, str]:
    out = {}
    for tool in ("gfortran", "gcc"):
        path = shutil.which(tool)
        version = ""
        if path:
            try:
                version = subprocess.run([path, "--version"], capture_output=True, text=True,
                                         check=False).stdout.splitlines()[0]
            except (OSError, IndexError):
                version = ""
        out[tool] = version
    return out

def stubbed_components(licence_opt_in: Iterable[str] = ()) -> list[str]:
    """The components build.py stubs for these opt-ins (every one not opted in to by its terms)."""
    named = set(licence_opt_in)
    return sorted(component for terms, (component, _) in LICENCE_COMPONENTS.items() if terms not in named)

def library_licence(library: Path) -> dict[str, Any]:
    """What a built library carries of each gated component, from the build receipt beside it.

    ``flavour`` is ``stubbed`` (every gated component is a stub: the default build), ``opted-in`` (every
    one compiled from upstream) or ``partly-stubbed``; a receipt that predates stubs, or a library with
    no receipt beside it (a lab build named by path), compiled everything: ``unrecorded``, nothing stubbed.
    """
    path = Path(library).with_name("build-receipt.json")
    try:
        licence = json.loads(path.read_text()).get("licence")
    except (OSError, ValueError):
        licence = None
    if not licence:
        return {"flavour": "unrecorded", "stubbed": {}, "compiled": {}}
    stubbed, compiled = licence.get("stubbed", {}), licence.get("compiled", {})
    flavour = "stubbed" if stubbed and not compiled else "opted-in" if not stubbed else "partly-stubbed"
    return {"flavour": flavour, "stubbed": stubbed, "compiled": compiled}

def stage_library(cache: Path, library: Path, *, licence_opt_in: Iterable[str] = (),
                  pin_set: str = "v8.4.2") -> dict[str, Any]:
    """Copy a built library with its ABI manifest and receipt under its own SHA-256."""
    library = Path(library).resolve()
    beside = {name: library.with_name(name) for name in ("abi.json", "build-receipt.json")}
    if not library.is_file() or not all(path.is_file() for path in beside.values()):
        raise NativeLibraryError(f"{library}: a native library needs abi.json and build-receipt.json beside it "
                                 "(tools/native_physics_reference/backend/build.py writes both)")
    receipt = json.loads(beside["build-receipt.json"].read_text())
    digest = _sha(library)
    if receipt.get("library_sha256") != digest or receipt.get("abi_sha256") != _sha(beside["abi.json"]):
        raise NativeLibraryError(f"{library} does not match the build receipt beside it")
    target = cache / "native" / "library" / digest
    staged = target / PIN_SETS[pin_set]["library"]
    if not (staged.is_file() and _sha(staged) == digest):
        target.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(dir=target, prefix=".partial-")
        os.close(handle)
        shutil.copyfile(library, temporary)
        os.chmod(temporary, 0o755)
        for name, path in beside.items():
            shutil.copyfile(path, target / name)
        os.replace(temporary, staged)
    licence = library_licence(staged)
    if licence["flavour"] != "unrecorded":
        licence["opt_in"] = sorted(set(licence_opt_in) & {row["terms"] for row in licence["compiled"].values()})
    return {"path": str(staged), "sha256": digest, "source_pin": receipt.get("source_pin"),
            "compiler": receipt.get("compiler"), "source_verification": receipt.get("source_verification"),
            "mpif90": receipt.get("mpif90"), "licence": licence}

def referee_builder(tool: str = "native_library_build"):
    """Load the source checkout's proof tool, never a packaged compiler recipe."""
    if tool not in {"native_library_build", "thompson_tables"}:
        raise NativeLibraryError(f"unknown native referee tool {tool!r}")
    path = Path(__file__).resolve().parents[3] / "components" / "hex" / f"tools/native_dry_reference/{tool}.py"
    if not path.is_file():
        raise NativeLibraryError(
            "native compilation is a source-checkout referee tool: "
            f"tools/native_dry_reference/{tool}.py is absent; "
            "use pinned prebuilt resources or an existing --native-library")
    name = "_hex_native_referee_builder" if tool == "native_library_build" else "_hex_native_referee_tables"
    module = sys.modules.get(name)
    if module is not None and Path(module.__file__).resolve() == path:
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def build_native_library(*args, **kwargs):
    """Explicit source builds delegate to the proof tool outside the package."""
    return referee_builder().build_native_library(*args, **kwargs)
