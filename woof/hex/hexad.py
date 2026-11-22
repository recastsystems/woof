"""Locate the differentiable engine's runner and C-ABI library (rust/hexad).

The wheel ships two products of the one crate in woof/hex/libexec:

* ``ad-binder``        the replay / tangent / adjoint window runner, and
* ``libhexad_phys.so`` the same runner behind the libhexad-phys C ABI (v2).

Each resolves from an explicit environment variable, then the wheel's libexec, then a
source-tree build. A missing product refuses by name; nothing falls back to another path.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

BINARY_ENV = "WOOF_HEX_AD_BINDER"
LIBRARY_ENV = "WOOF_HEX_HEXAD_LIBRARY"
BINARY = "ad-binder"
LIBRARY = "libhexad_phys.so"


class HexadNotInstalled(RuntimeError):
    """The runner or its library is not in this install."""


def _resolve(env: str, name: str, source_dir: str) -> Path:
    explicit = os.environ.get(env)
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise HexadNotInstalled(f"{env}={path} does not exist")
        return path
    here = Path(__file__).resolve()
    candidates = [here.with_name("libexec") / name]
    # (WOOF 1.1.7) A PyPI wheel of recast-woof carries the hex helpers' pins, not the helpers: ad-binder and
    # libhexad_phys.so come in the release's woof-hex-helpers archive, staged on first use like every other hex
    # helper (woof.hex.helpers).  THE BREAKAGE: the README's 12 km `woof hex adjoint-test`, from the built wheel with
    # [ad], stopped at once on "no window runner: this install has no ad-binder" (cut-117-r2.sh adgate, dry run on
    # the round-1 inputs, 17 Nov).  Only a helper this install pins is fetched; a source checkout has no pins.
    from woof.hex import helpers
    if helpers.on_demand(name):
        candidates.append(helpers.path(name))
    candidates.append(here.parents[2] / "rust/hexad" / source_dir / "release" / name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise HexadNotInstalled(
        f"{name} is not installed (it ships in the Linux woof hex wheel's woof/hex/libexec; a source build "
        f"needs cargo, a C++ compiler and NVCC for the rule fatbins). Set {env} to a built copy.")


def binary() -> Path:
    """The ad-binder executable."""
    return _resolve(BINARY_ENV, BINARY, "target")


def library() -> Path:
    """libhexad_phys.so (built with the capi-exports feature)."""
    return _resolve(LIBRARY_ENV, LIBRARY, "target-capi")


def header() -> Path:
    """hexad_phys.h, the C ABI's header (the wheel's woof/hex/include, else the source tree's rust/hexad/include)."""
    here = Path(__file__).resolve()
    for candidate in (here.with_name("include") / "hexad_phys.h", here.parents[2] / "rust/hexad/include/hexad_phys.h"):
        if candidate.is_file():
            return candidate
    raise HexadNotInstalled("hexad_phys.h is not installed (it ships in the woof hex wheel's woof/hex/include)")


def version() -> str:
    """``ad-binder --version``: crate version, AD source commit and whether the rule fatbins are compiled in."""
    done = subprocess.run([str(binary()), "--version"], capture_output=True, text=True, check=True)
    return done.stdout.strip()


def fatbins_present() -> bool:
    """False when the runner was built without NVCC: ``window`` and the C ABI then refuse at module load."""
    return "rule fatbins absent" not in version()
