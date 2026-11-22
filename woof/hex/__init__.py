"""GPU-native variable-resolution atmospheric model core.

A CUDA port of the MPAS-Atmosphere v8.4.2 dynamical core, with a NumPy CPU
authority beside it for the operators that are verified against
source-extracted fixtures.

Evidence is component- and configuration-specific; this package does not make
a package-wide numerical-equivalence, production-physics, or GPU claim.
Modules carry their own frozen-source citations and exact refusal boundaries.
"""

from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version
from types import MappingProxyType

from .errors import ConfigurationRefusal, EvidenceError, MeshValidationError

__all__ = [
    "COMPONENT_EVIDENCE",
    "DISTRIBUTION_NAME",
    "ConfigurationRefusal",
    "EvidenceError",
    "MeshValidationError",
    "__version__",
]

#: The distribution this package is published as.  It carries no MPAS token,
#: because that name may be a trademark; the import name still does, and that
#: gap is recorded as a pre-public blocker in
#: tests/test_packaging_declaration.py.
DISTRIBUTION_NAME = "recast-woof"

# Read back out of installed metadata rather than restated here.  pyproject's
# [project].version is the single place the number is written; a second
# literal is a promise to update two files at every cut, and woof broke
# exactly that promise -- its wheel shipped 0.1.1 for four releases while a
# refusal message told users which release was speaking.
try:
    __version__ = _distribution_version(DISTRIBUTION_NAME)
except PackageNotFoundError:  # pragma: no cover - uninstalled source tree
    # Says so rather than inventing a number.
    __version__ = "0+unknown"

__evidence__ = "component-specific"

# Every Rust and OpenMP child hex starts (vertical build, rw_mpas_init, static,
# remap) sizes its pool from RAYON_NUM_THREADS / OMP_NUM_THREADS.  Hold them to
# the container's CPU quota before any child starts: 257 threads on a 61-core
# quota stalled a 15 km init for an hour (cpu_budget.py has the measurement).
from .cpu_budget import bind_thread_defaults as _bind_thread_defaults

_bind_thread_defaults()

# WOOF_HEX_PAIR_ADD=plain (the adjoint door's recordings): the plain pair add is installed over the forecast's fused
# one here, so cuda_driver.py keeps the bytes the regional kernel set was minted on (pair_add_form.py says why).
# With the switch unset or fused nothing is imported.
from .pair_add_form import install_from_environment as _install_pair_add_form

_install_pair_add_form()

# Keep package metadata lightweight: importing ``woof.hex`` must not eagerly
# import the full driver, SciPy regridder, or NetCDF output stack.
COMPONENT_EVIDENCE: Mapping[str, str] = MappingProxyType(
    {
        "m1_operators": "source-extracted-fortran-oracle",
        "jw_nomix_whole_step": "frozen-jw-nomix-step-linked",
        "jw_original_mixed_whole_step": "implemented-original-jw-branch",
        "dry_24h": "actual repeated full-step Python-port execution",
        "gfs_initialization": "implemented-unverified",
        "gfs_6h_forecast": (
            "real GFS-initialized repeated full-step Python-port forecast"
        ),
        "history_output": "implemented-unverified",
        "visualization_regrid": "implemented-unverified; non-conservative",
        "rust_renderer_2d": "real unchanged-renderer product-path execution",
        "physics": "seam-only; no production physics",
    }
)
