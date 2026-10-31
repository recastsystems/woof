"""Every module the wheel imports is declared, and every declaration is used.

Both directions, because the project has been bitten by both.

*Imported but undeclared* is the ``pyshp`` defect: the demo renderer read the
vendored Natural Earth shapefiles with ``import shapefile`` for months while
no table anywhere named ``pyshp``, and it was found the only way it can be
found -- a fresh environment that installed exactly what the quickstart said,
then died on ``ModuleNotFoundError: No module named 'shapefile'`` after a
whole forecast and every DA cycle had already run.  ``huggingface_hub`` was
the same defect, still open, at the time this file was written.

*Declared but unimported* is the other half of the same inaccurate: a
requirement nobody consumes costs every user download bytes and install time
for nothing, and it makes the table unreadable -- a reviewer cannot tell which
lines are essential.

*In an extra when it should be in the base install* is the third, and it is
the one that actually removes features from users.  ``pip install recast-woof`` must
give a working product; a capability behind an extra that no documented
install line names does not exist.  Before this gate,
``pip install 'recast-woof[all-cu12]'`` -- the README's own recommended line --
shipped without velocity dealiasing, because ``scipy`` sat in an ``[obs]`` and
a ``[dealias]`` extra that appear in no user-facing document.

So the rule this file enforces is not "declare things".  It is:

1. every third-party top-level module the *shipped* code imports resolves to
   a distribution this project declares (:func:`test_every_imported_module_is_declared`);
2. that distribution is in the **base** ``dependencies`` unless the module is
   written down in :data:`_EXTRA_GATED` together with the extra that gates it
   and the reason -- so "put it behind an extra" is a decision someone has to
   record, not a default (:func:`test_extra_gating_is_declared_deliberately`);
3. nothing is declared that nothing imports
   (:func:`test_every_declaration_is_imported`);
4. the built distribution's own ``Requires-Dist`` metadata agrees with the
   table above, so a stale install cannot report green
   (:func:`test_installed_metadata_agrees_with_pyproject`).

The scan is an AST walk, not a text search, so a name inside a string or a
comment cannot satisfy it and a conditional import inside a function cannot
hide from it.

What "shipped" means
--------------------
Exactly what the wheel carries: ``woof`` and every subpackage, plus the
*top level* of ``tools`` -- ``[tool.setuptools.packages.find]`` names
``tools`` and not ``tools*``, so ``tools/ftz_receipt/`` and its siblings are
build-time material that never reaches a user.  :func:`_shipped_sources`
re-derives that from ``pyproject.toml`` rather than hard-coding it, and
:func:`test_the_shipped_surface_matches_the_packaging_declaration` fails if
the include list changes shape underneath it.

Two of the include entries are DATA anchors that carry no import surface --
``configs*`` and ``docs`` -- and both are asserted to keep carrying none, so
neither can grow Python without somebody deciding it should.

``tests/`` is scanned too, but only to justify the ``[dev]`` extra: the test
suite is the only consumer ``pytest`` has.
"""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from functools import lru_cache
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"

#: The distribution's own name, in every spelling that can appear.
_SELF = "recast-woof"

#: Packages that live in this repository.  A bare import of one of these is a
#: cross-import question, not a dependency question (see
#: ``test_no_shipped_module_imports_a_sibling_by_bare_name``).
_FIRST_PARTY = frozenset({"woof", "tools", "tilestream",
                          # The cycle spine's port-facing seam, top-level
                          # on purpose (pyproject includes it and says
                          # why): the forecast worker must reach it
                          # without the woof name.
                          "mpas_cycle_bridge"})

#: Every third-party top-level module the shipped code is allowed to name,
#: mapped to the distribution(s) that can provide it.  A module that reaches
#: the scanner without a row here fails the test rather than passing quietly:
#: the point is that adding an import forces a decision about who ships it.
_PROVIDERS: dict[str, tuple[str, ...]] = {
    # The folded hex model's own third-party imports (woof/hex); each is gated below.
    "boto3": ("boto3",),
    "cfgrib": ("cfgrib",),
    "cftime": ("cftime",),
    "mpi4py": ("mpi4py",),
    "tomli": ("tomli",),
    "h5py": ("h5py",),
    "rustmatrix": ("rustmatrix",),
    "PIL": ("pillow",),
    "affine": ("affine",),
    "cupy": ("cupy-cuda12x", "cupy-cuda13x"),
    "cupy_backends": ("cupy-cuda12x", "cupy-cuda13x"),
    "cupyx": ("cupy-cuda12x", "cupy-cuda13x"),
    "cuda": ("cuda-pathfinder",),
    "huggingface_hub": ("huggingface_hub",),
    "jsonschema": ("jsonschema",),
    "threadpoolctl": ("threadpoolctl",),
    "matplotlib": ("matplotlib",),
    "mcp": ("mcp",),
    "netCDF4": ("netCDF4",),
    "numpy": ("numpy",),
    "yaml": ("pyyaml",),
    "packaging": ("packaging",),
    "psutil": ("psutil",),
    "pyproj": ("pyproj",),
    "pytest": ("pytest",),
    "rasterio": ("rasterio",),
    "scipy": ("scipy",),
    "shapefile": ("pyshp",),
    "py7zr": ("py7zr",),
    "wrf": ("wrf-rust",),
    # The companion distribution, split out of this one in 2.5.0 when the
    # woof wheel measured 103.62 MiB against PyPI's 100 MiB per-file cap.
    # It carries the RRTMGP and Thompson table directories; woof pins it
    # `==` and woof.data_assets resolves through it.
    "woof_data": ("recast-woof-data",),
    # NO distribution provides this: it is the repository's own mapped
    # parity battery (tests/test_mapped_engine_parity.py), and tests/ does
    # not ship.  The two decode instruments that import it are
    # checkout-only by construction -- see _OPTIONAL_BY_DESIGN.
    "test_mapped_engine_parity": (),
    # NO distribution provides these either: tools/wrf_exact_localize's trace
    # hooks and comparer, reached by the exact chain's localization
    # instruments after a sys.path insert -- see _OPTIONAL_BY_DESIGN.
    "cmpphys": (),
    "woof_hook": (),
    "woof_hook2": (),
    "woof_hook3": (),
    "woof_gf_inputs": (),
    "woof_wsm6_inputs": (),
    "arwen_global": ("arwen-global",),
    "cdsapi": ("cdsapi",),
    "setuptools": ("setuptools",),
    "xdist": ("pytest-xdist",),
    # tests/test_no_undefined_names.py locates the linter through the
    # wheel's own module (ruff.__main__.find_ruff_bin).
    "ruff": ("ruff",),
    # The Earth-2 doors (2.9.3): the runner scripts execute in the separate
    # earth2studio interpreter WOOF_EARTH2_PYTHON names -- see
    # _OPTIONAL_BY_DESIGN.
    "earth2studio": ("earth2studio",),
    "hexcore": ("woof hex",),
    "torch": ("torch",),
    "torch_harmonics": ("torch-harmonics",),
    "xarray": ("xarray",),
    # tools/hf_inventory.py, a maintainer inventory instrument -- see
    # _OPTIONAL_BY_DESIGN.
    "requests": ("requests",),
}

#: Modules this project deliberately never declares, because a *declared*
#: dependency is contractually obliged to install them.  Naming them ourselves
#: would pin a version we do not care about and would hide the real
#: relationship.  If the guarantor ever leaves the table, this fires.
_GUARANTORS: dict[str, tuple[str, str]] = {
    "cftime": (
        "netcdf4",
        "netCDF4 requires cftime, so a base install always has it. The hex model's serial NetCDF guards "
        "(woof/hex/multi_member.py serial_netcdf, woof/hex/drivers/samehost_serial_io_v841.py) wrap "
        "cftime's date functions beside netCDF4's and read it through that guarantee.",
    ),
    "cuda": (
        "cupy-cuda13x",
        "The CuPy 14 wheels (cupy-cuda13x and cupy-cuda12x alike, both pinned "
        ">=14.0 by the gpu extras) require cuda-pathfinder, which ships the "
        "`cuda.pathfinder` namespace.  woof/forecast_readiness.py imports it "
        "inside binding(), only after it has established CuPy 14, and reads "
        "it through CuPy's guarantee, never on its own account.",
    ),
    "PIL": (
        "matplotlib",
        "matplotlib requires pillow>=8, so a base install always has it. "
        "woof/pair_compose.py imports PIL inside its functions and reads it "
        "through matplotlib's guarantee, never on its own account.",
    ),
    "affine": (
        "rasterio",
        "rasterio requires affine, so wherever rasterio resolves affine does "
        "too. woof/static/highres.py reads it through that guarantee rather "
        "than pinning a version of somebody else's internal transform type. "
        "Both are now [geog]-gated together (see _EXTRA_GATED): the one "
        "function that touches affine is the pure-Python warp fallback's "
        "raster geometry, which cannot run without rasterio anyway.",
    ),
}

#: Consumers an AST walk cannot see, because the import lives inside a
#: *string* that some other process executes.  Recorded rather than waved
#: through: :func:`test_generated_source_consumers_still_exist` re-checks that
#: the named file still contains the named import, so deleting the consumer
#: still trips the gate.
_GENERATED_SOURCE_CONSUMERS: dict[str, tuple[str, str, str]] = {
    "psutil": (
        "tests/test_prepare_hrrr_wrf.py",
        "import psutil",
        "_fixture_decoder writes a decoder script with textwrap.dedent and "
        "launches it as a real process. On Windows that script walks its own "
        "ancestry to find the cmd.exe shim Popen returned, because the "
        "executable path is a .cmd. The import is inside the generated "
        "source, so no walk of the test file's own AST can see it.",
    ),
    "woof_data": (
        "woof/data_assets.py",
        'COMPANION_PACKAGE = "woof_data"',
        "woof.data_assets resolves the companion with "
        "importlib.resources.files(COMPANION_PACKAGE), so the package name "
        "is a string constant and never an import statement -- deliberately, "
        "because the module is bytes on disk and importing it would buy "
        "nothing. No AST walk can see that, and without this row the "
        "recast-woof-data pin reads as a declared dependency nobody consumes.",
    ),
    "xdist": (
        "tools/battery/no_silent_deselection.py",
        "pytest-xdist",
        "The suite runs under `pytest -n` in the battery and in publish.yml, "
        "and pytest loads xdist through its plugin entry point, so no file "
        "imports it by name. The deselection guard names the plugin because "
        "its controller process holds no collected items, which is the one "
        "place the dependency is consumed in source.",
    ),
}

#: Modules that are genuinely optional at run time: the importing code has a
#: try/except and a documented answer for the absent case, so declaring them
#: would be declaring something the product does not need.
#: Why the Earth-2 runner scripts' imports are not declared (2.9.3).
_EARTH2_INTERPRETER = (
    "the Earth-2 doors (2.9.3): woof/earth2_runner.py, "
    "earth2_diagnose_runner.py, earth2_diagnostic_runner.py and earth2_io.py "
    "are the scripts woof earth2 and woof earth2-diagnose launch as a "
    "subprocess in the interpreter WOOF_EARTH2_PYTHON names, the one with "
    "earth2studio and its torch stack installed. The woof interpreter never "
    "imports them, and without that interpreter the doors refuse by naming "
    "the variable. Declaring the stack here would pull a multi-GiB torch "
    "install into every woof install for a door that runs beside it, not "
    "inside it."
)

_OPTIONAL_BY_DESIGN: dict[str, str] = {
    "boto3": (
        "woof/hex/analysis_bundle.py tries boto3 inside a try/except for an s3:// analysis bundle and "
        "falls back to the HTTPS route without it; no default door needs it."
    ),
    "cfgrib": (
        "woof/hex/initialization.py imports cfgrib only for an explicitly selected engine='cfgrib' GRIB "
        "grouping and refuses by name (cfgrib_runtime) when it is absent; the default route decodes "
        "through the Rust bridges."
    ),
    "mpi4py": (
        "woof/hex/mpasphys/launch_v842.py imports mpi4py inside worker(), the private MPI worker of a "
        "multi-rank native MPAS reference run launched under mpiexec; the GPU forecast route never "
        "reaches it."
    ),
    "tomli": (
        "woof/hex/nest_triggers.py falls back to tomli only when tomllib is missing, which is Python "
        "below 3.11; the distribution requires 3.11 or newer, so the fallback never runs."
    ),
    "rustmatrix": (
        "tools/build_radar_scattering_tables.py (labs/radar-bands, 2.9.11) is the "
        "maintainer build that regenerates the shipped C/X-band scattering tables "
        "(bowecho_simradar data/scattering-*.bin) with the T-matrix package; no "
        "forecast, preparation or radar door imports it, and the tables it writes "
        "ship as data."
    ),
    "hexcore": (
        "woof/output_formats.py imports woof.hex.forecast_door only for a store "
        "whose engine is woof hex (self.engine == 'woof hex'), i.e. on a "
        "machine where woof hex wrote the run; woof hex depends on woof, not "
        "the reverse, and a WOOF store never reaches the import."
    ),
    "earth2studio": _EARTH2_INTERPRETER,
    "torch": _EARTH2_INTERPRETER,
    "xarray": _EARTH2_INTERPRETER,
    "torch_harmonics": (
        "woof/earth2_runner.py imports torch_harmonics inside a try/except "
        "(and the runner itself runs in WOOF_EARTH2_PYTHON): the banded DISCO "
        "contraction is a memory optimisation and a no-op when "
        "torch-harmonics is absent."
    ),
    "requests": (
        "tools/hf_inventory.py is the read-only Hugging Face Earth-2 "
        "inventory a maintainer runs from a checkout to rebuild the model "
        "table. No woof door imports it (woof hf-fetch downloads with "
        "urllib), so a user's install never reaches it."
    ),
    "cmpphys": (
        "the WRF exact chain's localization instruments (2.9.1): tools/coupled_exact_trace.py, tools/coupled_gf_receipt.py and tools/coupled_trace_receipt.py put tools/wrf_exact_localize on sys.path and import its trace hooks and comparer by bare name. That directory does not ship (packages.find names `tools`, not `tools*`), so from a wheel the import fails by name; the scripts are maintainer instruments that replay a recorded WRF run from a checkout, and no wheel-user path reaches them."
    ),
    "woof_hook": (
        "the WRF exact chain's localization instruments (2.9.1): tools/coupled_exact_trace.py, tools/coupled_gf_receipt.py and tools/coupled_trace_receipt.py put tools/wrf_exact_localize on sys.path and import its trace hooks and comparer by bare name. That directory does not ship (packages.find names `tools`, not `tools*`), so from a wheel the import fails by name; the scripts are maintainer instruments that replay a recorded WRF run from a checkout, and no wheel-user path reaches them."
    ),
    "woof_hook2": (
        "the WRF exact chain's localization instruments (2.9.1): tools/coupled_exact_trace.py, tools/coupled_gf_receipt.py and tools/coupled_trace_receipt.py put tools/wrf_exact_localize on sys.path and import its trace hooks and comparer by bare name. That directory does not ship (packages.find names `tools`, not `tools*`), so from a wheel the import fails by name; the scripts are maintainer instruments that replay a recorded WRF run from a checkout, and no wheel-user path reaches them."
    ),
    "woof_hook3": (
        "the WRF exact chain's localization instruments (2.9.1): tools/coupled_exact_trace.py, tools/coupled_gf_receipt.py and tools/coupled_trace_receipt.py put tools/wrf_exact_localize on sys.path and import its trace hooks and comparer by bare name. That directory does not ship (packages.find names `tools`, not `tools*`), so from a wheel the import fails by name; the scripts are maintainer instruments that replay a recorded WRF run from a checkout, and no wheel-user path reaches them."
    ),
    "woof_gf_inputs": (
        "the WRF exact chain's localization instruments (2.9.1): tools/coupled_exact_trace.py, tools/coupled_gf_receipt.py and tools/coupled_trace_receipt.py put tools/wrf_exact_localize on sys.path and import its trace hooks and comparer by bare name. That directory does not ship (packages.find names `tools`, not `tools*`), so from a wheel the import fails by name; the scripts are maintainer instruments that replay a recorded WRF run from a checkout, and no wheel-user path reaches them."
    ),
    "woof_wsm6_inputs": (
        "the WRF exact chain's localization instruments (2.9.1): tools/coupled_exact_trace.py, tools/coupled_gf_receipt.py and tools/coupled_trace_receipt.py put tools/wrf_exact_localize on sys.path and import its trace hooks and comparer by bare name. That directory does not ship (packages.find names `tools`, not `tools*`), so from a wheel the import fails by name; the scripts are maintainer instruments that replay a recorded WRF run from a checkout, and no wheel-user path reaches them."
    ),
    "test_mapped_engine_parity": (
        "tools/extract_mapped_engine_goldens.py and "
        "tools/mapped_engine_parity_sweep.py import the mapped parity "
        "battery beside the goldens it measures, both under tests/, which "
        "does not ship. Both are maintainer instruments -- one WRITES the "
        "committed goldens, the other sweeps the two engines over staged "
        "private bytes -- so no wheel-user path can reach them. The import "
        "is guarded and refuses from a wheel by naming the checkout it "
        "needs."
    ),
    "packaging": (
        "woof/version_cli.py:_is_behind imports packaging.version inside a "
        "try/ImportError and falls back to an equality comparison it "
        "documents in the except branch ('different is not behind'). The "
        "command is correct without it. matplotlib happens to bring it, but "
        "the code does not rely on that."
    ),
    "arwen_global": (
        "The local DA doors read the separately published arwen-global "
        "distribution's observation streams only when it is installed: "
        "woof/local_da_observations.py asks importlib.util.find_spec for it "
        "before importing, and woof/local_da_fetch.py imports it inside the "
        "neutral-stream route branch that only exists for a stream that "
        "inspection registered. Without the package the doors list no such "
        "route and refuse nothing."
    ),
    "setuptools": (
        "tools/promote_prepared_release.py enumerates the expected wheel "
        "contents with setuptools' own package finder, inside the function "
        "that does it. The controller runs only in publish.yml, which "
        "installs setuptools>=77 explicitly beside twine before calling it; "
        "a user's install never reaches that function."
    ),
}

#: Modules whose distribution is allowed to sit in an extra rather than in the
#: base ``dependencies``, with the extras that gate them and why.  Everything
#: NOT listed here must be reachable from a bare ``pip install recast-woof``.
#:
#: This is the table the rule lives in.  Adding a row is how you say "this
#: capability is genuinely optional"; it is deliberately more work than
#: promoting the dependency.
_EXTRA_GATED: dict[str, tuple[tuple[str, ...], str]] = {
    "py7zr": (
        ("hydro-conus",),
        "Decodes the CONUS NHDPlus V2.1 .7z source archives only, inside an "
        "explicit --hydro run on a CONUS domain. py7zr and its codecs are "
        "LGPL-2.1-or-later, outside the permissive licences a bare install "
        "carries; without it the door prepares the worldwide fabric and says why.",
    ),
    "cuda": (
        ("gpu-cu13",),
        "Reachable only through CuPy 14's own requirement on cuda-pathfinder "
        "(the same guarantee under [gpu-cu12], whose cupy-cuda12x>=14.0 "
        "carries it), inside the forecast-readiness binding that runs only "
        "beside a card; a base install never imports it.",
    ),
    "cupy": (
        ("gpu-cu12", "gpu-cu13"),
        "CuPy ships one wheel per CUDA major and a pip extra cannot detect "
        "which major the box serves, so the choice is named rather than "
        "defaulted. woof doctor reads the real major off the driver and "
        "prints the matching extra.",
    ),
    "cupy_backends": (
        ("gpu-cu12", "gpu-cu13"),
        "Ships inside the same CuPy wheel as `cupy`.",
    ),
    "cupyx": (
        ("gpu-cu12", "gpu-cu13"),
        "Ships inside the same CuPy wheel as `cupy`.",
    ),
    "rasterio": (
        ("geog",),
        "The high-resolution warp substrate -- GeoTIFF decode, mosaic, void "
        "fill, clip, the reproject onto the WPS spherical grid -- runs in "
        "the Rust static-fields library on the bare default of "
        "[static.highres]. That library is a bundled artifact a wheel "
        "install stages, so a base install builds high-resolution terrain "
        "with no rasterio anywhere. What [geog] buys is the PARITY "
        "FALLBACK the WOOF_STATIC_PYTHON=1 workaround runs on. "
        "tests/test_static_highres_warp_routing.py is the gate: it makes "
        "rasterio unimportable and requires every default high-resolution "
        "call to still answer.",
    ),
    "pyproj": (
        ("geog",),
        "Same seam as rasterio: the CRS construction and the point "
        "transforms belong to the Rust substrate now, and pyproj is what "
        "the pure-Python parity fallback projects with.",
    ),
    "affine": (
        ("geog",),
        "Ships with rasterio (see _GUARANTORS) and is read by the same "
        "fallback body, so it is gated by the same extra.",
    ),
    "wrf": (
        ("render",),
        "wrf-rust is the mandated science core for derived quantities. The "
        "default rust render engine does not need it -- it renders from the "
        "rw_wrfbatch bridge -- so a base install still draws maps.",
    ),
    "mcp": (
        ("mcp",),
        "The MCP SDK is what the arwen-mcp stdio server speaks the "
        "protocol with. The server is an agent integration surface, not "
        "a forecast capability: no woof door and no data path imports "
        "it, so a person's install does not carry it. The import is "
        "guarded in woof/mcp/__init__.py with a one-sentence remedy "
        "naming `pip install recast-woof[mcp]`.",
    ),
    "huggingface_hub": (
        ("publish",),
        "tools/publish_geog_mirror.py `upload` publishes the WPS_GEOG mirror "
        "to Hugging Face. That is a maintainer action -- it needs write "
        "credentials on a dataset repository nobody else owns -- so no user "
        "path can reach it. The import is guarded and names this extra.",
    ),
    "h5py": (
        ("publish",),
        "tools/harvest_radar_heights.py fills the antenna heights in the "
        "frozen European radar site table by reading /where/height out of "
        "one polar volume per radar. Rebuilding that table is a maintainer "
        "action; the wheel ships it frozen and a user only reads it. No user "
        "path opens HDF5 from Python -- the product decodes ODIM through "
        "rw_odim, which carries its own pure-Rust reader -- so this is a "
        "build-the-table dependency, not a read-the-table one. The import is "
        "guarded and names this extra.",
    ),
    "yaml": (
        ("dev",),
        "2.9.0 (r290/publish-test-job): tools/workflow_job_replay.py and "
        "tools/workflow_rehearsals.py read the repository's own workflow "
        "YAML to replay a CI job and census the release rehearsals before a "
        "publish. Both are release-engineering tools run from a checkout with "
        "the [dev] extra, which already declares pyyaml for the publish.yml "
        "contract tests; no forecast or data door imports them.",
    ),
    "pytest": (
        ("dev",),
        "The test suite. tilestream/ ships 41 `test_*.py` modules inside the "
        "wheel and one of them imports pytest at module level, so this is "
        "the one gated module that a user's site-packages does contain -- "
        "but nothing outside those files imports them, and only `pytest` "
        "collects them. Shipping them at all is the tiles lane's call.",
    ),
}

#: Extras that carry no requirements of their own and exist only so an
#: install line somebody already wrote keeps resolving.  Each says what it
#: used to mean and where that moved.
_NEUTRALISED_EXTRAS: dict[str, str] = {
    "obs": (
        "was scipy>=1.11, for the observation-battery referee's cKDTree. "
        "scipy is a base dependency now: `pip install recast-woof` scores against "
        "observations. Kept so `pip install 'recast-woof[obs]'` still resolves."
    ),
    "dealias": (
        "was scipy>=1.11, for velocity dealiasing's connected-components "
        "region labelling -- the DEFAULT engine. scipy is a base dependency "
        "now. Kept so `pip install 'recast-woof[dealias]'` still resolves."
    ),
}

#: Shipped files that reach a sibling module by bare name after putting its
#: directory on ``sys.path`` themselves.  Measured to resolve from an
#: installed wheel (``<site-packages>/tools`` exists, because ``tools`` ships
#: flat), so these are recorded rather than broken -- but each is a standing
#: invitation to a shadowed import, and the row is what makes adding another
#: one a decision.
_SYS_PATH_SIBLING_SITES: dict[str, str] = {
    "tools/coupled_wsm6_receipt.py": (
        "the exact chain's WSM6 localization receipt (2.9.1): inserts its own "
        "directory and imports coupled_dycore_receipt, which ships beside it. "
        "A checkout instrument."
    ),
    "tools/coupled_dycore_receipt.py": (
        "the exact chain's localization receipt (2.9.1): inserts its own directory and "
        "imports coupled_trace_receipt, which ships beside it in tools/; it also "
        "inserts tools/wrf_exact_localize, which does not ship. A checkout instrument."
    ),
    "tools/coupled_reflectivity_probe.py": (
        "the exact chain's reflectivity probe (2.9.1): inserts its own directory and "
        "imports coupled_dycore_receipt and coupled_trace_receipt, which ship beside it. "
        "A checkout instrument."
    ),
    "tools/nssl2_variant_probe.py": (
        "inserts its own directory and imports nssl2_mp18_digest_probe. The "
        "comment above it says the insert is deliberate because "
        "PYTHONSAFEPATH=1 keeps the script directory off sys.path; it pops "
        "nothing, so the 142 names stay ahead of the stdlib for the rest of "
        "the process."
    ),
    "tools/perf_dealias_hotspot.py": (
        "same shape, for perf_obs_timing's shared box definition. Also does "
        "not pop."
    ),
    "tools/render_static_terrain.py": (
        "inserts tools/wt-intl and tools/wt-intl/tools -- neither of which "
        "ships, because packages.find names `tools` and not `tools*` -- then "
        "imports terrain_source_crossvalidation. This one genuinely cannot "
        "resolve from a wheel, but it also reads sys.argv[1] at import time, "
        "so it is a checkout-only script by construction. Terrain lane's "
        "file; handed off rather than changed here."
    ),
}

_REQUIREMENT_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")
_SELF_REFERENCE = re.compile(
    rf"^\s*{_SELF}\s*\[\s*([A-Za-z0-9._,\s-]+)\s*\]\s*$")


def _canonical(name: str) -> str:
    """PEP 503 normalisation, so netCDF4 and netcdf4 are one distribution."""

    return re.sub(r"[-_.]+", "-", name).lower()


@pytest.fixture(scope="module")
def pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# The shipped surface
# --------------------------------------------------------------------------
def _shipped_sources(config: dict) -> list[Path]:
    """Every ``.py`` the wheel carries, derived from the include list."""

    include = config["tool"]["setuptools"]["packages"]["find"]["include"]
    files: list[Path] = []
    for pattern in include:
        if pattern.endswith(".*"):          # e.g. "woof.*": the SUBpackages
            root = REPO_ROOT.joinpath(*pattern[:-2].split("."))
            files.extend(sorted(path for path in root.rglob("*.py")
                                if path.parent != root))
        elif pattern.endswith("*"):         # e.g. "tilestream*": tree
            root = REPO_ROOT / pattern[:-1]
            files.extend(sorted(root.rglob("*.py")))
        else:                               # e.g. "tools": that package only
            root = REPO_ROOT / pattern
            files.extend(sorted(root.glob("*.py")))
    return files


@lru_cache(maxsize=None)
def _top_level_imports(path: Path):
    """Immutable import census, shared by the checks in each worker."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((alias.name.split(".")[0], node.lineno)
                           for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            imports.append((node.module.split(".")[0], node.lineno))
    return tuple(imports)


#: Shipped tools that reach a module of the TEST TREE by bare name after
#: putting ``tests/`` on ``sys.path`` themselves.  Both recapture GPU pins
#: whose registries (``tests/_card_pins.py``, ``tests/_phase2_pin.py``)
#: live beside the tests that read them, so they are checkout-only by
#: construction: nothing under ``tests/`` ships.  Recorded here so the
#: gate does not read the two names as distributions nobody declared
#: (which is how they showed up on the Linux release node,
#: proof/node-reds-276), and checked by
#: :func:`test_every_test_tree_import_site_is_live` so a row cannot outlive
#: the import it excuses.
_TEST_TREE_IMPORT_SITES: dict[str, tuple[str, ...]] = {
    "tools/recapture_card_pins.py": ("_card_pins",),
    "tools/recapture_phase2_pin.py": ("_phase2_pin",),
    # The legacy RRTMG speed lane's node-side prep check (668fa4c56): it
    # drives tests/test_rrtmg_legacy_prep_device.py through pytest, and its
    # --timing arm imports that suite's synthetic-input and bit-compare
    # helpers so the timing runs on the inputs the suite certifies.
    # Checkout-only by the same construction as the two above.
    "tools/rrtmg_prep_device_check.py": ("test_rrtmg_legacy_prep_device",),
    # The packed DA member-leg proof (69a8b88ad, 098816bc9): it binds the
    # established public GPU fixture -- the cycle-join and nested-forecast
    # suites' experiment, wiring, terrain and land builders -- and hashes
    # those two suites' bytes as the proof's source identity, so the proof
    # and the suites cannot drift apart.  Its GPU arms refuse without the
    # node operator's OWNER check and one complete card UUID, and the import
    # sits inside that guard (_fixture_imports).  Checkout-only by the same
    # construction as the three above.
    "tools/da_member_leg_proof.py": ("test_da_cycle_join_gpu",
                                     "test_da_nested_forecast_gpu"),
    # These numerical review tools explicitly insert the checkout's tests
    # directory. Their fixture helpers are verified by the liveness gate
    # below and are not runtime dependencies of an installed distribution.
    "tools/mynn_exact_review.py": ("_mynn_families", "_mynn_families_gsd41"),
    "tools/mynn_tkeadvect_check.py": ("_mynn_families", "_mynn_families_gsd41"),
    "tools/mynn_review_restart.py": ("test_mynn_pbl_runtime",),
    "tools/terrain_clock_review_bench.py": ("test_terrain_clock_review",),
    # The correct-mode water-budget receipt (2.9.0 7abd12219) runs the same
    # WSM6 and New Tiedtke columns as tests/test_correct_mode_water_budget.py
    # through that suite's mirror and residual helpers, so the receipt and
    # the suite cannot measure different columns.  Checkout-only by the same
    # construction as the rows above.
    "tools/correct_mode_water_budget_receipt.py": ("test_correct_mode_water_budget",),
}


def _third_party_imports(paths, siblings=frozenset()) -> dict[str, list[str]]:
    """Module -> the ``file:line`` sites that import it.

    ``siblings`` are module names that live in this repository.  A bare import
    of one of those is a cross-import question rather than a dependency
    question, and belongs to
    :func:`test_no_shipped_module_imports_a_sibling_by_bare_name`; a name that
    is both a sibling and a real distribution stays a dependency question.
    A name recorded in :data:`_TEST_TREE_IMPORT_SITES` for the importing
    file is a test-tree module, not a distribution.
    """

    stdlib = set(sys.stdlib_module_names)
    sites: dict[str, list[str]] = {}
    for path in paths:
        rel = path.relative_to(REPO_ROOT).as_posix()
        for name, lineno in _top_level_imports(path):
            if name in stdlib or name in _FIRST_PARTY:
                continue
            if name in siblings and name not in _PROVIDERS:
                continue
            if name in _TEST_TREE_IMPORT_SITES.get(rel, ()):
                continue
            sites.setdefault(name, []).append(f"{rel}:{lineno}")
    return sites


def test_every_test_tree_import_site_is_live():
    """A recorded test-tree import excuses exactly the import it names."""

    for rel, names in sorted(_TEST_TREE_IMPORT_SITES.items()):
        path = REPO_ROOT / rel
        assert path.is_file(), f"{rel} is recorded and does not exist"
        source = path.read_text(encoding="utf-8")
        assert "sys.path.insert" in source and '"tests"' in source, (
            f"{rel} is recorded as reaching the test tree through a "
            "sys.path insert of tests/, and it does not insert that directory")
        imported = {name for name, _ in _top_level_imports(path)}
        for name in names:
            assert name in imported, (
                f"{rel} no longer imports {name!r}; delete it from the row")
            assert (REPO_ROOT / "tests" / f"{name}.py").is_file(), (
                f"{name!r} is recorded as a test-tree module and "
                f"tests/{name}.py does not exist")


def _sibling_module_names(config: dict) -> frozenset[str]:
    return frozenset(p.stem for p in _shipped_sources(config))


@pytest.fixture(scope="module")
def shipped_imports(pyproject) -> dict[str, list[str]]:
    return _third_party_imports(_shipped_sources(pyproject),
                                _sibling_module_names(pyproject))


@pytest.fixture(scope="module")
def test_suite_imports(pyproject) -> dict[str, list[str]]:
    tests = sorted((REPO_ROOT / "tests").rglob("*.py"))
    return _third_party_imports(
        tests, _sibling_module_names(pyproject)
        | frozenset(p.stem for p in tests))


# --------------------------------------------------------------------------
# The declaration
# --------------------------------------------------------------------------
def _requirement_distribution(requirement: str) -> str | None:
    match = _REQUIREMENT_NAME.match(requirement)
    return _canonical(match.group(1)) if match else None


def _resolve_extra(name: str, optional: dict[str, list[str]],
                   seen: frozenset[str] = frozenset()) -> set[str]:
    """Distributions an extra pulls, following ``recast-woof[other]`` aliases."""

    if name in seen:
        return set()
    seen = seen | {name}
    resolved: set[str] = set()
    for requirement in optional.get(name, []):
        alias = _SELF_REFERENCE.match(requirement)
        if alias:
            for referenced in alias.group(1).split(","):
                resolved |= _resolve_extra(referenced.strip(), optional, seen)
            continue
        distribution = _requirement_distribution(requirement)
        if distribution and distribution != _canonical(_SELF):
            resolved.add(distribution)
    return resolved


@pytest.fixture(scope="module")
def declaration(pyproject) -> dict[str, object]:
    project = pyproject["project"]
    optional = project.get("optional-dependencies", {})
    base = {d for d in (_requirement_distribution(r)
                        for r in project.get("dependencies", [])) if d}
    extras = {name: _resolve_extra(name, optional) for name in optional}
    return {"base": base, "extras": extras,
            "all": base | set().union(*extras.values(), set())}


def _providers(module: str) -> tuple[str, ...]:
    return tuple(_canonical(d) for d in _PROVIDERS.get(module, ()))


# --------------------------------------------------------------------------
# Direction 1: imported -> declared
# --------------------------------------------------------------------------
def test_every_shipped_import_has_a_known_provider(shipped_imports):
    """No module reaches the gate without someone having said who ships it."""

    unknown = {m: s for m, s in shipped_imports.items() if m not in _PROVIDERS}
    assert not unknown, (
        "these modules are imported by the wheel and appear in no table in "
        "this file. Add a row to _PROVIDERS naming the distribution that "
        "supplies each, then declare it in pyproject.toml (or record it in "
        "_GUARANTORS / _OPTIONAL_BY_DESIGN with the reason):\n"
        + "\n".join(f"  {m}: {sites[0]}"
                    + (f" (+{len(sites) - 1} more)" if len(sites) > 1 else "")
                    for m, sites in sorted(unknown.items())))


def test_every_imported_module_is_declared(shipped_imports, declaration):
    """The pyshp defect, as a gate."""

    undeclared: dict[str, list[str]] = {}
    for module, sites in sorted(shipped_imports.items()):
        if module in _OPTIONAL_BY_DESIGN:
            continue
        if module in _GUARANTORS:
            guarantor = _canonical(_GUARANTORS[module][0])
            assert guarantor in declaration["all"], (
                f"{module} is declared nowhere because {guarantor} is "
                f"supposed to guarantee it, but {guarantor} is not declared "
                f"either. Sites: {sites}")
            continue
        if not (set(_providers(module)) & declaration["all"]):
            undeclared[module] = sites
    assert not undeclared, (
        "imported by the shipped wheel, declared in no dependency table:\n"
        + "\n".join(
            f"  {m} (from {'/'.join(_providers(m))}) at " + ", ".join(s[:3])
            for m, s in undeclared.items()))


def test_extra_gating_is_declared_deliberately(shipped_imports, declaration):
    """`pip install recast-woof` must give a working product.

    A module the wheel imports is a base dependency unless someone wrote down
    which extra gates it and why.  This is the gate that would have caught
    scipy sitting in an undocumented ``[obs]``/``[dealias]`` pair.
    """

    should_be_base: dict[str, list[str]] = {}
    for module, sites in sorted(shipped_imports.items()):
        if module in _OPTIONAL_BY_DESIGN:
            continue
        if module in _EXTRA_GATED:
            gates, reason = _EXTRA_GATED[module]
            assert reason.strip(), f"{module}'s gating reason is empty"
            # A guaranteed module is supplied by its guarantor, so the extra
            # has to pull the guarantor rather than the module's own name.
            wanted = ({_canonical(_GUARANTORS[module][0])}
                      if module in _GUARANTORS else set(_providers(module)))
            for gate in gates:
                assert gate in declaration["extras"], (
                    f"{module} is recorded as gated behind [{gate}], which "
                    f"pyproject.toml does not define")
                assert wanted & declaration["extras"][gate], (
                    f"{module} is recorded as gated behind [{gate}], but "
                    f"[{gate}] does not pull {' or '.join(sorted(wanted))}")
            continue
        if module in _GUARANTORS:
            continue
        if not (set(_providers(module)) & declaration["base"]):
            should_be_base[module] = sites
    assert not should_be_base, (
        "these are reachable from the shipped code but not from a bare "
        "`pip install recast-woof`. Either move the distribution into [project]."
        "dependencies, or add a row to _EXTRA_GATED naming the extra that "
        "gates it and why that capability is genuinely optional:\n"
        + "\n".join(f"  {m} (from {'/'.join(_providers(m))}) at "
                    + ", ".join(s[:3]) for m, s in should_be_base.items()))


# --------------------------------------------------------------------------
# Direction 2: declared -> imported
# --------------------------------------------------------------------------
def test_every_declaration_is_imported(shipped_imports, test_suite_imports,
                                       declaration, pyproject):
    """Nothing is downloaded for a consumer that does not exist."""

    consumed = (shipped_imports.keys() | test_suite_imports.keys()
                | _GENERATED_SOURCE_CONSUMERS.keys())
    wanted = set()
    for module in consumed:
        wanted |= set(_providers(module))
    for module in _GUARANTORS:                      # guarantors are consumed
        wanted.add(_canonical(_GUARANTORS[module][0]))

    unused = sorted(declaration["all"] - wanted)
    assert not unused, (
        "declared in pyproject.toml, imported by nothing in gpuwm/, tools/ "
        "or tests/. Remove the line, or wire the dependency -- do not leave "
        "a third state:\n" + "\n".join(f"  {d}" for d in unused))


def test_generated_source_consumers_still_exist():
    """A dependency justified by a string is justified by a string that exists."""

    for module, (relative, needle, reason) in sorted(
            _GENERATED_SOURCE_CONSUMERS.items()):
        path = REPO_ROOT / relative
        assert path.is_file(), (
            f"{module} is declared because {relative} generates source that "
            f"imports it, and {relative} is gone: {reason}")
        assert needle in path.read_text(encoding="utf-8"), (
            f"{module} is declared because {relative} generates source "
            f"containing {needle!r}, and it no longer does. Either the "
            f"consumer moved -- update this table -- or the dependency is "
            f"dead and belongs out of pyproject.toml. Reason on record: "
            f"{reason}")


def test_neutralised_extras_are_still_offered(declaration, pyproject):
    """An install line somebody already wrote must keep resolving."""

    optional = pyproject["project"].get("optional-dependencies", {})
    for name, reason in _NEUTRALISED_EXTRAS.items():
        assert name in optional, (
            f"[{name}] was neutralised, not deleted: {reason}")
        assert not declaration["extras"][name], (
            f"[{name}] is recorded as neutralised but still pulls "
            f"{sorted(declaration['extras'][name])}; either it is not "
            f"neutralised or the record is stale")
        assert reason.strip()


def test_all_covers_every_extra_a_forecast_user_needs(declaration, pyproject):
    """``[all]`` must mean what the docs say it means.

    Stated as an invariant rather than an enumeration, deliberately: every
    extra except the CUDA-major alternatives, the maintainer publishing extra
    and the test extra must be *covered* by ``[all-*]``.  An empty extra is
    covered trivially, so this passes today for ``geog``/``obs``/``dealias``
    without ``[all]`` having to name three no-ops -- and it starts failing
    the moment one of them regains a requirement, which is the failure that
    2.3.2 shipped.
    """

    optional = pyproject["project"].get("optional-dependencies", {})
    aggregates = {"all", "all-cu12", "all-cu13", "gpu"}
    excused = {
        "dev": "the test suite, not a runtime capability",
        "publish": "maintainer-only mirror publishing, needs credentials "
                   "on a dataset repository no user has",
        "geog": "the PARITY FALLBACK for the high-resolution warp, not "
                "the warp. The default engine is the Rust static-fields "
                "library a wheel install stages, so [all] already carries "
                "high-resolution terrain; adding [geog] would add 118.8 "
                "MiB of GDAL stack that changes no default and unlocks no "
                "product. What stops this becoming 2.3.2 again is not the "
                "aggregate but tests/test_static_highres_warp_routing.py, "
                "which makes rasterio/pyproj/affine unimportable and "
                "requires the default terrain build to still answer.",
        "hydro-conus": "the LGPL-2.1+ 7z decoder for the CONUS NHDPlus hydro "
                       "source archives; hydro is opt-in and the worldwide "
                       "fabric covers CONUS without it, and [all] stays on "
                       "permissive licences (licence ruling 2026-10-14).",
        "mcp": "the arwen-mcp server's protocol stack, an agent "
               "integration surface rather than a forecast capability; "
               "a person following the quickstart runs forecasts from "
               "their shell and never speaks MCP, so [all] carrying it "
               "would be dead weight in the one-liner. The agent that "
               "does need it is told the extra by name in the guarded "
               "import's remedy sentence and in docs/mcp-server.md.",
    }
    for major, other in (("cu12", "cu13"), ("cu13", "cu12")):
        covered = declaration["extras"][f"all-{major}"]
        for name in optional:
            if name in aggregates or name in excused:
                continue
            if name == f"gpu-{other}":
                continue
            wanted = declaration["extras"][name]
            assert wanted <= covered, (
                f"[all-{major}] does not cover [{name}]: missing "
                f"{sorted(wanted - covered)}. Either add woof[{name}] to "
                f"[all-{major}] or record why it is excused.")
    assert declaration["extras"]["all"] == declaration["extras"]["all-cu12"]


# --------------------------------------------------------------------------
# The built artifact must agree with the source of truth
# --------------------------------------------------------------------------
def _installed_metadata_is_this_tree() -> bool:
    """False when the INSTALLED woof distribution's recorded origin is
    a different checkout -- the borrowed-install case the provenance
    banner names.  The metadata then describes the OTHER tree, and
    comparing it against this tree's pyproject measures the wrong
    artifact (a dev box holds one editable install and many worktrees;
    exactly one can be current).  The probe reads the distribution's
    own recorded origin (direct_url.json for an editable install),
    NOT the import resolution -- pytest runs put this tree first on
    sys.path, so the import always resolves here even when the
    metadata belongs elsewhere.  Where the installed woof IS this
    tree, the gate runs at full strength -- nothing loosened, scope
    corrected the way the ULP gate scopes out other checkouts."""
    import json
    from importlib import metadata as _md
    from pathlib import Path
    here = Path(__file__).resolve().parents[1]
    try:
        dist = _md.distribution("recast-woof")
    except _md.PackageNotFoundError:
        return False
    try:
        raw = dist.read_text("direct_url.json")
        if raw:
            url = json.loads(raw).get("url", "")
            if url.startswith("file://"):
                from urllib.request import url2pathname
                origin = Path(url2pathname(url[len("file://"):])).resolve()
                return origin == here
    except (OSError, ValueError, KeyError):
        pass
    # No recorded origin (a real wheel install): the metadata governs
    # whatever is installed; run the gate.
    return True


def test_installed_metadata_agrees_with_pyproject(pyproject, declaration):
    """A stale install cannot report green.

    This is the direction that matters when the suite is pointed at a real
    installation: the METADATA a user's ``pip`` reads is the artifact, and
    everything above it in this file is only a claim about the source.

    It skips for a ``.egg-info`` sitting inside this source tree.  That is a
    build by-product of an editable install, it goes stale by design the
    moment ``pyproject.toml`` is edited, and the ``pyproject.toml`` beside it
    is the authority anyway.  A real site-packages install gets no such
    excuse.
    """
    # No function-local `import pytest` here: the module already imports
    # it, and a local import would make `pytest` function-local
    # everywhere in this body -- on the branch where the gate RUNS, the
    # first later `pytest` reference then dies UnboundLocalError.  Found
    # at the first execution on a box where the installed woof IS the
    # tree under test (the borrowed-install skip had hidden the arm).
    if not _installed_metadata_is_this_tree():
        pytest.skip(
            "the installed woof resolves outside this checkout "
            "(borrowed install); its metadata describes that tree, "
            "not this one -- the gate runs where this tree is the "
            "installed one")

    from importlib.metadata import PackageNotFoundError, distribution

    try:
        dist = distribution(_SELF)
    except PackageNotFoundError:
        pytest.skip("woof is not installed in this environment")
    location = getattr(dist, "_path", None)
    if location is not None:
        try:
            Path(location).resolve().relative_to(REPO_ROOT)
        except ValueError:
            pass
        else:
            pytest.skip(
                f"the resolved woof metadata is {Path(location).name} inside "
                "this source tree, a by-product of an editable install rather "
                "than an installed artifact; pyproject.toml is the authority "
                "here. Run this test against a wheel in a clean venv for the "
                "check that counts.")
    if dist.version != pyproject["project"]["version"]:
        pytest.skip(f"installed woof is {dist.version}, the tree is "
                    f"{pyproject['project']['version']}")

    metadata_names = set()
    for requirement in dist.metadata.get_all("Requires-Dist") or []:
        name = _requirement_distribution(requirement.split(";")[0])
        if name and name != _canonical(_SELF):
            metadata_names.add(name)
    assert metadata_names == declaration["all"], (
        "the installed distribution's Requires-Dist disagrees with "
        "pyproject.toml. Only in the install: "
        f"{sorted(metadata_names - declaration['all'])}; only in the tree: "
        f"{sorted(declaration['all'] - metadata_names)}. Rebuild and "
        "reinstall before trusting any other result in this file.")

    declared_extras = set(pyproject["project"]
                          .get("optional-dependencies", {}))
    provided = set(dist.metadata.get_all("Provides-Extra") or [])
    assert declared_extras <= provided, (
        "pyproject.toml declares extras the built metadata does not offer: "
        f"{sorted(declared_extras - provided)}. An empty extra that "
        "setuptools drops is a broken install line, not a no-op alias.")


# --------------------------------------------------------------------------
# Cross-imports: a wheel has no sys.path pointing at a checkout
# --------------------------------------------------------------------------
def test_no_shipped_module_imports_a_sibling_by_bare_name(pyproject):
    """``import da_nowcast`` must not be the ONLY way a module finds a sibling.

    Inside the wheel these are ``tools.da_nowcast`` and friends.  A bare
    top-level import of one resolves when the module is run as a path from a
    checkout; whether it resolves from an installed wheel depends entirely on
    whether something put the sibling's directory on ``sys.path`` first.

    Measured, on a wheel installed into a clean venv with no checkout
    anywhere on the path: every one of the recorded sites below DOES resolve,
    because ``tools`` ships as a flat top-level package and every hack points
    at ``<site-packages>/tools``, which exists.  So this is not a live
    breakage on this base -- the 2.3.2 audit filed it as latent and the
    measurement agrees.

    It is still worth a gate, for the reason the hacks are fragile rather
    than wrong: each one puts 142 top-level module names at ``sys.path[0]``,
    ahead of the standard library and every installed distribution, for the
    lifetime of the process in the two cases that never pop it.  Nothing
    under ``tools/`` collides today (checked: no stdlib name, no common
    distribution name), and the day one does, the failure is a shadowed
    import in an unrelated library.

    So the two permitted shapes are: a package-qualified import, with the
    bare name only as a ``python tools/x.py`` fallback; or a recorded entry
    in :data:`_SYS_PATH_SIBLING_SITES` saying which file does it and why
    nobody has package-qualified it yet.
    """

    shipped = _shipped_sources(pyproject)
    module_names = {p.stem for p in shipped}
    stdlib = set(sys.stdlib_module_names)
    offenders: list[str] = []
    seen_recorded: set[str] = set()
    for path in shipped:
        rel = path.relative_to(REPO_ROOT).as_posix()
        package = path.parent.relative_to(REPO_ROOT).as_posix().replace(
            "/", ".")
        source = path.read_text(encoding="utf-8")
        qualified = set(re.findall(
            r"(?:from|import)\s+(?:tools|woof)[\w.]*\.(\w+)", source))
        qualified |= {name.strip().split(" as ")[0].strip()
                      for line in re.findall(
                          r"from\s+(?:tools|woof)[\w.]*\s+import\s+([^\n(]+)",
                          source)
                      for name in line.split(",")}
        for name, lineno in _top_level_imports(path):
            if name in stdlib or name in _FIRST_PARTY or name in _PROVIDERS:
                continue
            if name not in module_names or name == path.stem:
                continue
            if name in qualified:
                continue                     # guarded package-qualified form
            if rel in _SYS_PATH_SIBLING_SITES:
                assert "sys.path.insert" in source, (
                    f"{rel} is recorded as reaching {name!r} through a "
                    "sys.path insert, and there is no sys.path.insert in it")
                seen_recorded.add(rel)
                continue
            offenders.append(
                f"  {rel}:{lineno} imports the sibling {name!r} by bare "
                f"name; inside the wheel it is {package}.{name}")
    assert not offenders, (
        "a shipped module reaches a sibling by bare name with nothing to "
        "make that resolve from an installed wheel. Import it as "
        "`from tools.<name> import ...` (keeping the bare form as an "
        "except-ImportError fallback if the file is also run as a path), or "
        "record it in _SYS_PATH_SIBLING_SITES with the reason:\n"
        + "\n".join(offenders))
    stale = sorted(set(_SYS_PATH_SIBLING_SITES) - seen_recorded)
    assert not stale, (
        "_SYS_PATH_SIBLING_SITES names files that no longer reach a sibling "
        f"by bare name. Delete the rows: {stale}")


def test_the_shipped_surface_matches_the_packaging_declaration(pyproject):
    """If the include list changes shape, the scanner above must be revisited."""

    include = pyproject["tool"]["setuptools"]["packages"]["find"]["include"]
    # `docs` joined the include list in commit 9223560b4, and this
    # DECLARATION was the stale side: the anchor is deliberate and the
    # scanner already understands its shape. It ships exactly one file,
    # docs/mpas-seam.md, named by its own package-data key -- an external
    # consumer hashes sixteen engine files by REPOSITORY path against both
    # a checkout and site-packages, and this was the one key no
    # distribution placed anywhere. It is anchored (`docs`, never `docs*`)
    # precisely so the receipt trees under docs/superpowers, which do carry
    # .py, stay out of the wheel; a bare name means `docs/*.py` only, and
    # the assertion below holds that at none.
    assert include == ["woof", "woof.*", "tools", "tilestream*",
                       "configs*", "docs", "mpas_cycle_bridge*"], (
        f"packages.find include is {include}; _shipped_sources understands "
        "a trailing '.*' as 'this package's subpackages', a trailing '*' as "
        "'this package tree' and a bare name as 'this package's top level "
        "only'. Teach it the new shape before changing this assertion.")
    shipped = _shipped_sources(pyproject)
    # `docs` is the second DATA entry, on the same terms as `configs*`
    # below: the scanner walks it for `.py` and must keep finding none. A
    # Python file dropped at docs/ top level would join the shipped import
    # surface, and -- because the anchor exists to keep docs/superpowers
    # out -- would do it while looking like documentation.
    assert not any("/docs/" in p.as_posix() for p in shipped), (
        "docs/ now carries top-level Python, so it is part of the shipped "
        "import surface; it was anchored to ship one markdown file")
    # `configs*` was added for the repository configs two headline documents
    # name, and it is a DATA entry: the scanner above walks it for `.py` the
    # same way it walks the rest, and must keep finding none. A Python file
    # appearing under configs/ would join the shipped import surface without
    # anybody deciding it should.
    assert not any("/configs/" in p.as_posix() for p in shipped), (
        "configs/ now carries Python, so it is part of the shipped import "
        "surface; it was declared as a data directory only")
    assert any(p.as_posix().endswith("woof/cli.py") for p in shipped)
    assert any(p.as_posix().endswith("tools/da_nowcast.py") for p in shipped)
    assert any(p.as_posix().endswith("tilestream/harness.py") for p in shipped)
    assert not any("ftz_receipt" in p.as_posix() for p in shipped), (
        "tools/ subpackages have no __init__ route into the wheel; the "
        "scanner must not pretend they ship")
    # This generator reads tests/data and imports sealed test helpers. It
    # belongs with the checkout-only battery, outside the tools top level.
    assert (REPO_ROOT / "tools/battery/diffusion_cache_source_receipt.py").is_file()
    assert not any(p.name == "diffusion_cache_source_receipt.py" for p in shipped)
