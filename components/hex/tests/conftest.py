"""Tier gates for the woof hex test battery.

Three things in this tree cannot run everywhere, and each one used to fail in
a way that told the reader nothing:

* **CUDA tests** compiled and ran kernels on whatever card was present, even
  on a box whose owner had asked that no GPU work happen there.
* **The x4.163842 full-physics tier** needs the re-derived free-memory
  floor (about 20.5 GiB: the measured x4 peak plus headroom).  On a
  smaller card it does not print "this card is too small"; it dies inside a
  CuPy allocation with an out-of-memory traceback several frames below
  anything a reader recognises.
* **The byte-pinned authority tier** (mesh, static, init, native history)
  needs about 6.9 GiB of files that ship with no fetch path and live on
  the reference node.  Without them four tests raise ``FileNotFoundError`` naming
  a path inside this checkout that has never existed on this machine, which
  reads as a broken checkout rather than a missing asset.

So each tier gets a gate that names the concrete breakage, and each gate
fails closed.  The marker discipline is woof's, deliberately: a convention
that depends on every author remembering a decorator fails open, and woof
measured what that costs -- 34 CUDA gates once compiled and ran on a machine
that had been declared off limits, because an unmarked test is not excluded
by ``-m "not gpu"``.  Detection here is by AST, and an explicit marker is
additive rather than the only route in.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
import re
from typing import Iterable

import pytest

# Kernel numerics for the battery (owner ruling 2026-10-16: fast is the door default, strict is the opt-in).
# The battery's compile-option, oracle and replay tests pin the strict route, the one that is bitwise to
# native MPAS, so the battery names strict, and so does every subprocess it starts.  Tests of the fast
# default select it themselves (test_numerics.py).
os.environ["WOOF_HEX_NUMERICS"] = "strict"
os.environ["GPUWM_NUMERICS"] = "strict"


ROOT = Path(__file__).resolve().parents[1]

import sys

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

# The x4 runner in the layout this tree runs in (tests/_layout.py).
from _layout import RUNNER_PATH  # noqa: E402

from woof.hex.device_admission import native_device_floor_bytes

#: Free device memory the x4.163842 full-physics tier requires: the same
#: measured floor the driver admits on (the merged-tip x4 peak plus the
#: shared headroom), read from the one admission
#: surface so this gate cannot drift from the gate the run itself applies.
X4_FULL_PHYSICS_BYTES = native_device_floor_bytes()

#: Honoured under both spellings.  A box configured for the engine sets
#: GPUWM_NO_LOCAL_GPU; the port must not quietly ignore that and light up the
#: same card from a different distribution.
_GPU_BAN_VARIABLES = ("WOOF_HEX_NO_LOCAL_GPU", "GPUWM_NO_LOCAL_GPU")

NO_LOCAL_GPU = any(
    os.environ.get(name, "") not in ("", "0") for name in _GPU_BAN_VARIABLES
)

if NO_LOCAL_GPU:
    # Planted where every route converges.  The marker automation cannot see
    # transitive device contact -- a helper three imports deep that allocates
    # -- and the CUDA runtime can.  An import-level ban was considered and
    # rejected: importing cupy is not the crime, opening the device is.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# The host-preparation cache (woof.hex.prep_cache) is off for the battery unless
# a test turns it on for itself.  THE BREAKAGE THIS PREVENTS: a test that
# monkeypatches a helper inside a cached function to provoke a refusal would
# be served the earlier test's stored result for the same mesh, and the
# battery would also write cache entries into the runner's home directory.
# tests/test_prep_cache.py enables it explicitly against a temporary root.
os.environ.setdefault("WOOF_HEX_PREP_CACHE", "off")
# Kernel caching has its own switch. Keep test compiles out of the user's
# cache as well; cache tests opt in with an isolated directory.
os.environ.setdefault("WOOF_HEX_KERNEL_PREWARM", "off")

# The MPAS route's first-use physics-resources build is off for the battery.
# THE BREAKAGE THIS PREVENTS: a test that registers hex-cuda slots without a
# packet would fetch upstream tables and run gfortran for minutes, writing into
# the runner's home cache.  Tests that exercise first use replace
# woof.hex.mpasphys.resources_build.ensure_packet themselves.
os.environ.setdefault("WOOF_HEX_PHYSICS_RESOURCES_FIRST_USE", "off")

# A copied checkout can carry another host's queue settings. Tests that
# exercise site configuration supply their own file explicitly.
os.environ.setdefault("HEX_SITE_FILE", os.devnull)


@pytest.fixture
def cpu_physics_phase_release(monkeypatch):
    """Keep CPU transaction fakes independent of CuPy device allocation.

    The phase allocator has its own stand-in-device tests. These callers
    exercise the physics transaction with NumPy arrays and no device.
    """
    from contextlib import nullcontext
    from woof.hex import device_memory_release

    monkeypatch.setattr(device_memory_release, "phase_release", lambda name: nullcontext())


def pytest_configure(config: pytest.Config) -> None:
    for line in (
        "gpu: needs a CUDA device and cupy; auto-applied to any test whose "
        "module or function imports cupy",
        "bigcard: needs the x4.163842 full-physics free-memory floor "
        f"({X4_FULL_PHYSICS_BYTES / 1024**3:.1f} GiB, the re-derived "
        "measured requirement)",
        "assets: needs the byte-pinned mesh/static/init/native-authority "
        "files, which ship with no fetch path",
        "slow: minutes rather than seconds",
    ):
        config.addinivalue_line("markers", line)


# ---------------------------------------------------------------------------
# tier 1: the GPU marker, detected rather than declared
# ---------------------------------------------------------------------------
def _cupy_scope(source: str) -> tuple[bool, frozenset[str]]:
    """Return ``(whole_module, {function names})`` that reach cupy.

    Per-function rather than per-file because one file in woof carried a
    single cupy import at line 1548 while its other two hundred tests
    deliberately stubbed cupy out; marking the whole module would have
    excluded them all from every CPU run.
    """

    try:
        tree = ast.parse(source)
    except SyntaxError:  # pragma: no cover - a broken test file fails louder
        return False, frozenset()

    def _touches(node: ast.AST) -> bool:
        for child in ast.walk(node):
            if isinstance(child, ast.Import):
                if any(alias.name.split(".")[0] == "cupy" for alias in child.names):
                    return True
            elif isinstance(child, ast.ImportFrom):
                if (child.module or "").split(".")[0] == "cupy":
                    return True
            elif isinstance(child, ast.Call):
                function = child.func
                name = getattr(function, "attr", None) or getattr(function, "id", None)
                if name == "importorskip" and child.args:
                    first = child.args[0]
                    if isinstance(first, ast.Constant) and first.value == "cupy":
                        return True
        return False

    module_level: list[ast.AST] = [
        node
        for node in tree.body
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    whole_module = any(_touches(node) for node in module_level)

    functions: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _touches(node):
            functions.add(node.name)
    return whole_module, frozenset(functions)


# ---------------------------------------------------------------------------
# tier 4: the measurement receipts under evidence/
#
# A shipped number in this tree is checked against the receipt that produced
# it -- the ledger a footprint row was fitted from, the contract run an
# admission anchor was earned by.  Those receipts live under `evidence/`, and
# `evidence/` is deliberately HELD OUT of every published surface: it carries
# home-directory paths and node labels.  The assembled public tree drops it,
# and the sdist packs exactly one member of it.
#
# THE BREAKAGE THIS GATE PREVENTS, measured 2026-08-27 on two independent
# surfaces: 30 of these tests failed on an assembled public tree
# (evidence/assembly-rehearsal-20260827/) and about 28 in a fresh-venv run of
# the sdist a user unpacks (evidence/xmachine-20260827/).  `ci.yml` runs on
# push, so the first public act would have gone red on the release commit --
# every one of them raising FileNotFoundError on a path inside the checkout
# that never existed on the reader's machine, which reads as a broken
# distribution rather than a record that does not travel.
#
# THE DISCRIMINATOR IS THE DISTRIBUTION, NOT THE FILE, and that is the whole
# design.  "This tree does not carry its receipts" is a skip.  "This tree
# carries its receipts and the one this row names is not among them" is a
# FAILURE, and it is exactly the defect the anchor gates were written to
# catch: a row citing a measurement nobody can check.  A per-file existence
# test would collapse the two into one answer and retire a real gate while
# looking like it fixed a packaging problem.
# ---------------------------------------------------------------------------
RECEIPTS = ROOT / "evidence"

#: The one member of `evidence/` that travels.  MANIFEST.in packs it by name
#: because README.md and docs/source-matrix.md link it: it is what tells a
#: reader that an `evidence/...` reference names the IDENTITY of a
#: measurement rather than a file they are missing.  Its presence therefore
#: says nothing about whether the receipts themselves are here, which is why
#: it is excluded from the question below rather than answering it.
RECEIPT_POINTER = "EVIDENCE.md"


def receipts_carried() -> bool:
    """Does THIS tree carry the measurement receipts at all?"""

    if not RECEIPTS.is_dir():
        return False
    return any(entry.name != RECEIPT_POINTER for entry in RECEIPTS.iterdir())


RECEIPTS_ABSENT_REASON = (
    "this tree does not carry the measurement receipts under evidence/.  "
    "They are held out of every published surface on purpose -- the receipts "
    "carry home-directory paths and node labels -- so the assembled public "
    "tree drops the directory and the sdist packs only "
    f"evidence/{RECEIPT_POINTER}, the pointer that explains the convention.  "
    "This test reads a receipt to check a shipped number against the "
    "measurement that produced it, and with the receipt absent there is "
    "nothing to check.  A HELD-OUT RECORD, NOT A BROKEN CHECKOUT: in a tree "
    "that does carry its receipts this same test fails rather than skips "
    "when the one it names is missing"
)


@pytest.fixture
def receipts() -> Path:
    """The receipt tree, or a skip that says why it is not here.

    Requesting this fixture is how a test declares "I read a receipt".  It is
    a fixture rather than a marker so that the gate travels with the test
    function that needs it and cannot be applied to a whole module by
    accident -- three of the six files involved have members that must keep
    running on a published tree, including the negative controls that assert
    the anchor gates refuse a receipt path which does not exist.
    """

    if not receipts_carried():
        pytest.skip(RECEIPTS_ABSENT_REASON)
    return RECEIPTS


# ---------------------------------------------------------------------------
# tier 2: the byte-pinned authority assets
# ---------------------------------------------------------------------------
_AUTHORITY_LITERAL = re.compile(
    r"^AUTHORITY_PINS[^=]*=\s*(?P<body>\{.*?^\})", re.DOTALL | re.MULTILINE
)


def _authority_relative_paths() -> tuple[str, ...]:
    """The pinned authority paths, read without importing the runner.

    The runner is 163 KB and pulls numpy and netCDF4 at import; paying that
    on every collection to learn nine filenames is not a trade worth making,
    and a conftest that imports the module under test is its own hazard.
    """

    try:
        source = RUNNER_PATH.read_text(encoding="utf-8")
    except OSError:  # pragma: no cover - only outside a checkout
        return ()
    match = _AUTHORITY_LITERAL.search(source)
    if match is None:  # pragma: no cover - the literal moved; fail open loudly
        return ()
    try:
        pins = ast.literal_eval(match.group("body"))
    except (ValueError, SyntaxError):  # pragma: no cover
        return ()
    return tuple(
        str(pin["relative_path"])
        for pin in pins.values()
        if isinstance(pin, dict) and "relative_path" in pin
    )


def _missing_authorities() -> tuple[str, ...]:
    return tuple(
        relative
        for relative in _authority_relative_paths()
        if not (ROOT / relative).is_file()
    )


def _authority_touching_functions(source: str) -> frozenset[str]:
    """Test functions that open a byte-pinned authority file.

    Detected on ``default_authority_paths``, the one accessor that turns a
    pin into a path on disk.  Reading ``AUTHORITY_PINS`` itself is not
    detected and must not be: several tests assert on the pin metadata --
    byte counts, digest shapes -- and those are exactly the tests that should
    keep running on a box with no assets, because they are what proves the
    pins are well formed before anyone spends 6.9 GiB fetching files.
    """

    try:
        tree = ast.parse(source)
    except SyntaxError:  # pragma: no cover
        return frozenset()
    functions: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Attribute)
                and child.attr == "default_authority_paths"
            ) or (isinstance(child, ast.Name) and child.id == "default_authority_paths"):
                functions.add(node.name)
                break
    return frozenset(functions)


# ---------------------------------------------------------------------------
# tier 3: device capacity
# ---------------------------------------------------------------------------
def free_device_bytes() -> int | None:
    """Free memory on device 0, or ``None`` when there is no usable device.

    Answers ``None`` under the ban WITHOUT importing cupy, because importing
    it and asking for a device count is itself device contact and used to
    happen on every pytest invocation on this box.
    """

    if NO_LOCAL_GPU:
        return None
    try:
        import cupy
    except Exception:
        return None
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            return None
        free, _total = cupy.cuda.runtime.memGetInfo()
        return int(free)
    except Exception:  # pragma: no cover - driver present but unusable
        return None


def _bigcard_reason() -> str | None:
    if NO_LOCAL_GPU:
        return (
            "GPU work is banned on this box by "
            f"{'/'.join(_GPU_BAN_VARIABLES)}; the x4.163842 full-physics tier "
            "belongs on the dedicated device"
        )
    free = free_device_bytes()
    if free is None:
        return (
            "no CUDA device with a working cupy is reachable, and the "
            "x4.163842 full-physics tier needs "
            f"{X4_FULL_PHYSICS_BYTES / 1024**3:.1f} GiB of free device "
            "memory (the re-derived measured floor) -- there is nothing "
            "here to hold it"
        )
    if free < X4_FULL_PHYSICS_BYTES:
        return (
            "this card has "
            f"{free / 1024**3:.1f} GiB free and the x4.163842 full-physics "
            "tier needs "
            f"{X4_FULL_PHYSICS_BYTES / 1024**3:.1f} GiB (the measured "
            "floor: the x4 peak plus shared headroom).  It "
            "would not fail a check, it would die inside a CuPy allocation "
            "part-way through a run.  In practice that is a 32 GiB device"
        )
    return None


def pytest_collection_modifyitems(
    config: pytest.Config, items: Iterable[pytest.Item]
) -> None:
    missing = _missing_authorities()
    assets_skip = pytest.mark.skip(
        reason=(
            "byte-pinned authority assets are absent from this checkout "
            f"({len(missing)} of {len(_authority_relative_paths())} missing, "
            f"first: {missing[0] if missing else '-'}).  They are about "
            "6.9 GiB of mesh, static, init and native-history files that "
            "woof hex ships with no fetch path; they live on the dedicated "
            "node.  This is a missing asset, not a broken checkout"
        )
    )
    gpu_ban_skip = pytest.mark.skip(
        reason=(
            f"{'/'.join(_GPU_BAN_VARIABLES)} is set: GPU work belongs on the "
            "dedicated device"
        )
    )
    bigcard_reason = _bigcard_reason()
    bigcard_skip = (
        pytest.mark.skip(reason=bigcard_reason) if bigcard_reason else None
    )

    # Parametrized tests share source. Parse each module once per collection,
    # rather than once per item in every xdist worker. Keep this cache local:
    # another pytest.main() call may collect edited sources in the same process.
    scopes: dict[Path, tuple[bool, frozenset[str], frozenset[str]]] = {}
    for item in items:
        path = Path(str(getattr(item, "fspath", "")))
        if path not in scopes:
            try:
                source = path.read_text(encoding="utf-8")
            except OSError:  # pragma: no cover
                source = ""
            whole_module, cupy_functions = _cupy_scope(source)
            scopes[path] = (whole_module, cupy_functions,
                            _authority_touching_functions(source))
        whole_module, cupy_functions, authority_functions = scopes[path]
        if whole_module or item.name.split("[")[0] in cupy_functions:
            item.add_marker(pytest.mark.gpu)

        if item.name.split("[")[0] in authority_functions:
            item.add_marker(pytest.mark.assets)

        if NO_LOCAL_GPU and item.get_closest_marker("gpu") is not None:
            item.add_marker(gpu_ban_skip)
        if missing and item.get_closest_marker("assets") is not None:
            item.add_marker(assets_skip)
        if bigcard_skip is not None and item.get_closest_marker("bigcard") is not None:
            item.add_marker(bigcard_skip)
        not_carried = _not_carried_folded(item)
        if not_carried is not None:
            item.add_marker(pytest.mark.skip(reason=not_carried))


# (WOOF) Checks of gpuwm-hex's own distribution and repository, which a tree that folds this package in under
# another import name does not carry: gpuwm-hex's pyproject.toml, setup.py, MANIFEST.in, NOTICE,
# THIRD-PARTY-NOTICES.txt and engine-patches/, its wheel layout, its Rust crates (rust/) and vendored METIS
# sources (third_party/), and its release tooling on its own repository layout.  The folding tree declares and
# checks its own distribution, and ships the helpers those crates build prebuilt.  Each row names what is not
# carried; nothing here skips in this repository's own layout.
_NOT_CARRIED_FILES = {
    "test_packaging_declaration.py": "gpuwm-hex's pyproject.toml, MANIFEST.in and sdist",
    "test_wheel_platform_check.py": "gpuwm-hex's wheel layout and publish workflow",
    "test_licence_notices.py": "gpuwm-hex's NOTICE, THIRD-PARTY-NOTICES.txt and licenses/ "
                               "(the folding tree's NOTICE carries this NOTICE under its own heading)",
    "test_s2s_renderer_packaging.py": "gpuwm-hex's setup.py",
    "test_gpmetis_bundled.py": "gpuwm-hex's vendored METIS sources (third_party/); hex_gpmetis ships prebuilt",
    "test_repin_source_tables.py": "gpuwm-hex's repository layout, where its repin tool runs",
}
_NOT_CARRIED_TESTS = {
    "test_engine_bridges_bundled.py::test_source_prebuilts_match_this_tree_so_publish_can_build":
        "gpuwm-hex's engine-patches/",
    "test_netcdf_copy_packaged.py::test_the_sdist_carries_the_copier_crate_and_not_its_build_output":
        "gpuwm-hex's MANIFEST.in and rust/netcdf-copy",
    "test_netcdf_copy_packaged.py::test_the_wheel_build_installs_the_copier_into_libexec": "gpuwm-hex's setup.py",
    "test_obs_scorer_optional.py::test_setup_builds_the_scorer_only_from_a_source_checkout": "gpuwm-hex's setup.py",
    "test_static_coordinate_precision.py::test_the_wheel_bundles_a_static_builder_that_holds_binary64":
        "gpuwm-hex's setup.py",
    "test_physics_resources_build.py::test_the_crate_builds_every_decoder_the_builder_binds":
        "gpuwm-hex's rust/physics-tables crate",
    "test_physics_route_resources.py::test_the_pool_reader_is_gpuwm_24b6f08_byte_for_byte":
        "gpuwm-hex's rust/physics-pool crate",
    "test_physics_route_resources.py::test_the_pool_reader_is_gpuwm_24b6f08_plus_the_field_parallel_transport":
        "gpuwm-hex's rust/physics-pool crate",
    "test_physics_route_resources.py::test_the_pool_transport_takes_the_cpus_it_may_use":
        "gpuwm-hex's rust/physics-pool crate",
    "test_libm_batch_packaged.py::test_setup_names_the_bridge_by_the_runtime_source_key": "gpuwm-hex's setup.py",
}
#: a fixture that loads gpuwm-hex's setup.py
_NOT_CARRIED_FIXTURES = {"hexsetup": "gpuwm-hex's setup.py"}


def _not_carried_folded(item: pytest.Item) -> str | None:
    from _layout import FOLDED, NOT_CARRIED_FOLDED

    if not FOLDED:
        return None
    module = Path(str(getattr(item, "fspath", ""))).name
    what = (_NOT_CARRIED_FILES.get(module)
            or _NOT_CARRIED_TESTS.get(f"{module}::{item.name.split('[')[0]}")
            or next((why for name, why in _NOT_CARRIED_FIXTURES.items()
                     if name in getattr(item, "fixturenames", ())), None))
    return None if what is None else f"{NOT_CARRIED_FOLDED}: {what}"
