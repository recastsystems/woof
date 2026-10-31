"""Every compiled site reports the translation unit it actually compiled.

The manifest is only worth carrying if it cannot drift from the compiler call
beside it.  These tests read the tree's own syntax: at each site the arguments
handed to :func:`record_module` must be the *same expressions* handed to the
compiler, so recording a different source, or collapsing the option tuples
into one global flags string, is a syntactic difference a test can see
without a device.

TWO compiler routes are audited, not one.  ``cp.RawModule`` is the common
one; ``cupy.cuda.compiler.compile_using_nvrtc`` is the bypass a site must
take when it needs an option CuPy would otherwise duplicate -- CuPy appends
``-ftz=true`` after the caller's options, NVRTC 13.0 rejects the repeat
outright, and ``woof/core/rrtmg_sw.py`` moved onto the bypass for exactly
that reason.  Auditing only RawModule would have let that site keep
recording an option tuple no compiler was handed.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

from conftest import requires_gpu

from woof.certify.kernel_manifest import (KernelManifestConflict,
                                           kernel_manifest, record_module,
                                           reset_kernel_manifest,
                                           source_sha256)
from woof.core.constants import CUDA_DEFINES

REPO = Path(__file__).resolve().parents[1]
KERNEL_DIR = REPO / "woof" / "core" / "kernels"

#: The manifest sites: every module that constructs a ``cp.RawModule`` for
#: a forecast translation unit or records one.  The Noah-MP factories no
#: longer construct anything themselves -- they delegate to the one site in
#: ``noahmp_kernel_sources.py`` -- and stay listed so :func:`audit_source`
#: keeps reading them: a factory that grew a RawModule of its own again
#: would be caught here.  :func:`test_every_rawmodule_constructor_in_the_tree_is_named`
#: is the census behind this list; nothing under ``gpuwm/`` may construct a
#: RawModule without appearing either here or in its explained allow-list.
SITE_FILES = (
    "woof/core/spp_kernel_sources.py",
    "woof/core/kernels/__init__.py",
    "woof/core/nest_interp.py",
    "woof/core/noahmp_driver_gpu.py",
    "woof/core/noahmp_energy_gpu.py",
    "woof/core/noahmp_glacier_gpu.py",
    "woof/core/noahmp_kernel_sources.py",
    "woof/core/noahmp_slab_libm.py",
    "woof/core/noahmp_thermal_gpu.py",
    "woof/core/noahmp_vegeflux_gpu.py",
    "woof/core/rrtmg_sw.py",
    "woof/core/ruc_spp.py",
    # Kain-Fritsch's unflushed, uncontracted direct-NVRTC unit (2026-10-07,
    # the WRF 4.6.1 port graded by tools/kf_wrf461_oracle).
    "woof/core/kf.py",
    # The fused forecast units the 2026-09-30 speed lanes added: New
    # Tiedtke's fused column (b11fc66ab), Noah's forcing prologue
    # (3e23476ce) and SFCDIAGS (9aea8904f), the cumulus clock (da604fe01,
    # two units) and the tendency coupling (4c1cab3c0).
    "woof/core/ntiedtke_fused.py",
    "woof/core/noah_forcing.py",
    "woof/core/noah_sfcdiags.py",
    "woof/core/cumulus_clock.py",
    "woof/core/tendency_coupling.py",
    "woof/ensemble/batch_fluxes.py",
    "woof/ensemble/batch_kernel.py",
    "woof/ensemble/batch_perturbation.py",
    "woof/ensemble/batch_physics.py",
    "woof/ensemble/batch_physics_init.py",
    "woof/ensemble/batch_product_output.py",
    "woof/ensemble/batch_products.py",
    # 3fa941d96, lane/ensemble-nested-pack delivered at ff238b1d9, adds
    # the native member RUC unit. Audit its actual source/options recording.
    "woof/ensemble/batch_ruc.py",
    "woof/ensemble/surface_recipe.py",
    "woof/da/letkf_device.py",
    # The land no-FTZ route (WRF 4.7 Noah-MP night snow-aging port): the one
    # Noah-MP compile site moved from a RawModule in noahmp_kernel_sources.py
    # to direct NVRTC here, recording source, options and image.
    "woof/core/ieee_module.py",
)

#: Two cached loaders, nest interpolation, and the one Noah-MP compile
#: site (``noahmp_kernel_sources.compile_runtime_unit``); the six Noah-MP
#: factories that used to be sites each delegate to it.  ``rrtmg_sw.py`` is
#: NOT among them -- it compiles through direct NVRTC instead, see the
#: module docstring -- but it stays in
#: :data:`SITE_FILES` because it still records.  Plus the six fused units
#: of the speed lanes' five files (the cumulus clock compiles two), and the
#: CLM lake's own ``--fmad=false`` site beside the two cached loaders
#: (``_load_module_without_fmad``, 2.8.5), which records in its own function.
# The surface-state recipe records its seeded preparation unit at the
# compiler call too, so the same source/options audit covers that site.
# 3fa941d96, lane/ensemble-nested-pack: the one recorded member RUC
# constructor joins the 28 retained sites; no compiler source is changed.
# lane/mynn-exact: the integer-define tier of a no-FMA unit
# (``_load_module_int_defines_without_fmad``, the gsd_41 MYNN build) records
# in its own function beside its compile, as the lake's site does.
# The recorded LETKF member-count loader joins the 30 retained sites.
# Its original source expression and compiler options remain unchanged.
# 2.9.0 (r290/ens-showcase 4acb14f7f): the surface-state recipe leaves
# RawModule for a direct compile_using_nvrtc site, because RawModule appends
# -ftz=true and NVRTC refused it beside the unit's --ftz=false ("--ftz
# defined more than once"); it still records its source and options, so the
# count moves one site from here to EXPECTED_NVRTC_SITE_COUNT.
# golden/wrf-fixes (2.9.0 round 2): Noah-MP's RawModule site left for direct
# NVRTC (ieee_module.py, counted above) and the urban no-FTZ loader
# (kernels/__init__.py _load_urban_module_without_ftz) joined: 30 - 1 + 1.
EXPECTED_SITE_COUNT = 30

#: ``cp.RawModule`` constructors under ``gpuwm/`` that are NOT manifest
#: sites, each with the reason.  Closed and literal: a new constructor
#: anywhere else fails the census below until it is either made a site or
#: explained here.
RAWMODULE_CONSTRUCTORS_OUTSIDE_THE_MANIFEST = {
    "woof/ensemble/device_digest.py": (
        "an integer XXH64 tree checksum over an ensemble member bank's bytes "
        "for integrity receipts; it does no floating arithmetic, computes no "
        "forecast field and is not a translation unit the manifest freezes"),
    "woof/kernel_aot.py": (
        "the AOT loader opens a verified precompiled image using path alone; "
        "it never compiles source or supplies numerical options, and its "
        "path-only boundary is checked separately below"),
    "woof/core/tempo_runtime.py": (
        "the TEMPO column unit's own compile site; TEMPO_SOURCE_FINGERPRINT binds "
        "every included header's bytes into the compile key, and the unit is "
        "graded word for word in strict and default arithmetic by "
        "tests/test_tempo_adapter_oracle_gpu.py"),
    "woof/core/urban_bem.py": (
        "the BEP+BEM composed unit's own compile site (sf_urban_physics = 3, "
        "-fmad=false --ftz=false); it records through record_module in the "
        "same function and its frame is priced as urban_bem_composed in "
        "woof/core/preflight.py"),
    "woof/doctor.py": (
        "a one-kernel self-contained probe inside a subprocess source string, "
        "compiled to prove the toolchain works; not a forecast translation unit"),
    "woof/core/mynn_pbl_gpu.py": (
        "a one-kernel finiteness and range check over the MYNN inputs, built "
        "from a predicate string at call time; it reads the arrays and sets "
        "flags, computes no forecast field and is not a translation unit the "
        "manifest freezes"),
    "woof/core/ruc_gpu.py": (
        "a one-kernel finiteness check over the RUC land-surface inputs, built "
        "per argument count at call time in the shape of the MYNN check above; "
        "it reads the arrays and sets flags, computes no forecast field and is "
        "not a translation unit the manifest freezes"),
    "woof/core/ruc_tier.py": (
        "the fused RUC unit's own compile site (ruc.cu read as device "
        "functions plus the fused sfctmp and driver sources), one module per "
        "soil geometry; it records through record_module in the same function "
        "under its own key, as the BEP+BEM composed unit above does"),
    "woof/core/storm_tracking.py": (
        "the vortex tracker's one-kernel isobaric height, a transcription of "
        "the rw-isobaric crate's column read graded word for word against it "
        "by tests/test_storm_tracking_isobaric_gpu.py; it runs between steps "
        "to steer a nest, computes no forecast field and is not a "
        "translation unit the manifest freezes"),
    "woof/da/fixed_order_gemm.py": (
        "the LETKF analysis's fixed-order batched products, one RawKernel per "
        "float dtype compiled on first use; they run in the data-assimilation "
        "analysis, not a forecast step, record nothing and are not a "
        "translation unit the manifest freezes"),
    "woof/da/radar_tten.py": (
        "the research data-assimilation line's radar latent heating, one "
        "RawKernel per entry point compiled on first use: the builder runs "
        "between legs, and its one model-side kernel runs only while a "
        "forcing is attached by the cycle driver's opt-in flag; it records "
        "nothing and is graded against NOAA's compiled Fortran instead"),
    "woof/da/hydrometeor_analysis.py": (
        "the radar precipitation analysis of the research DA line (NOAA's "
        "Thompson retrieval and GSD precipitation block, glibc_flt64.cuh "
        "prepended), one RawModule compiled on first use; it runs in the "
        "data-assimilation analysis, not a forecast step, records nothing and "
        "is graded bit for bit against NOAA's own Fortran by "
        "tools/gsd_precip_oracle"),
    "woof/core/tempo_runtime.py": (
        "the TEMPO unit's own compile site (mp_physics = 900, "
        "--fmad=false), its headers composed into the one compiled string; "
        "it records through record_module in the same function under its "
        "own key, as the BEP+BEM composed unit above does"),
    "woof/da/letkf_device.py": (
        "the LETKF analysis's member-count kernels, one RawModule per "
        "ensemble size and card compiled on first use; they run in the "
        "data-assimilation analysis, not a forecast step, record nothing "
        "and are not a translation unit the manifest freezes"),
    "woof/core/noah_mosaic.py": (
        "the Noah mosaic tile loop's two compile sites (with and without the "
        "urban canopy, --fmad=false); each records through record_module in "
        "the same function under its own key, as the BEP+BEM composed unit above "
        "does"),
    # The compiled WRF v4.7.1 oracle harnesses (2.8.2, lane/282-oracle-*).
    "woof/verify/advect_oracle.py": (
        "the compiled WRF advection oracle's harness: it compiles control "
        "variants of advection.cu to compare output words with WRF Fortran "
        "under tests/test_advect_wrf471_parity.py, never in a forecast"),
    "woof/verify/smallstep_bookkeeping_oracle.py": (
        "the small-step bookkeeping oracle's harness: it compiles an "
        "instrumented acoustic source to compare words with compiled WRF "
        "under tests/test_smallstep_bookkeeping_wrf471_parity.py, never in a "
        "forecast"),
    "woof/verify/smallstep_horizontal_oracle.py": (
        "the small-step horizontal oracle's harness: it compiles a probe of "
        "the acoustic kernels to compare words with compiled WRF under "
        "tests/test_smallstep_horizontal_wrf471_parity.py, never in a forecast"),
    "woof/verify/smallstep_vertical_oracle.py": (
        "the small-step vertical oracle's harness: it compiles the acoustic "
        "vertical solve to compare words with compiled WRF under "
        "tests/test_smallstep_vertical_wrf471_parity.py, never in a forecast"),
}

#: ``compile_using_nvrtc`` sites among SITE_FILES: shortwave, RUC SPP,
#: shared diffusion, shared no-FTZ, member physics initialization and
#: (since the WRF 4.6.1 port) Kain-Fritsch.
#: (``rrtmg_lw.py`` and ``rrtmg_mcica.py`` take the same route but record
#: nothing, so they are not manifest sites and are not listed above.)
# The shared diffusion and no-FTZ loaders each compile and record their
# actual source/options alongside the three prior direct compiler sites.
# 2.9.0: the surface-state recipe joins them (see EXPECTED_SITE_COUNT), the
# WRF 4.6.1 Kain-Fritsch port (golden/parity), the land no-FTZ Noah-MP
# site (woof/core/ieee_module.py, golden/wrf-fixes) and the lightning unit
# (golden/ports-rad), which compiles through its own direct no-FTZ, no-FMA
# site in kernels.load_module and records it (record_module).
EXPECTED_NVRTC_SITE_COUNT = 9

# These direct-NVRTC transforms precede forecast construction. They bind
# compiler options and source bytes in their own preparation receipts.
PREPARATION_NVRTC_SITES = {
    "woof/ensemble/recentered.py": (
        "one-time physical source preparation before initialize_real at "
        "initial and boundary knots, never a forecast execution kernel; "
        "its operator receipt binds the source SHA, compiler options and full population mapping"),
}


def _is_rawmodule(node: ast.AST) -> bool:
    return (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "RawModule")


def _is_nvrtc(node: ast.AST) -> bool:
    """A direct ``compile_using_nvrtc`` call, however it was imported."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    name = (func.attr if isinstance(func, ast.Attribute)
            else func.id if isinstance(func, ast.Name) else None)
    return name == "compile_using_nvrtc"


#: Per route, how to reach the SOURCE and the OPTIONS argument:
#: ``(keyword, positional index)``.  ``compile_using_nvrtc(source, options,
#: arch, filename)`` takes both positionally at every site in this tree.
_COMPILE_ARGS = {
    "rawmodule": {"source": ("code", None), "options": ("options", None)},
    "nvrtc": {"source": ("source", 0), "options": ("options", 1)},
}


def _compiled_argument(call: ast.Call, route: str, which: str):
    keyword, index = _COMPILE_ARGS[route][which]
    found = _keyword(call, keyword)
    if found is not None:
        return found
    if index is not None and len(call.args) > index:
        return call.args[index]
    return None


def _is_record(node: ast.AST) -> bool:
    return (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "record_module")


def _keyword(call: ast.Call, name: str) -> ast.AST | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _functions(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def _literal_argument(tree, function, argument):
    """Resolve a singly assigned literal, without executing package code."""
    if not isinstance(argument, ast.Name):
        return argument
    globals_ = [node for node in tree.body if isinstance(node, ast.Assign)]
    locals_ = [node for node in ast.walk(function) if isinstance(node, ast.Assign)]
    local_values = [node.value for node in locals_ if any(
        isinstance(target, ast.Name) and target.id == argument.id
        for target in node.targets)]
    values = local_values or [node.value for node in globals_ if any(
        isinstance(target, ast.Name) and target.id == argument.id
        for target in node.targets)]
    if len(values) != 1:
        return argument
    try:
        value = ast.literal_eval(values[0])
    except (ValueError, TypeError):
        return argument
    return ast.parse(repr(value), mode="eval").body


def audit_source(path: str, text: str) -> list[str]:
    """Problems with the record/compile pairing in one module's source.

    Returns an empty list when every compile -- ``cp.RawModule`` or a direct
    ``compile_using_nvrtc`` -- is accompanied, in the same function, by a
    ``record_module`` call whose ``source`` and ``options`` expressions are
    syntactically the arguments the compiler was given.
    """
    tree = ast.parse(text)
    problems: list[str] = []
    for function in _functions(tree):
        compiles = [(node, "rawmodule" if _is_rawmodule(node) else "nvrtc")
                    for node in ast.walk(function)
                    if _is_rawmodule(node) or _is_nvrtc(node)]
        if not compiles:
            continue
        records = [node for node in ast.walk(function) if _is_record(node)]
        where = f"{path}:{function.name}"
        if len(records) != len(compiles):
            problems.append(
                f"{where} compiles {len(compiles)} module(s) but records "
                f"{len(records)}")
            continue
        for (compiled, route), recorded in zip(compiles, records):
            for which in ("source", "options"):
                left = _compiled_argument(compiled, route, which)
                right = _keyword(recorded, which)
                if left is None or right is None:
                    problems.append(f"{where} is missing {which}")
                    continue
                if ast.dump(_literal_argument(tree, function, left)) != ast.dump(
                        _literal_argument(tree, function, right)):
                    problems.append(
                        f"{where} records {which}={ast.unparse(right)} "
                        f"but compiles {which}={ast.unparse(left)}")
    return problems


def _site_text(path: str) -> str:
    return (REPO / path).read_text(encoding="utf-8")


def _discover_manifest_site_files(seeds=SITE_FILES):
    """Find new recorded compiler sites and unaccounted RawModule calls.

    Retain old sites even if their recording disappears, so deleting a
    record cannot remove the site from the audit. Existing explained
    non-manifest constructors keep their separate boundary checks.
    """
    found = set(seeds)
    for path in sorted((REPO / "woof").rglob("*.py")):
        relative = path.relative_to(REPO).as_posix()
        if relative in RAWMODULE_CONSTRUCTORS_OUTSIDE_THE_MANIFEST:
            continue
        if relative.startswith("woof/hex/"):
        # The folded hex model compiles through its own KernelCache and compile contract
        # (woof/hex/cuda_backend), held by its own tier-1 tests; this audit is the regional engine's.
            continue
        text = path.read_text(encoding="utf-8")
        if not any(token in text for token in ("RawModule", "record_module")):
            continue
        nodes = list(ast.walk(ast.parse(text)))
        if any(_is_rawmodule(node) for node in nodes) or (
                any(_is_record(node) for node in nodes)
                and any(_is_nvrtc(node) for node in nodes)):
            found.add(relative)
    return tuple(sorted(found))


SITE_FILES = _discover_manifest_site_files()


def test_every_rawmodule_site_is_accounted_for():
    total = 0
    for path in SITE_FILES:
        source = _site_text(path)
        tree = ast.parse(source)
        total += sum(1 for node in ast.walk(tree) if _is_rawmodule(node))
        assert audit_source(path, source) == [], path
    assert total > 0, "no RawModule compiler site was audited"


def test_every_rawmodule_constructor_in_the_tree_is_named():
    """The census behind SITE_FILES: no RawModule constructor may hide.

    The breakage this prevents: a module that compiles a translation unit
    outside the manifest records nothing about it, and a frame table, a
    parity receipt or a memory price can then describe a source the run
    never compiled.  Found when the Noah-MP factories were collapsed onto
    one site and the count dropped 8 -> 4: the old contract said "every
    module that constructs a RawModule" and nothing checked it.  In
    particular ``noahmp_frame_provenance.py`` constructs none -- its
    measurement compiles through the one Noah-MP site -- and this is
    where that stays asserted.
    """
    constructors = set()
    for path in sorted((REPO / "woof").rglob("*.py")):
        if path.relative_to(REPO).as_posix().startswith("woof/hex/"):
        # The folded hex model compiles through its own KernelCache and compile contract
        # (woof/hex/cuda_backend), held by its own tier-1 tests; this audit is the regional engine's.
            continue
        text = path.read_text(encoding="utf-8")
        if "RawModule" not in text and "RawKernel" not in text:
            continue
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (func.attr if isinstance(func, ast.Attribute)
                    else func.id if isinstance(func, ast.Name) else None)
            if name in ("RawModule", "RawKernel"):
                constructors.add(path.relative_to(REPO).as_posix())
    # woof/core/preflight.py's probe compiles nothing; doctor's probe is a
    # string, so its constructor is found by the text scan below instead.
    doctor = (REPO / "woof/doctor.py").read_text(encoding="utf-8")
    if "RawModule(" in doctor:
        constructors.add("woof/doctor.py")
    named = set(SITE_FILES) | set(RAWMODULE_CONSTRUCTORS_OUTSIDE_THE_MANIFEST)
    assert constructors <= named, sorted(constructors - named)
    assert "woof/core/noahmp_frame_provenance.py" not in constructors
    # Noah-MP's one compile site is direct NVRTC (woof/core/ieee_module.py)
    # since the land no-FTZ route; it constructs no RawModule any more.
    assert "woof/core/noahmp_kernel_sources.py" not in constructors
    for path in RAWMODULE_CONSTRUCTORS_OUTSIDE_THE_MANIFEST:
        assert path in constructors, f"{path} no longer constructs one; drop it"
    for path, reason in RAWMODULE_CONSTRUCTORS_OUTSIDE_THE_MANIFEST.items():
        assert len(reason.split()) >= 12, path


def test_aot_loader_only_loads_precompiled_images():
    tree = ast.parse(_site_text("woof/kernel_aot.py"))
    constructors = [node for node in ast.walk(tree) if _is_rawmodule(node)]
    assert constructors
    for call in constructors:
        assert not call.args
        assert [keyword.arg for keyword in call.keywords] == ["path"]


def test_every_direct_nvrtc_site_is_accounted_for():
    """The bypass route is inventoried too, or a site could leave silently.

    A site that moves off ``cp.RawModule`` and onto ``compile_using_nvrtc``
    -- which is what CUDA 13 forced on ``rrtmg_sw`` -- would otherwise drop
    out of the RawModule count and out of :func:`audit_source` at the same
    time, taking its manifest record with it and leaving both tests green.
    """
    total = 0
    for path in SITE_FILES:
        source = _site_text(path)
        tree = ast.parse(source)
        total += sum(1 for node in ast.walk(tree) if _is_nvrtc(node))
        assert audit_source(path, source) == [], path
    assert total > 0, "no direct compiler site was audited"


def test_preparation_nvrtc_route_preserves_subnormals_and_records_its_options():
    from tools.ftz_receipt import route_inventory
    from woof.ensemble import recentered
    for path,reason in PREPARATION_NVRTC_SITES.items():
        source = _site_text(path)
        sites = route_inventory.scan_source(path,source)
        assert len(sites) == 1
        assert sites[0]["constructor_kind"] == "cupy.cuda.compiler.compile_using_nvrtc"
        assert sites[0]["options_expression"] == "_CUDA_OPTIONS"
        assert len(reason.split()) >= 12
        assert "receipt[\"compiler_options\"] = list(_CUDA_OPTIONS)" in source
    assert recentered._CUDA_OPTIONS == ("-std=c++17", "--fmad=false", "--ftz=false")


def test_ruc_hydraulic_nvrtc_route_preserves_small_conductivities():
    from tools.ftz_receipt import route_inventory
    from woof.core import ruc_spp
    path = "woof/core/ruc_spp.py"
    sites = route_inventory.scan_source(path, _site_text(path))
    assert len(sites) == 1
    assert sites[0]["constructor_kind"] == "cupy.cuda.compiler.compile_using_nvrtc"
    assert sites[0]["options_expression"] == "MODULE_OPTIONS"
    assert ruc_spp.MODULE_OPTIONS == ("-std=c++17", "--fmad=false", "--ftz=false")


def test_kf_nvrtc_route_keeps_subnormals_and_contracts_nothing():
    """The KF column is graded word for word against gfortran -O0: no FMA,
    and no flush -- CuPy's RawModule appends -ftz=true, and the flushed
    build leaves the WRF oracle (tests/test_kf_wrf461_oracle.py)."""
    from tools.ftz_receipt import route_inventory
    from woof.core import kf
    path = "woof/core/kf.py"
    sites = route_inventory.scan_source(path, _site_text(path))
    assert len(sites) == 1
    assert sites[0]["constructor_kind"] == "cupy.cuda.compiler.compile_using_nvrtc"
    assert sites[0]["options_expression"] == "KF_MODULE_OPTIONS"
    assert kf.KF_MODULE_OPTIONS == ("-std=c++17", "--fmad=false", "--ftz=false")


@pytest.mark.parametrize("path", SITE_FILES)
def test_recorded_arguments_are_the_compiled_arguments(path):
    assert audit_source(path, _site_text(path)) == []


def test_option_tuples_are_per_site_and_not_one_global_flags_string():
    """The nest and shortwave modules carry options no other site carries."""
    recorded: dict[str, list[str]] = {}
    for path in SITE_FILES:
        tree = ast.parse(_site_text(path))
        for node in ast.walk(tree):
            if _is_record(node):
                options = _keyword(node, "options")
                recorded.setdefault(path, []).append(ast.unparse(options))
    nest = recorded["woof/core/nest_interp.py"]
    shortwave = recorded["woof/core/rrtmg_sw.py"]
    # Since aefc5a20 the nest loader builds its option tuple once (nest
    # preparation adds a define) and records the same name it compiles with;
    # the site's own -fmad=false is that tuple's literal start.
    assert nest == ["options"]
    assert 'options = ("-std=c++17", "-fmad=false")' in _site_text("woof/core/nest_interp.py")
    assert shortwave == ["('-std=c++17', '--ftz=false')"]
    assert nest != shortwave
    flattened = [option for options in recorded.values() for option in options]
    assert len(set(flattened)) > 1, (
        "every site recording the same options expression is the collapse "
        "this manifest exists to prevent")


# --- failure controls ------------------------------------------------------

def test_control_a_site_that_stops_recording_is_caught():
    """Failure control 1: delete a site's record_module call."""
    path = "woof/core/nest_interp.py"
    text = _site_text(path)
    mutated = text.replace(
        '    record_module(f"woof.core.nest_interp:{name}", source=src,\n'
        '                  options=options, module=mod)\n',
        "")
    assert mutated != text, "the control did not modify the site"
    problems = audit_source(path, mutated)
    assert problems, "a site that records nothing must be reported"
    assert "records 0" in problems[0]


def test_control_options_collapsed_to_a_global_string_is_caught():
    """Failure control 2: record the common flags instead of the site's."""
    path = "woof/core/nest_interp.py"
    text = _site_text(path)
    mutated = text.replace(
        '                  options=options, module=mod)',
        '                  options=("-std=c++17",), module=mod)')
    assert mutated != text, "the control did not modify the site"
    problems = audit_source(path, mutated)
    assert problems, "a site recording the wrong option tuple must be reported"
    assert "records options=" in problems[0]


def test_control_an_nvrtc_site_recording_other_options_is_caught():
    """Failure control 3: the bypass route is audited as strictly.

    ``rrtmg_sw`` compiles with ``--ftz=false`` because CuPy's RawModule route
    would inject ``-ftz=true`` and flush the subnormal transmittance products
    (MEASURED: the LW preflight's ``1e-30 * 1e-10`` comes back 0.0 through
    RawModule and 1e-40 through this route).  Recording an option tuple the
    compiler never saw would put that claim in the manifest without it being
    true, so mutate the record and require a complaint.
    """
    path = "woof/core/rrtmg_sw.py"
    text = _site_text(path)
    mutated = text.replace(
        '        record_module("woof.core.rrtmg_sw:rrtmg_sw", source=code,\n'
        '                      options=("-std=c++17", "--ftz=false"),',
        '        record_module("woof.core.rrtmg_sw:rrtmg_sw", source=code,\n'
        '                      options=("-std=c++17",),')
    assert mutated != text, "the control did not modify the site"
    problems = audit_source(path, mutated)
    assert problems, "an nvrtc site recording the wrong options must be caught"
    assert "records options=" in problems[0]


# --- the source hash, recomputed independently -----------------------------

def _independent_preamble() -> str:
    """Rebuild the preamble from the published construction, not from woof.

    ``woof/core/kernels/__init__.py`` builds one ``#define`` per entry of
    ``CUDA_DEFINES`` in ``float`` repr with an ``f`` suffix, then appends
    ``common.cuh``.  This is that construction written out again; if the
    production one changes, the two stop agreeing.
    """
    lines = [f"#define {key} {float(value)!r}f"
             for key, value in CUDA_DEFINES.items()]
    lines.append((KERNEL_DIR / "common.cuh").read_text())
    return "\n".join(lines) + "\n"


def test_independent_preamble_reconstruction_matches_production():
    from woof.core.kernels import _preamble

    assert _independent_preamble() == _preamble()


@pytest.mark.parametrize("name", ["acoustic", "advection", "diff6"])
def test_recorded_source_hash_is_the_hash_of_the_translation_unit(name):
    reset_kernel_manifest()
    source = _independent_preamble() + (
        KERNEL_DIR / f"{name}.cu").read_text()
    record_module(f"woof.core.kernels:{name}", source=source,
                  options=("-std=c++17",))
    expected = hashlib.sha256(source.encode("utf-8")).hexdigest()
    entry = kernel_manifest()[f"woof.core.kernels:{name}"]
    assert entry["source_sha256"] == expected
    assert entry["options"] == ["-std=c++17"]
    assert entry["compiled_image"]["status"] == "unavailable"
    assert entry["compiled_image"]["reason"]
    reset_kernel_manifest()


def test_source_sha256_is_a_plain_sha256_over_utf8_bytes():
    assert source_sha256("abc") == hashlib.sha256(b"abc").hexdigest()


def test_one_key_records_two_images_without_losing_either():
    """A negative control's variant compile is recorded, never overwritten."""
    reset_kernel_manifest()
    record_module("woof.core.kernels:probe", source="a",
                  options=("-std=c++17",))
    record_module("woof.core.kernels:probe", source="a",
                  options=("-std=c++17",))
    assert list(kernel_manifest()) == ["woof.core.kernels:probe"]

    # Same unit, different options: a second compiled image, a second entry.
    record_module("woof.core.kernels:probe", source="a",
                  options=("-std=c++17", "-fmad=false"))
    # Same options, different source: a third.
    record_module("woof.core.kernels:probe", source="b",
                  options=("-std=c++17",))
    manifest = kernel_manifest()
    assert len(manifest) == 3, manifest
    sources = {entry["source_sha256"] for entry in manifest.values()}
    assert sources == {source_sha256("a"), source_sha256("b")}
    options = {tuple(entry["options"]) for entry in manifest.values()}
    assert options == {("-std=c++17",), ("-std=c++17", "-fmad=false")}
    # Deterministic: recording the same three again adds nothing.
    record_module("woof.core.kernels:probe", source="a",
                  options=("-std=c++17", "-fmad=false"))
    record_module("woof.core.kernels:probe", source="b",
                  options=("-std=c++17",))
    assert kernel_manifest() == manifest
    reset_kernel_manifest()


def test_the_conflict_error_still_exists_for_an_indistinguishable_collision():
    assert issubclass(KernelManifestConflict, RuntimeError)


@requires_gpu
def test_gpu_a_compiled_module_records_the_source_the_test_rebuilds():
    """The GPU half: compile through the production loader and check the record.

    D-19's CPU-checkable half is the source and option record; this exercises
    it end to end on a device, and reports whichever compiled-image status the
    installed CuPy actually permits rather than requiring one.
    """
    from woof.core.kernels import load_module

    reset_kernel_manifest()
    # The loader is lru_cached, so an earlier test in this process may already
    # hold the module; the record only fires on a real compile.
    load_module.cache_clear()
    load_module("diff6")
    entry = kernel_manifest()["woof.core.kernels:diff6"]
    expected = hashlib.sha256(
        (_independent_preamble()
         + (KERNEL_DIR / "diff6.cu").read_text()).encode("utf-8")).hexdigest()
    assert entry["source_sha256"] == expected
    expected_options = ["-std=c++17", "--fmad=false", "--ftz=false",
                        "--prec-div=true", "--prec-sqrt=true",
                        "-DGPUWM_WRF_EXACT_C_DIFFUSION=1"]
    from woof.wrf_exact import ENABLED, effective_controls
    if ENABLED:
        expected_options = ["-std=c++17", "--fmad=false", "--ftz=false",
                            "--prec-div=true", "--prec-sqrt=true",
                            "-DGPUWM_WRF_EXACT=1"]
        for control, macro in (("diagnostics", "GPUWM_WRF_EXACT_D_DIAGNOSTICS"),
                               ("advection", "GPUWM_WRF_EXACT_C_ADVECTION"),
                               ("diffusion", "GPUWM_WRF_EXACT_C_DIFFUSION"),
                               ("bigstep", "GPUWM_WRF_EXACT_C_BIGSTEP")):
            if effective_controls()[control]:
                expected_options.append(f"-D{macro}=1")
    assert entry["options"] == expected_options
    assert entry["compiled_image"]["status"] in {"resolved", "unavailable"}
    reset_kernel_manifest()


def test_letkf_device_records_its_original_compiler_inputs(monkeypatch):
    import importlib
    import sys
    import types
    from woof.da import letkf_device
    # The package exports a function named kernel_manifest. Import the
    # submodule explicitly so this seam patches the compiler's recorder.
    manifest = importlib.import_module("woof.certify.kernel_manifest")
    events = []
    class Device:
        def __init__(self, device):
            self.device = device
        def __enter__(self):
            events.append(("device", self.device))
        def __exit__(self, *args):
            return False
    class Module:
        def __init__(self, *, code, options):
            self.code, self.options = code, options
            events.append(("construct", code, options))
        def compile(self):
            events.append(("compile",))
    records = []
    monkeypatch.setitem(sys.modules, "cupy", types.SimpleNamespace(
        cuda=types.SimpleNamespace(Device=Device), RawModule=Module))
    monkeypatch.setattr(manifest, "record_module", lambda key, **args: records.append((key, args)))
    letkf_device._module.cache_clear()
    try:
        module = letkf_device._module(7, 2)
        assert events == [("device", 2), ("construct", module.code, module.options), ("compile",)]
        assert module.code == letkf_device._SOURCE.replace("RR", "7")
        assert module.options == ("--fmad=false", "-std=c++14")
        assert records == [("letkf_device_members_7", {
            "source": module.code, "options": module.options, "module": module})]
    finally:
        letkf_device._module.cache_clear()
