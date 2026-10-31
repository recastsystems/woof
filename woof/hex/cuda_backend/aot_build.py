"""Wheel-time CUDA driver orchestration. No CUDA arithmetic lives here."""
from __future__ import annotations
from woof.hex import numerics as _hex_numerics

import ast
import base64
import hashlib
import importlib
import json
import lzma
from functools import lru_cache
from pathlib import Path
import re

from .aot import (ARCHITECTURES, DIRECT_ROUTE, RAW_ROUTE, compiler_identity, local_options,
                  portable_options, preserve_final_options, recipe, source_hash)


def inventory(package_root=None, numerics=None, compiler_family=None, *, compiler=None, catalogs=True):
    """Read production KernelCache declarations, using their source bindings.

    ``numerics`` names the kernel numerics the rows compile with (``fast`` or ``strict``,
    :mod:`woof.hex.numerics`); ``None`` is this process's mode.
    ``compiler`` binds the recorded CuPy header layout to its CUDA family.
    ``compiler_family`` selects that family's recorded include paths; ``None``
    retains all catalogs for inspection. The legacy catalog belongs to CUDA 13.

    Proof instruments have local source variables and are excluded. Production
    declarations that cannot be resolved fail the build rather than disappear.
    ``catalogs=False`` returns the declared rows only (no recorded recipes).
    """
    root = Path(package_root or Path(__file__).resolve().parents[1])
    result = {}
    excluded = {"cuda_ftz.py", "cuda_ftz_v841.py", "cuda_dualrun.py"}
    for path in sorted(root.rglob("*.py")):
        if path.name in excluded or "drivers" in path.parts:
            continue
        # Shape resolution forwards runtime source/key arguments. Its generic
        # carriers remain declared at the callers; concrete shapes are recorded
        # by first-use discovery, rather than evaluated from this forwarding ABI.
        if path.name in {"runtime.py", "aot.py", "aot_build.py", "cuda_shape.py"}:
            continue
        text = path.read_text(encoding="utf-8-sig")
        # Only a file whose text names raw_kernel can hold a raw_kernel(s) call, so the rest are not parsed:
        # the same rows.  THE COST THIS REMOVES (V8X6000, 2026-10-29): parsing and walking all ~500 files was
        # 10 of the 15 s the cold catalog took on one thread before the compile pool could start.
        if "raw_kernel" not in text:
            continue
        tree = ast.parse(text)
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute)
                 and n.func.attr in ("raw_kernel", "raw_kernels") and len(n.args) >= 2]
        if not calls:
            continue
        module_name = "woof.hex." + ".".join(path.relative_to(root).with_suffix("").parts)
        namespace = vars(importlib.import_module(module_name))
        # The resident launcher takes a unit and source as arguments. Resolve
        # its two production bindings using the same source providers, without
        # constructing a session or querying a GPU during wheel discovery.
        bindings = [namespace]
        if module_name == "woof.hex.mpasphys.resident_v842":
            bindings = [dict(namespace, unit="latitude", source=namespace["_LATITUDE"]),
                        dict(namespace, unit="preparation", source=namespace["preparation_source"]())]
        for call in calls:
            keywords = {k.arg: k.value for k in call.keywords}
            if "module_key" not in keywords:
                raise ValueError(f"{path}:{call.lineno}: missing module_key")
            for binding in bindings:
                def value(node):
                    return eval(compile(ast.Expression(node), str(path), "eval"), binding)
                try:
                    key = value(keywords["module_key"])
                    source = value(call.args[1])
                    extra = value(keywords["options"]) if "options" in keywords else ()
                except Exception as error:
                    raise ValueError(f"{path}:{call.lineno}: unresolved kernel declaration: {error}") from error
                row = {"module_key": key, "source": source,
                       "options": [*_hex_numerics.kernel_base(numerics), *extra],
                       "source_sha256": source_hash(source)}
                if module_name == "woof.hex.mpasphys.resident_v842":
                    # These calls use MpasPhysicsKernelCache, whose carrier
                    # route appends these flags and bypasses RawModule's FTZ
                    # override. Record that actual compile request.
                    carrier = (*row["options"], "--fmad=false", "--ftz=false",
                               "--prec-div=true", "--prec-sqrt=true")
                    row = recipe({"source": source,
                                  "options": _hex_numerics.options(carrier, numerics=numerics),
                                  "route": "direct-nvrtc", "filename": "kern.cu"})
                digest = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
                result[digest] = row
    # Direct-NVRTC regional declarations retain native subnormal words.
    from woof.hex.cuda_regional_v841 import (
        MODULE_KEY, CUDA_REGIONAL_SOURCE, REGIONAL_CUDA_OPTIONS,
    )
    regional = {"module_key": MODULE_KEY, "source": CUDA_REGIONAL_SOURCE,
                "options": list(_hex_numerics.options(REGIONAL_CUDA_OPTIONS, numerics=numerics)),
                "source_sha256": source_hash(CUDA_REGIONAL_SOURCE),
                "compiler_route": "direct-nvrtc"}
    digest = hashlib.sha256(json.dumps(regional, sort_keys=True).encode()).hexdigest()
    result[digest] = regional
    for row in physics_inventory(numerics):
        digest = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
        result[digest] = row
    from . import arch_admission
    from woof.hex.cuda_fp32 import CUDA_FTZ_HELPERS
    probe = recipe({"source": CUDA_FTZ_HELPERS + arch_admission._PROBE_KERNEL,
                    "options": list(_hex_numerics.kernel_base(numerics))})
    result[probe["module_key"]] = probe
    # Preserve each compiler family's exact include layout. A CUDA 13
    # recording has nvidia/cu13, while CUDA 12 requests cuda_runtime; treating
    # those options as aliases would weaken the loader's compile contract.
    # Only source-bound KernelCache declarations suppress recorded copies.
    # Earlier catalogs must not suppress a later family's different options.
    if not catalogs:
        return list(result.values())
    declared = {row["source_sha256"] for row in result.values()}
    selections = {selection(row) for row in result.values()}
    catalogs = sorted((root / "data").glob("compiler-recipes*.json"))
    if compiler_family is not None:
        family_catalog = root / "data" / f"compiler-recipes-cuda{int(compiler_family)}.json"
        if not family_catalog.is_file() and int(compiler_family) == 13:
            family_catalog = root / "data" / "compiler-recipes.json"
        if not family_catalog.is_file():
            raise ValueError(f"no compiler recipe catalog for CUDA {compiler_family}: {family_catalog}")
        catalogs = [family_catalog]
    for catalog in catalogs:
        entries = json.loads(catalog.read_text(encoding="utf-8"))
        if not isinstance(entries, list) or len(entries) > 4000:
            raise ValueError("compiler recipe inventory is malformed or oversized")
        # A KernelCache module is served under its own module key before
        # CuPy is asked, so the same source recorded through CuPy is skipped.
        for entry in entries:
            row = catalog_recipe(entry, compiler)
            # A direct or raw NVRTC recording is its own request (its own key,
            # options and program name) even when a KernelCache declares the
            # same source.  THE BREAKAGE (VRFS, 0.4.9, 2026-10-28): the output
            # diagnostics, number carriers, resident latitude and v841 physics
            # prep were recorded on the direct route and dropped here, so every
            # fresh box compiled them at first use in both numerics.
            if (row["source_sha256"] in declared
                    and row.get("compiler_route") not in (DIRECT_ROUTE, RAW_ROUTE)):
                continue
            if selection(row) in selections:
                continue
            selections.add(selection(row))
            digest = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
            result[digest] = row
    if not result:
        raise ValueError("empty CUDA module inventory cannot produce an AOT wheel")
    return list(result.values())


def selection(row):
    """What a pack row is selected by: module key, exact options and compile request."""
    return (row["module_key"], tuple(row.get("options", ())),
            json.dumps(row.get("compile_request"), sort_keys=True))


def declared_requests(package_root=None):
    """What the declared inventory already serves, under either numerics.

    Returns the declared source digests (a CuPy-route recording of one is served
    by its declared module key before CuPy is asked, and :func:`inventory` skips
    it) and the declared selections (a direct recording identical to a declared
    physics row adds nothing), so a recipe catalog need not carry either.
    """
    rows = [row for mode in ("strict", "fast")
            for row in inventory(package_root, numerics=mode, catalogs=False)]
    return {row["source_sha256"] for row in rows}, {selection(row) for row in rows}


_GLOBAL_NAME = re.compile(r'__global__\s+(?:__launch_bounds__\s*\([^)]*\)\s*)?void\s+'
                          r'(?:__launch_bounds__\s*\([^)]*\)\s*)?([A-Za-z_][A-Za-z_0-9]*)\s*\(')


def recipe_numerics(options):
    """``fast``, ``strict`` or ``neutral`` (a unit whose options name no hex numerics flag)."""
    options = [str(option) for option in options]
    if "-DHEX_NUMERICS_FAST=1" in options or "--fmad=true" in options:
        return "fast"
    if "--fmad=false" in options:
        return "strict"
    return "neutral"


def _recorded_unit(entry):
    """(file name, entry points) of a direct or raw NVRTC row, catalog or recipe form."""
    route = entry.get("route") or entry.get("compiler_route")
    if route not in (DIRECT_ROUTE, RAW_ROUTE):
        return None
    filename = entry.get("filename") or (entry.get("compile_request") or {}).get("filename")
    names = frozenset(_GLOBAL_NAME.findall(entry.get("source") or ""))
    return (str(filename), names) if names else None


def numerics_gaps(entries, fast_only=()):
    """Direct-NVRTC units held under one numerics mode only.

    THE BREAKAGE THIS NAMES (VRFS, woof hex 0.4.9, 2026-10-28): the catalog's
    MPAS-route units (RRTMG SW, YSU GWDO, the cloud fraction, sea-ice and
    surface units, the RRFS schemes) were recorded from fast runs only, so the
    pack held no strict image and a fresh box's first strict step compiled them
    on one core: 547 s on one card, one module 519 s.  A unit is its file name
    and its ``__global__`` entry points; two rows of one file sharing an entry
    point are the same unit.  Pass the catalog rows and
    the declared direct rows (:func:`physics_inventory`, both modes).  A unit
    held with hex fast flags needs a strict row; a strict unit needs a fast or
    flag-free row (the engine's own units carry no hex flag in fast numerics).
    Returns ``(missing mode, file name, entry points)``, empty when complete.
    ``fast_only`` names units strict numerics never launches (:data:`FAST_ONLY_FAMILIES`); a fast unit
    whose entry points all lie in one of them needs no strict row.
    """
    held = {"fast": set(), "strict": set(), "neutral": set()}
    for entry in entries:
        unit = _recorded_unit(entry)
        if unit is not None:
            held[recipe_numerics(entry.get("options", ()))].add(unit)
    def paired(unit, others):
        # Same file and a shared entry point: a unit whose fast source adds or
        # splits kernels (Thompson's level-parallel phases) still pairs.
        return any(unit[0] == other[0] and unit[1] & other[1] for other in others)
    def fast_only_unit(unit):
        return any(unit[0] == other[0] and unit[1] <= other[1] for other in fast_only)
    gaps = [("strict", unit) for unit in held["fast"]
            if not paired(unit, held["strict"]) and not fast_only_unit(unit)]
    gaps += [("fast", unit) for unit in held["strict"]
             if not paired(unit, held["fast"] | held["neutral"])]
    return sorted((mode, filename, tuple(sorted(names))) for mode, (filename, names) in gaps)


#: Physics families compiled under fast numerics only.  Strict never launches them, so a strict image
#: cannot exist and :func:`numerics_gaps` does not ask for one: TEMPO's phase module P
#: (tempo_phase_v303.phase_source, kernels tempo_phase_a..e; rrfs/tempo-yolo 0a09dec9), because strict
#: TEMPO is the warp-per-column module, bitwise to native RRFS-MPAS.  Module S (tempo_rrfs_tail) carries
#: the warp fallback, so it pairs with the strict warp module and is not listed.
FAST_ONLY_FAMILIES = frozenset({"tempo_rrfs_phase"})


def catalog_numerics_gaps(catalog_rows):
    """:func:`numerics_gaps` of a recipe catalog beside the declared direct physics rows."""
    declared = [row for mode in ("strict", "fast") for row in physics_inventory(mode)]
    fast_only = {unit for row in declared if row.get("physics_family") in FAST_ONLY_FAMILIES
                 for unit in (_recorded_unit(row),) if unit is not None}
    return numerics_gaps([*catalog_rows, *declared], fast_only=fast_only)


_PIP_CUDA_INCLUDE = "-I@SITE@/nvidia/"


def toolkit_includes():
    """The CUDA header ``-I`` options this CuPy appends, when they name a toolkit outside site-packages.

    CuPy adds its CUDA include directory from the NVRTC it loaded: the pip
    headers (``<site-packages>/nvidia/cu13/include``) or, when a system toolkit
    shadows the pip wheel, ``/usr/local/cuda/include``.  ``None`` for the pip
    layout or without CuPy, so the recorded layout stands.
    """
    try:
        from cupy._core import core
        assembled = core.assemble_cupy_compiler_options(())
    except Exception:
        return None
    portable = portable_options([option for option in assembled if str(option).startswith("-I")])
    toolkit = [option for option in portable if not option.startswith("-I@SITE@/")]
    return toolkit or None


def catalog_recipe(entry, compiler=None):
    """Bind recorded CuPy include roots to the compiler the rows are formed for.

    CUDA 12 and 13 pip runtimes put headers in different directories. Keep
    those paths exact in the pack so production selection needs no aliases.
    With ``compiler`` (this process's compiler) and a system toolkit's headers
    in CuPy's own options, the recorded pip include becomes that toolkit's.
    THE BREAKAGE THIS ENDS (CUDA 13.0 driver 580, 2026-10-24): CuPy
    loaded the system libnvrtc and compiled with ``-I/usr/local/cuda/include``,
    so 150 of 217 first-run requests matched neither the pack nor the
    ahead-of-time pool and compiled in the foreground on one core.
    """
    options = portable_options(entry.get("options", ()))
    version = (compiler or {}).get("nvrtc") or ()
    if version and version[0] == 12:
        options = ["-I@SITE@/nvidia/cuda_runtime/include"
                   if option == "-I@SITE@/nvidia/cu13/include" else option
                   for option in options]
    toolkit = toolkit_includes() if compiler is not None else None
    if toolkit and any(option.startswith(_PIP_CUDA_INCLUDE) for option in options):
        bound = []
        for option in options:
            if option.startswith(_PIP_CUDA_INCLUDE):
                bound.extend(toolkit)
                toolkit = []
            else:
                bound.append(option)
        options = bound
    return recipe({**entry, "options": options})


def physics_inventory(numerics=None):
    """Direct MYNN, MYNN-EDMF and sfclay images, in exact and corrected forms.

    Raw NVRTC call sites are not KernelCache declarations and were absent from
    every wheel inventory, leaving minutes of serial compilation on a new box.
    Assemble these with the production source providers and the same flags.
    """
    from woof.hex.mpasphys import bl_mynn_v842 as bl, sf_mynn_v842 as sf, sfclay_mm5_v842 as mm5
    from woof.hex.mpasphys import resources_build as resources
    from woof.hex.mpasphys import noah_v842 as noah, rrtmg_lw_v842 as lw
    from woof.hex.mpasphys import gf_mpas_v842 as gf, thompson_v842 as thompson
    from woof.hex.mpasphys import thompson_aero_v842 as thompson_aero
    from woof.hex.mpasphys import ntiedtke_v842 as ntiedtke
    from woof.hex.mpasphys import ysu_v842 as ysu
    from woof.hex.mpasphys import ysu_gwdo_v842 as ysu_gwdo
    from woof.hex.mpasphys.rrfs import sfclay_mm5 as rrfs_mm5
    from woof.hex.mpasphys.rrfs import bl_mynnedmf_c942418 as edmf
    from woof.hex.mpasphys.rrfs import smoke_9652bbf as smoke, smoke_resident
    from woof.hex.cuda_fp32 import CUDA_FTZ_HELPERS
    from woof.hex.mynn_glibc_math import mynn_glibc_source

    declarations = [
        ("ntiedtke", ntiedtke.cuda_source(), ntiedtke.CUDA_OPTIONS, "kern.cu"),
        ("ysu", ysu.cuda_source(ysu.shared_device_math_header()), ysu.COMPILER_OPTIONS, "kern.cu"),
        ("bl_mynn_adapter", bl.ADAPTER_CUDA_SOURCE, bl.MYNN_CUDA_OPTIONS, "kern.cu"),
    ]
    # One MYNN-EDMF image per RRFS pin the route table names (resources_build.rrfs_mynnedmf_images), so the
    # pack carries the hrrrv5 default pin's image, not only the 1d6e370e one.
    import importlib
    for _key, module in resources.rrfs_mynnedmf_images():
        edmf = importlib.import_module("woof.hex.mpasphys." + module)
        declarations.append(("bl_mynnedmf", CUDA_FTZ_HELPERS + mynn_glibc_source() + edmf.ARRAY_SUPPORT
                             + edmf.MODULE_CUDA, edmf.CUDA_OPTIONS, "rrfs_mynnedmf.cu"))
    # The forecast applies RP7 before selecting the fast column scheduler.
    # A recorded fast run cannot supply strict/correct's different source:
    # 0.4.9 omitted that image and its first MYNN call compiled for ~499 s.
    # Assemble every supported request with the production transformations,
    # without changing either the kernel text or the process's numerics mode.
    from woof.hex.mpasphys.driver_v842 import correct_source
    mode = _hex_numerics.mode() if numerics is None else _hex_numerics.normalise(numerics)
    for corrections in (frozenset(), frozenset({"RP7"})):
        numerical = correct_source(bl.MYNN_MODULE_CUDA, bl.MYNN_TEXT_CORRECTIONS,
                                   corrections, "bl_mynn hex-cuda")
        if mode == "fast":
            numerical = bl._dynamic_columns(numerical)
        declarations.append(("bl_mynn", CUDA_FTZ_HELPERS
                             + resources.mynn_maths(resources._vendored_maths())
                             + bl.MYNN_ARRAY_SUPPORT + numerical,
                             bl.MYNN_CUDA_OPTIONS, "kern.cu"))
    for corrections in ((), ("RP3",)):
        declarations.extend([
            ("sf_mynn", sf._cuda_source(frozenset(corrections)), sf.CUDA_OPTIONS, "kern.cu"),
            ("sfclay_mm5", mm5._shared_math_source() + mm5._cuda_text(corrections),
             mm5.COMPILE_OPTIONS, "sfclay_mm5_v842.cu"),
        ])
    declarations.append(("rrfs_sfclay_mm5", rrfs_mm5.cuda_source(),
                         mm5.COMPILE_OPTIONS, "rrfs_mm5.cu"))
    noah_source = (noah._device_math_source() + noah._CORE_CUDA + noah._ENTRY_CUDA
                   + noah._LANDUSE_CUDA + noah._MPAS_IO_CUDA + noah._STATE_CUDA)
    noah_source = noah._shared_tables_source(noah_source)
    lw_source = (lw._solver_source() + "\n" + lw._RANDOM_SOURCE + "\n"
                 + lw._DRIVER_SOURCE + "\n" + lw._INIT_SOURCE)
    lw_source = lw_source.replace('#define RLW_MAXLAY 128',
                                  f'#define RLW_MAXLAY {lw.MAX_RADIATION_LAYERS}')
    declarations.append(("rrtmg_lw", lw_source,
                         ("-std=c++17", "--ftz=false", "--fmad=false",
                          "--prec-div=true", "--prec-sqrt=true"), "mpas_rrtmg_lw_v842.cu"))
    from woof.hex.mpasphys import rrtmg_sw_v842 as sw
    declarations.append(("rrtmg_sw", shortwave_source(), sw.COMPILE_OPTIONS,
                         "mpas_rrtmg_sw_v842.cu"))
    rows = [{**recipe({"source": source,
                       "options": list(_hex_numerics.options(options, numerics=numerics)),
                       **({"cache_in_memory": True} if family == "ysu" else {}),
                       "route": "direct-nvrtc", "filename": filename}), "physics_family": family}
            for family, source, options, filename in declarations]
    # One image per ported SMOKE pin: 9652bbf (1d6e370e row) and 5d2c71e2 (the shipping 44c0849 row).
    from woof.hex.mpasphys.rrfs import smoke_5d2c71e2
    for pinned in (smoke, smoke_5d2c71e2):
        rows.append({**recipe({"source": pinned.cuda_source(smoke_resident.math_source(pinned), columns=True),
                               "options": list(_hex_numerics.options(pinned.cuda_options(columns=True),
                                                                     numerics=numerics)),
                               "route": "raw-nvrtc", "filename": "rrfs_smoke.cu"}),
                     "physics_family": "rrfs_smoke"})
    rows.append({**recipe({"source": noah_source,
                           "options": list(_hex_numerics.options(noah.CUDA_OPTIONS,
                                           keep_ftz=True, numerics=numerics)),
                           "route": "direct-nvrtc", "filename": "kern.cu"}),
                 "physics_family": "noah_shared_tables"})
    rows.append({**recipe({"source": gf.cuda_source(),
                           "options": list(_hex_numerics.options(gf.COMPILER_OPTIONS, numerics=numerics)),
                           "route": "raw-nvrtc", "filename": "gf_mpas_v842.cu"}),
                 "physics_family": "gf_mpas"})
    from woof.hex.mpasphys import seaice_v842 as seaice
    from woof.hex.mpasphys.rrfs import noah_seaice as rrfs_seaice
    for family, source, filename in (
            ("noah_seaice", seaice.cuda_source(4), "kern.cu"),
            ("rrfs_noah_seaice", rrfs_seaice.cuda_source(4), "rrfs_noah_seaice.cu")):
        rows.append({**recipe({"source": source,
                               "options": list(_hex_numerics.options(("--std=c++11", "--fmad=false",
                                   "--ftz=false", "--prec-div=true", "--prec-sqrt=true"),
                                   keep_ftz=True, numerics=numerics)),
                               "route": "direct-nvrtc", "filename": filename}),
                     "physics_family": family})
    # MPAS and RRFS standard vertical grids, with production workspace storage.
    for levels in (55, 59):
        for native_dx in (False, True):
            rows.append({**recipe({"source": ysu_gwdo.kernel_source(levels, native_dx),
                                   "options": list(_hex_numerics.options(ysu_gwdo.COMPILE_OPTIONS, numerics=numerics)),
                                   "cache_in_memory": True,
                                   "route": "direct-nvrtc", "filename": "ysu_gwdo_v842.cu"}),
                         "physics_family": "ysu_gwdo"})
        for corrections in ((), ("WB3",)):
            source = thompson.cuda_source(device_math_source=resources.thompson_maths(resources._vendored_maths()),
                                          max_levels=levels, corrections=corrections, workspace=True, numerics=numerics)
            rows.append({**recipe({"source": source,
                                   "options": list(_hex_numerics.options(thompson.COMPILE_OPTIONS, numerics=numerics)),
                                   "route": "direct-nvrtc", "filename": "kern.cu"}),
                         "physics_family": "thompson_workspace"})
        # The same column module without the per-column workspace (thompson_v842._module workspace=False, exact
        # text). THE BREAKAGE THIS ENDS: absent here, a cold convection_permitting run compiled it on one core at
        # its first radar/history output, 12.3-13.5 s after step 1 (V8X6000, 64 -> 2 km zoom globe, 2026-10-29).
        source = thompson.cuda_source(device_math_source=resources.thompson_maths(resources._vendored_maths()),
                                      max_levels=levels, corrections=(), workspace=False, numerics=numerics)
        rows.append({**recipe({"source": source,
                               "options": list(_hex_numerics.options(thompson.COMPILE_OPTIONS, numerics=numerics)),
                               "route": "direct-nvrtc", "filename": "kern.cu"}),
                     "physics_family": "thompson"})
        # Thompson-aerosol (mp_thompson_aerosols: the hrrr-like-aerosol profile, the regional GFS cold-start
        # default since 0.4.10) on the same shared driver build, as launch_driver compiles it.
        # THE BREAKAGE THIS ENDS: absent here, every pack missed the default's microphysics
        # (direct lookup module_missing at 55 and 59 levels, both numerics, every arch, while mp_thompson
        # hit), so a fresh box compiled the whole Thompson column on one core at the first microphysics step.
        source = thompson_aero.cuda_source(device_math_source=resources.thompson_maths(resources._vendored_maths()),
                                           max_levels=levels, workspace=True, numerics=numerics)
        rows.append({**recipe({"source": source,
                               "options": list(_hex_numerics.options(thompson.COMPILE_OPTIONS, numerics=numerics)),
                               "route": "direct-nvrtc", "filename": "kern.cu"}),
                     "physics_family": "thompson_aero_workspace"})
    from woof.hex.mpasphys import wsm6_v842 as wsm6
    from woof.hex.mpasphys.rrfs import wsm6 as rrfs_wsm6
    from woof.hex.rkind_libm import wsm6_device_math_header
    header = wsm6_device_math_header()
    rows.append({**recipe({"source": rrfs_wsm6.relative_humidity_source(),
                           "options": list(wsm6.compiler_options(numerics)),
                           "route": "direct-nvrtc", "filename": "kern.cu"}),
                 "physics_family": "rrfs_wsm6_relhum"})
    for levels in (55, 59):
        # Correct mode (the MPAS-suite door default) compiles WSM6 with its applied findings (WB3, WB5) beside
        # the exact text. THE BREAKAGE THIS ENDS: absent here, every default-mode mesoscale_reference run
        # NVRTC-compiled the corrected WSM6 unit on one core before its first microphysics step.
        # cuda_entry passes slot_corrections(call, CORRECT_MODE_SOURCE): every finding WSM6 has text for, sorted.
        corrected = tuple(sorted(wsm6.CORRECT_MODE_SOURCE))
        rows.append({**recipe({"source": wsm6.cuda_source(header, max_levels=levels, corrections=corrected,
                                                          numerics=numerics),
                               "options": list(wsm6.compiler_options(numerics)),
                               "route": "direct-nvrtc", "filename": "kern.cu"}),
                     "physics_family": "wsm6"})
        for family, source in (
                ("wsm6", wsm6.cuda_source(header, max_levels=levels, numerics=numerics)),
                ("wsm6_workspace", wsm6.workspace_size_source(header, levels, numerics=numerics)),
                ("rrfs_wsm6_radar", rrfs_wsm6.radar_source(header, levels, numerics=numerics))):
            rows.append({**recipe({"source": source,
                                   "options": list(wsm6.compiler_options(numerics)),
                                   "route": "direct-nvrtc", "filename": "kern.cu"}),
                         "physics_family": family})
    # RRFS TEMPO (hrrrv5 and every mp_tempo suite) at the RRFS init door's L60 grid (59 layers): the
    # warp-per-column module tempo_v303.launch_mpas compiles. THE BREAKAGE THIS ENDS: absent here, a
    # fresh box compiled it on one core at the first microphysics step (27-29 s on an RTX 5090 host, both numerics).
    from woof.hex.mpasphys.rrfs import tempo_v303 as tempo
    rows.append({**recipe({"source": tempo.warp_source(resources.tempo_maths(resources._vendored_maths()),
                                                        max_levels=TEMPO_RRFS_LEVELS, numerics=numerics),
                           "options": list(_hex_numerics.options(tempo.COMPILE_OPTIONS, numerics=numerics)),
                           "route": "direct-nvrtc", "filename": "kern.cu"}),
                 "physics_family": "tempo_rrfs"})
    # TEMPO's reflectivity of a given state (tempo_refl_v303: the f000 frame of a correct-mode TEMPO forecast
    # and the DA posterior observers). It compiles the same strict text under both numerics, so it is declared
    # once, with the strict rows; the pack serves it by its text to a fast run too. THE BREAKAGE THIS ENDS:
    # absent here, every correct-mode TEMPO forecast compiled it on one core beside its set-up.
    if (numerics or _hex_numerics.mode()) == "strict":
        from woof.hex.mpasphys.rrfs import tempo_refl_v303 as tempo_refl
        rows.append({**recipe({"source": tempo_refl.module_source(TEMPO_RRFS_LEVELS),
                               "options": list(tempo_refl.module_options()),
                               "route": "direct-nvrtc", "filename": "kern.cu", "cache_in_memory": True}),
                     "physics_family": "tempo_rrfs_refl"})
    # The same NGW image contains initialization, column execution and writeback.
    # RRFS L60 otherwise compiled it for 599.7 s at first use on NVRTC 13.4.
    from woof.hex.mpasphys import ugwp_ngw_v842 as ngw
    rows.append({**recipe({"source": ngw.cuda_source(TEMPO_RRFS_LEVELS),
                           "options": list(_hex_numerics.options(ngw.COMPILE_OPTIONS, numerics=numerics)),
                           "route": "direct-nvrtc", "filename": "ugwp_ngw_v842.cu"}),
                 "physics_family": "ugwp_ngw"})
    # Fast numerics runs TEMPO as the (level, cell) phase kernels (module P) plus the warp tail (module S,
    # which also carries the first-call warp fallback and the driver slice kernels): tempo_phase_v303._modules
    # compiles both through direct_image. THE BREAKAGE THIS ENDS: absent here, every fast run NVRTC-compiled
    # both modules on one core at the first microphysics step (33.2 s vs 5.6 s, tempo-yolo 0a09dec9).
    if (numerics or _hex_numerics.mode()) == "fast":
        from woof.hex.mpasphys.rrfs import tempo_phase_v303 as tempo_phase
        maths = resources.tempo_maths(resources._vendored_maths())
        inline = tempo_phase.DEFAULT_INLINE
        for family, source in (("tempo_rrfs_phase", tempo_phase.phase_source(maths, TEMPO_RRFS_LEVELS, inline)),
                               ("tempo_rrfs_tail", tempo_phase.tail_source(maths, TEMPO_RRFS_LEVELS, inline))):
            rows.append({**recipe({"source": source,
                                   "options": list(_hex_numerics.options(tempo.COMPILE_OPTIONS, numerics="fast")),
                                   "route": "direct-nvrtc", "filename": "kern.cu"}),
                         "physics_family": family})
    from woof.hex.mpasphys.rrfs import mp_nssl2m as nssl, nssl_math
    rows.append({**recipe({"source": nssl.cuda_source(shared_math_source=nssl_math.source(), numerics=numerics),
                           "options": list(_hex_numerics.options(nssl.CUDA_OPTIONS, numerics=numerics)),
                           "route": "raw-nvrtc", "filename": "nssl.cu"}),
                 "physics_family": "nssl_rrfs"})
    return rows


@lru_cache(maxsize=1)
def _shortwave_table_defines():
    """The certified table layout, without any recorded implementation text.

    The table offsets are fixed by the verified MPAS table set. Kernel code
    and math headers come from their live providers in shortwave_source.
    A different layout still has a different exact compiler request and
    follows the normal foreground path; no cache key is relaxed.
    """
    catalog = Path(__file__).resolve().parents[1] / "data" / "compiler-recipes.json"
    for entry in json.loads(catalog.read_text(encoding="utf-8")):
        if entry.get("filename") != "mpas_rrtmg_sw_v842.cu":
            continue
        match = re.match(r"\A(?:#define RSW_[TC]_[^\n]*\n)+", entry["source"])
        if match is not None and "#define RSW_T_EXP_TBL " in match[0]:
            return match[0]
    raise ValueError("the MPAS shortwave catalog has no certified raw-table layout")


def shortwave_source():
    """Assemble today's SW implementation, exactly as MPASShortwave does.

    The old recorded recipe missed the three aerosol passes added to the
    live module, so a cold new-mesh forecast compiled SW for 28 s on its
    foreground thread despite the parallel catalog. Bind source at use time
    as the existing LW/Noah/Thompson inventory does.
    """
    import woof.core.rrtmg_sw as legacy
    from woof.hex.mpasphys import rrtmg_sw_v842 as sw

    kernels = Path(legacy.__file__).parent / "kernels"
    return (_shortwave_table_defines() + (kernels / "glibc_flt32.cuh").read_text()
            + (kernels / "glibc_trig_flt32.cuh").read_text() + sw.CUDA_SOURCE)


#: Layers of the RRFS init door's L60 vertical grid (the hrrrv5 deck's nVertLevels).
TEMPO_RRFS_LEVELS = 59


def pack_inventory(package_root=None, compiler_family=None, *, compiler=None):
    """Every row the shipped pack holds: each declared module under strict AND fast numerics.

    THE BREAKAGE THIS PREVENTS: the pack selects on the exact options, so a pack of strict rows
    alone misses every hex kernel of a default (fast) run, and a fresh box JITs them all on one
    core before step 1. Rows both modes share (recorded CuPy recipes) are held once."""
    rows = {}
    for mode in ("strict", "fast"):
        for row in inventory(package_root, numerics=mode, compiler_family=compiler_family, compiler=compiler):
            # One image per selection: a declared physics row and its recorded
            # twin differ only in bookkeeping keys and must not compile twice.
            rows.setdefault(selection(row), row)
    return list(rows.values())


def cached_program_filename(row, options, target):
    """Reproduce CuPy's program name, including headers used only in its key.

    CUB embeds assertion filenames in its images. A placeholder filename
    produces different bytes despite identical source and arithmetic flags.
    """
    from woof.hex.kernel_plan import _cupy_entry, _cupy_extra_sources
    metadata = row.get("compile_request") or {}
    expected = metadata.get("extra_source")
    extra = None
    if expected is not None:
        candidates = [value for value in _cupy_extra_sources()
                      if value is not None and "sha256:" + source_hash(value) == expected]
        if not candidates:
            raise ValueError("compiler recipe headers differ from the installed CuPy headers")
        extra = candidates[0]
    return _cupy_entry(dict(source=row["source"], options=options,
                            name_expressions=metadata.get("name_expressions")), target, extra) + ".cu"


def compile_entry(row, arch):
    """Use CuPy's exact RawModule NVRTC route with an explicit Blackwell SM.

    CuPy chooses a cubin for both certified SMs. The driver is unnecessary for
    this step. --ftz and execution-space options match _compile_with_cache_cuda.
    """
    from cupy.cuda import compiler
    from cupy_backends.cuda.libs import nvrtc
    from .compile_contract import NvrtcCompileCapture
    target = arch.removeprefix("sm_")
    if compiler._use_ptx or int(target) not in nvrtc.getSupportedArchs():
        raise ValueError(f"{arch}: compiler cannot reproduce the default cubin route")
    options = tuple(local_options(row["options"]))
    raw = row.get("compiler_route") == "raw-nvrtc"
    if row.get("compiler_route") not in ("direct-nvrtc", "raw-nvrtc"):
        options += ("-ftz=true",)
    if not raw and compiler._get_bool_env_variable("CUPY_CUDA_COMPILE_WITH_DEBUG", False):
        options += ("--device-debug", "--generate-line-info")
    # CuPy's target selector is decorated with per-device memoization. The
    # decorator queries a GPU even for an explicit target. Supply the explicit
    # admitted target during this CPU-only build, keeping the actual compiler
    # entry and all option processing intact. Never install this in a forecast.
    selector = compiler._get_arch_for_options_for_nvrtc
    def explicit(selected=None):
        if str(selected) != target:
            raise ValueError("AOT compiler changed the explicitly requested SM")
        return "-arch=" + arch, "cubin"
    raw_evidence = None
    with NvrtcCompileCapture() as capture, preserve_final_options():
        compiler._get_arch_for_options_for_nvrtc = explicit
        try:
            # The direct route keeps the unit's own file name, as woof passes it.
            filename = (row.get("compile_request") or {}).get("filename") or "hex-aot.cu"
            if row.get("compile_request") and row.get("compiler_route") not in ("direct-nvrtc", "raw-nvrtc"):
                filename = cached_program_filename(row, options, target)
            if raw:
                program = nvrtc.createProgram(row["source"], filename, (), ())
                try:
                    raw_options = options + ("--gpu-architecture=" + arch,)
                    nvrtc.compileProgram(program, raw_options)
                    blob, mapping = nvrtc.getCUBIN(program), None
                    from .compile_contract import NvrtcObservation, _object_record
                    raw_evidence = {
                        "status": "resolved", "method": "raw nvrtc.compileProgram with explicit options",
                        "observations": [NvrtcObservation(row["source_sha256"], raw_options, 0,
                                                           _object_record(blob)).as_dict()]}
                except Exception as error:
                    raise RuntimeError(nvrtc.getProgramLog(program)) from error
                finally:
                    nvrtc.destroyProgram(program)
            else:
                blob, mapping = compiler._compile_using_nvrtc_no_warning(
                    row["source"], options, target, filename,
                    name_expressions=row.get("compile_request", {}).get("name_expressions"),
                    **({"cache_in_memory": True} if row.get("compile_request", {}).get("cache_in_memory") else {}))
        finally:
            compiler._get_arch_for_options_for_nvrtc = selector
    if not isinstance(blob, bytes) or blob[:4] != b"\x7fELF":
        raise ValueError(f"{row['module_key']} {arch}: expected cubin, received non-ELF compiler output")
    shipped = {key: value for key, value in row.items() if key != "source"}
    return {**shipped, "arch": arch, "cubin_sha256": hashlib.sha256(blob).hexdigest(),
            # Cubins are mostly zero padding and relocation tables: LZMA
            # stores the four architectures' images in about a twentieth.
            "cubin_xz_base64": base64.b64encode(
                lzma.compress(blob, preset=9 | lzma.PRESET_EXTREME)).decode("ascii"),
            "name_mapping": mapping,
            "effective_compile": raw_evidence or capture.evidence(row["source_sha256"])}


def _compile_job(job):
    import sys
    import time
    started = time.perf_counter()
    result = compile_entry(*job)
    print(json.dumps({"module_key": job[0]["module_key"], "arch": job[1],
                      "compile_seconds": time.perf_counter() - started}),
          file=sys.stderr, flush=True)
    return result


def build(output, *, entries=None, architectures=ARCHITECTURES, base_pack=None, workers=None):
    """Build requested images, optionally extending a verified shared pack.

    Existing selections are reused only with the exact compiler, source,
    options and compile request. Unrelated images remain in the shared pack.
    """
    from .aot import Pack
    from woof.hex.kernel_pack import write
    if tuple(architectures) != ARCHITECTURES:
        raise ValueError(f"the shipped pack covers exactly {', '.join(ARCHITECTURES)}")
    if workers is not None and workers < 1:
        raise ValueError("workers must be positive")
    identity = compiler_identity()
    rows = list(pack_inventory(compiler_family=identity["nvrtc"][0], compiler=identity) if entries is None else entries)
    retained = []
    selected = None
    if base_pack is not None:
        selected = Pack(base_pack, identity=identity)
        if selected.reason:
            raise ValueError(f"base pack cannot be reused: {selected.reason}")
        retained = list(selected.rows)
    # One NVRTC compile per (module, architecture), across worker processes: the
    # explicit-target hook in compile_entry is process-global, so each worker
    # is a process running one compile at a time.
    import concurrent.futures
    import multiprocessing
    import os
    jobs = []
    reused = 0
    queued = set()
    for row in rows:
        if selection(row) in queued:
            continue
        queued.add(selection(row))
        for arch in architectures:
            existing = None
            if selected is not None:
                existing, _ = selected.select(row["module_key"], row["source"],
                                              row["options"], arch,
                                              metadata=row.get("compile_request"))
            if existing is None:
                jobs.append((row, arch))
            else:
                if selected.image(existing) is None:
                    raise ValueError(f"base pack has a corrupt image: {row['module_key']} {arch}")
                reused += 1
    context = multiprocessing.get_context("fork")
    compiled = []
    if jobs:
        count = min(len(jobs), workers or max(1, os.cpu_count() or 1))
        with concurrent.futures.ProcessPoolExecutor(count, mp_context=context) as pool:
            compiled = list(pool.map(_compile_job, jobs, chunksize=1))
    from woof.certify.compile_platform import nvrtc_banner, nvrtc_build, nvrtc_build_id
    from .compile_contract import build_id_accepted
    banner = nvrtc_banner()
    if nvrtc_build(banner) == "unavailable" or not build_id_accepted(nvrtc_build_id(banner)):
        raise ValueError("NVRTC build banner could not be measured for the wheel")
    compiled = [{**row, "nvrtc_banner": banner} for row in compiled]
    write(output, identity, [*retained, *compiled])
    return {"modules": len(rows), "entries": len(retained) + len(compiled),
            "compiled": len(compiled), "reused": reused, "retained": len(retained),
            "architectures": list(architectures)}
