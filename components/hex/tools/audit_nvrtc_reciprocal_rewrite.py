#!/usr/bin/env python3
"""Census every CUDA translation unit NVRTC rewrites into a reciprocal multiply.

A grep for ``12.0f`` finds what a human already suspected.  This audit asks the
compiler instead: each translation unit is compiled to PTX twice with identical
options -- once for a target below the rewrite boundary and once for a target
above it -- and the two images are compared instruction for instruction.  A site
appears in the census only because the compiler actually emitted
``mul.rn[.ftz].f32`` against the reciprocal of a constant where the lower target
emitted ``div.rn[.ftz].f32`` against that constant.

That makes the audit arbitrary in the sense the house requires: a new
translation unit, a new literal, or a new NVRTC build needs no table entry
here.  It also runs anywhere NVRTC exists -- no card of the affected
architecture is required, because the rewrite happens in the compiler.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import datetime as _dt
import hashlib
from functools import lru_cache
import json
import re
import struct
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

# The options the shipped route requests, plus the terminal flag CuPy's
# RawModule appends and the execution-space flag CuPy adds on CUDA 12+.
SHIPPED_OPTIONS = (
    "--std=c++17",
    "--fmad=false",
    "-ftz=true",
    "--device-as-default-execution-space",
)
BELOW_BOUNDARY_ARCH = "86"
ABOVE_BOUNDARY_ARCH = "120"

# A PTX virtual register, whatever this NVRTC calls one.
#
# THE BREAKAGE THIS PREVENTS, measured 2026-08-27 on the proving RTX 5090 (the libm-identity finding).
# These patterns used to spell the register `%f\d+`, which is what NVRTC
# 13.0 emits for a float32 operand.  **NVRTC 13.3 targeting `compute_120`
# names the same registers `%r`**, so on that toolchain the census matched
# nothing and went blind in BOTH directions at once:
#
#   * `test_no_translation_unit_is_rewritten_into_a_reciprocal_multiply`
#     reported GREEN because `rewrite_sites` found zero rewritten sites --
#     a silent green, the worst failure this program has a name for;
#   * `test_the_affected_kernels_still_divide_on_every_target[120]` reported
#     RED for divisions that were, measured, right there in the PTX
#     (`div.rn.ftz.f32 %r333, %r24, %r2` -- two of them, exactly as pinned).
#
# Only the teeth test caught it. The opcode still carries `.f32`, so
# widening the REGISTER spelling widens what this can SEE and changes
# nothing about what it REQUIRES: an immediate operand is `0f........`,
# which does not begin with `%`, so a rewritten multiply can never be
# mistaken for a surviving division.
_REG = r"%\w+"
_DIV_CONST = re.compile(
    rf"^\s*div\.rn(?:\.ftz)?\.f32\s+({_REG}),\s*({_REG}),\s*0f([0-9A-Fa-f]{{8}});"
)
_MUL_CONST = re.compile(
    rf"^\s*mul\.rn(?:\.ftz)?\.f32\s+({_REG}),\s*({_REG}),\s*0f([0-9A-Fa-f]{{8}});"
)
_FUNC = re.compile(r"^\s*(?:\.\w+\s+)*\.entry\s+([A-Za-z_][A-Za-z0-9_]*)")


def _f32_from_hex(text: str) -> float:
    return float(struct.unpack(">f", bytes.fromhex(text))[0])


def _hex_from_f32(value: float) -> str:
    return struct.pack(">f", np.float32(value)).hex().upper()


def compile_ptx(source: str, arch: str) -> str:
    """PTX for ``source`` at ``arch`` under the shipped option set."""

    return _compile_ptx_cached(str(source), arch, tuple(getattr(source, "options", SHIPPED_OPTIONS)))


@lru_cache(maxsize=1024)
def _compile_ptx_cached(source, arch, options):
    from cupy.cuda import nvrtc
    import cupy
    include_dirs = [Path(cupy.__file__).parent / "_core/include"]
    # CuPy's own bundled CCCL (cupy/_core/include/cupy/_cccl/{libcudacxx,thrust,cub}), which CuPy's compiler
    # puts on the include path itself. THE BREAKAGE: on a pip-only CUDA install (no toolkit, so no
    # cuda_path/include/cccl) every unit including cupy/complex.cuh failed with "cannot open source file
    # thrust/detail/config/namespace.h" (hex 0.5.0 sealing pass, release box, seal-gpu-nvrtc-audit).
    bundled = include_dirs[0] / "cupy/_cccl"
    include_dirs.extend(bundled / name for name in ("libcudacxx", "thrust", "cub"))
    cuda_path = cupy.cuda.get_cuda_path()
    if cuda_path:
        include_dirs.extend((Path(cuda_path) / "include", Path(cuda_path) / "include/cccl"))
    options = (*options, *("-I" + str(path) for path in include_dirs if path.is_dir()))

    program = nvrtc.createProgram(source, "unit.cu", [], [])
    try:
        nvrtc.compileProgram(
            program,
            [*options, f"--gpu-architecture=compute_{arch}"],
        )
        image = nvrtc.getPTX(program)
    except Exception as error:
        raise RuntimeError(nvrtc.getProgramLog(program)) from error
    finally:
        nvrtc.destroyProgram(program)
    if isinstance(image, (bytes, bytearray)):
        return bytes(image).decode("utf-8", errors="replace")
    return str(image)


def _current_entry(lines: list[str], index: int) -> str:
    for back in range(index, -1, -1):
        match = _FUNC.match(lines[back])
        if match:
            return match.group(1)
    return "<file scope>"


def rewrite_sites(source: str) -> dict[str, Any]:
    """Sites where the higher target multiplies where the lower one divides."""

    low = compile_ptx(source, BELOW_BOUNDARY_ARCH)
    high = compile_ptx(source, ABOVE_BOUNDARY_ARCH)
    low_lines = low.splitlines()
    high_lines = high.splitlines()

    divisors_below: dict[str, int] = {}
    for line in low_lines:
        match = _DIV_CONST.match(line)
        if match:
            key = match.group(3).upper()
            divisors_below[key] = divisors_below.get(key, 0) + 1

    def opcode_counts(lines, pattern):
        counts = Counter()
        entry = "<file scope>"
        labels = []
        for line in lines:
            match = _FUNC.match(line)
            if match:
                entry = match.group(1)
            labels.append(entry)
            match = pattern.match(line)
            if match:
                counts[entry, match.group(3).upper()] += 1
        return counts, labels
    low_div, _ = opcode_counts(low_lines, _DIV_CONST)
    high_div, labels = opcode_counts(high_lines, _DIV_CONST)
    low_mul, _ = opcode_counts(low_lines, _MUL_CONST)
    high_mul, _ = opcode_counts(high_lines, _MUL_CONST)
    lost_divisions = low_div - high_div
    added_multiplications = high_mul - low_mul
    sites: list[dict[str, Any]] = []
    for index, line in enumerate(high_lines):
        match = _MUL_CONST.match(line)
        if not match:
            continue
        immediate = match.group(3).upper()
        multiplier = _f32_from_hex(immediate)
        if multiplier == 0.0:
            continue
        implied = np.float32(1.0) / np.float32(multiplier)
        implied_hex = _hex_from_f32(implied)
        entry = labels[index]
        if not lost_divisions[entry, implied_hex] or not added_multiplications[entry, immediate]:
            continue
        # Only count it when the reciprocal round-trips: mul by 1/d where the
        # lower target divides by exactly d.
        if _hex_from_f32(np.float32(1.0) / implied) != immediate:
            continue
        lost_divisions[entry, implied_hex] -= 1
        added_multiplications[entry, immediate] -= 1
        sites.append({
            "entry": entry,
            "divisor": float(implied),
            "divisor_bits": f"0x{implied_hex}",
            "reciprocal_bits": f"0x{immediate}",
            "reciprocal_is_exact": bool(
                np.float64(1.0) / np.float64(implied)
                == np.float64(np.float32(multiplier))
            ),
            "ptx_below_boundary": (
                f"div.rn.ftz.f32 ..., 0f{implied_hex}"
            ),
            "ptx_above_boundary": high_lines[index].strip(),
        })

    return {
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "ptx_sha256": {
            f"compute_{BELOW_BOUNDARY_ARCH}": hashlib.sha256(
                low.encode("utf-8")
            ).hexdigest(),
            f"compute_{ABOVE_BOUNDARY_ARCH}": hashlib.sha256(
                high.encode("utf-8")
            ).hexdigest(),
        },
        "div_by_constant_below_boundary": sum(divisors_below.values()),
        "rewritten_sites": sites,
        "count": len(sites),
    }


def core_translation_units() -> Iterator[tuple[str, str, str]]:
    """Yield ``(module_key, python_anchor, source)`` for every unit."""

    import importlib

    import woof.hex

    def attribute(module_name: str, attribute_name: str) -> Any:
        # Imported under the package's own import name, which a tree that
        # folds the package in may change.  The module_key strings below are
        # identities and keep their bytes either way.
        module = importlib.import_module(f"{woof.hex.__name__}.{module_name}")
        return getattr(module, attribute_name)

    simple = (
        ("hexcore.cuda_acoustic_fusion_v841", "cuda_acoustic_fusion_v841", "CUDA_FUSED_SOURCE"),
        ("hexcore.cuda_acoustic", "cuda_acoustic", "_CUDA_SOURCE"),
        ("hexcore.cuda_acoustic_v841", "cuda_acoustic_v841", "_CUDA_SOURCE"),
        ("hexcore.cuda_horizontal", "cuda_horizontal", "_CUDA_SOURCE"),
        ("hexcore.partition_overlap_v841", "partition_overlap_v841", "_CUDA_SOURCE"),
        (
            "hexcore.cuda_horizontal_v841",
            "cuda_horizontal_v841",
            "_CUDA_SOURCE",
        ),
        ("hexcore.cuda_transport", "cuda_transport", "_CUDA_SOURCE"),
        ("hexcore.cuda_transport_v841", "cuda_transport_v841", "_CUDA_SOURCE"),
        ("hexcore.cuda_dynamics_v841", "cuda_dynamics_v841", "_CUDA_SOURCE"),
        ("hexcore.cuda_driver", "cuda_driver", "_CUDA_SOURCE"),
        (
            "hexcore.cuda_driver.physics_v841",
            "cuda_driver",
            "CUDA_V841_PHYSICS_DRIVER_SOURCE",
        ),
        ("hexcore.cuda_gwdo_v841", "cuda_gwdo_v841", "_CUDA_SOURCE"),
        ("hexcore.cuda_physics_v841", "cuda_physics_v841", "_CUDA_SOURCE"),
        (
            "hexcore.cuda_physics_prep_v841",
            "cuda_physics_prep_v841",
            "_CUDA_SOURCE",
        ),
        ("hexcore.cuda_regional_v841", "cuda_regional_v841", "CUDA_REGIONAL_SOURCE"),
        (
            "hexcore.cuda_solve_region_v841",
            "cuda_solve_region_v841",
            "SOLVE_REGION_CUDA_SOURCE",
        ),
        # Arrived with 0.3.4 and left out of the census until the GPU tier's
        # coverage control named them: the native upper damping (CAM and
        # Rayleigh) and the same-host halo byte copy.
        ("hexcore.cuda_upper_damping_v841", "cuda_upper_damping_v841", "_CUDA_SOURCE"),
        (
            "hexcore.partition_samehost_transport_v841",
            "partition_samehost_transport_v841",
            "_BYTE_COPY",
        ),
    )
    for module_key, module_name, attribute_name in simple:
        yield (
            module_key,
            f"src/hexcore/{module_name}.py::{attribute_name}",
            attribute(module_name, attribute_name),
        )

    # Built lazily rather than held as a module constant.
    from woof.hex import cuda_acoustic_lts

    yield (
        "hexcore.cuda_acoustic_lts",
        "src/hexcore/cuda_acoustic_lts.py::_build_source()",
        cuda_acoustic_lts._build_source(),
    )

    from woof.hex.cuda_backend import arch_admission
    from woof.hex.cuda_fp32 import CUDA_FTZ_HELPERS

    # The numeric-route probe 0.3.4 compiles on the card in hand before it
    # admits an architecture, composed exactly as the probe composes it.
    yield (
        "hexcore.cuda_backend.arch_admission",
        "src/hexcore/cuda_backend/arch_admission.py::CUDA_FTZ_HELPERS + _PROBE_KERNEL",
        CUDA_FTZ_HELPERS + arch_admission._PROBE_KERNEL,
    )

    from woof.hex.cuda_backend import recovery

    yield (
        "hexcore.cuda_backend.recovery",
        "src/hexcore/cuda_backend/recovery.py::RECOVERY_CUDA_SOURCE",
        recovery.RECOVERY_CUDA_SOURCE,
    )


class CompilerSource(str):
    """Source bytes with the production request's compiler options."""
    def __new__(cls, source, options):
        obj = str.__new__(cls, source)
        obj.options = tuple(options)
        return obj


def translation_units() -> Iterator[tuple[str, str, str]]:
    """Enumerate production declarations and the concrete compiler recipes.

    Source contributors are authenticated by literal kernel entry definitions
    in the assembled request. A contributor without a request remains uncovered.
    """
    import importlib
    from woof.hex.cuda_backend.aot_build import inventory
    from woof.hex.cuda_backend.aot import local_options
    from woof.core import kernels
    import cupy
    header = Path(kernels.__file__).parent
    units = list(core_translation_units())
    for row in inventory(catalogs=True, compiler_family=13, numerics="strict"):
        units.append((row["module_key"], "compiler-recipe::" + row["source_sha256"],
                      CompilerSource(row["source"], (*local_options(row.get("options", ())), "-I" + str(header)))))
    def module(name):
        return importlib.import_module("hexcore." + name)
    def add(name, source, options=("--std=c++17", "--fmad=false", "--ftz=false")):
        units.append(("hexcore." + name, "src/hexcore/" + name.replace(".", "/") + ".py::production",
                      CompilerSource(source, options)))
    for name, attr in (("cuda_vertical_mixing_rrfs", "_SOURCE"),
                       ("cuda_regional_rrfs", "_CUDA_SOURCE"), ("iau_v842", "_CUDA_SOURCE"),
                       ("cuda_courant_fullstep_lts", "CUDA_COLUMN_TRANSPORT"),
                       ("mpasdiag.cloud_v842", "CUDA_SOURCE"),
                       ("mpasdiag.pv_v842", "CUDA_SOURCE"), ("mpasphys.precip_window_v842", "SOURCE"),
                       ("mpasphys.rrfs.glue_plan", "_SOURCE")):
        m = module(name);add(name, getattr(m, attr), getattr(m, "COMPILE_OPTIONS", getattr(m, "OPTIONS", SHIPPED_OPTIONS)))
    for name, source in (("mixing_v842_les", module("mixing_v842_les")._cuda_source()),
                         ("mpasphys.kain_fritsch_v842", module("mpasphys.kain_fritsch_v842").shared_math_source() + module("mpasphys.kain_fritsch_v842").CUDA_SOURCE),
                         ("mpasphys.tiedtke_v842", module("mpasphys.tiedtke_v842").cuda_source()),
                         ("mpasphys.oml_v842", (header / "glibc_trig_flt32.cuh").read_text() + module("mpasphys.oml_v842").CUDA_SOURCE),
                         ("mpasphys.rrfs.myj", module("mpasphys.rrfs.myj").cuda_source(60)),
                         ("mpasphys.rrfs.sf_mynn_rrfs", module("mpasphys.rrfs.sf_mynn_rrfs").cuda_source())):
        add(name, source)
    m = module("mpasdiag.soundings_v842");add("mpasdiag.soundings_v842", m.shared_device_math() + m.CUDA_SOURCE)
    # Both compile through cp.RawKernel with no options of their own, so CuPy's route adds only
    # its -ftz=true: the hourly-max UH layers (uh-layers fold) and the HIP admission probe (AMD
    # fold), whose source is an inline literal in _require_hip and is read here from the file.
    add("uh_layers", module("uh_layers")._SOURCE, ("-ftz=true",))
    for source in inline_kernel_literals("woof/hex/cuda_backend/runtime.py"):
        add("cuda_backend.runtime", source, ("-ftz=true",))
    m = module("mpasphys.cam_rad_v842");add("mpasphys.cam_rad_v842", m.CUDA_SOURCE, (*m.COMPILE_OPTIONS, "-I" + str(header)))
    m = module("mpasphys.wsm6_refl_start_v842");add("mpasphys.wsm6_refl_start_v842", m.source(m.device_math_header(), 60, numerics="strict"))
    # Wave feedback (waves fold): wave_surface rewrites both MYNN surface-layer sources; each variant is the exact
    # request its module compiles when a wave binding supplies the roughness.
    numerics = importlib.import_module("woof.hex.numerics")
    m = module("mpasphys.sf_mynn_v842");add("wave_surface", m._cuda_source(waves=True), numerics.options(m.CUDA_OPTIONS))
    m = module("mpasphys.rrfs.sf_mynnsfc_v600");add("wave_surface", m.cuda_source(waves=True), numerics.options(
        ("--std=c++17", "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true")))
    # These builders do no device work before constructing their compile request.
    # Capture that request, then submit its exact source/options to NVRTC below.
    from unittest.mock import patch
    captured = []
    class Request:
        def __init__(self, code=None, name=None, *, options=(), **kwargs):
            captured.append((code, options))
        def get_function(self, name):
            return self
    builders = {
        "cuda_backend.containers": [("_finite_word_kernel", (size,)) for size in (4, 8)],
        "moving_balance_cuda": [("_kernel", (60,))],
        "moving_vertical_cuda": [(name, ()) for name in ("_kernel", "_interface_kernel", "_layer_kernel", "_rate_kernels", "_profile_kernels")],
        "moving_surface_cuda": [("_kernel", ())], "moving_math_cuda": [("kernels", ())],
        "moving_advection_cuda": [("_kernel", (dtype, 8)) for dtype in ("float32", "float64")],
        "moving_horizontal_cuda": [("_kernel", ())],
        "moving_velocity_cuda": [("_kernel", (dtype,)) for dtype in ("float32", "float64")],
        "moving_momentum_cuda": [("_kernels", ())],
        "moving_saturation_cuda": [("_kernel", ()), ("_adjust_kernel", ())],
        "moving_pool_projection_cuda": [("_kernel", ())],
        "mpasphys.moving_rates_batch_v842": [("_pack_kernel", (size,)) for size in (4, 8)],
        "mpasphys.ocean_pwp_v842": [("_kernel", ())],  # the PWP ocean mixed layer (PWP fold)
    }
    with patch("cupy.RawModule", Request), patch("cupy.RawKernel", Request):
        for name, calls in builders.items():
            for function, args in calls:
                captured.clear()
                build = getattr(module(name), function)
                if hasattr(build, "cache_clear"):
                    build.cache_clear()
                build(*args)
                for source, options in captured:
                    add(name, source, options)
    definitions = re.compile(r'extern\s+"C"\s+__global__\s+(?:__launch_bounds__\([^)]*\)\s*)?void\s+(\w+)')
    contributors = {}
    for relative in source_bearing_files():
        names = set(definitions.findall((_REPO_SRC / relative).read_text()))
        contributors[relative] = names
    emitted = set()
    for key, anchor, source in units:
        digest = hashlib.sha256(source.encode()).hexdigest()
        entries = set(definitions.findall(source))
        for relative, names in contributors.items():
            direct = anchor.split("::", 1)[0] == "src/" + relative
            phased = (relative == "woof/hex/mpasphys/rrfs/nssl_phases.py" and
                      "struct NsslPhaseState" in source and "nssl_phase_body<" in source)
            if direct or phased or names & entries:
                identity = (digest, relative)
                if identity in emitted:
                    continue
                emitted.add(identity)
                yield key, "src/" + relative + "::assembled", source


def inline_kernel_literals(relative: str) -> list[str]:
    """Kernel sources written as an inline string literal, with no module name to import.

    Read from the file's syntax tree, so the module's own bytes stay as they are.
    """
    definition = re.compile(r'extern +"C" +__global__ +void +')
    tree = ast.parse((_REPO_SRC / relative).read_text(encoding="utf-8"))
    found = [node.value for node in ast.walk(tree)
             if isinstance(node, ast.Constant) and isinstance(node.value, str)
             and definition.search(node.value)]
    assert found, f"{relative} carries no inline kernel literal"
    return found


def source_bearing_files() -> set[str]:
    """Every module under ``src`` that carries CUDA source text.

    The census enumerates translation units by name.  This walk is the control
    on that list: a new CUDA-bearing module that nobody added to
    :func:`translation_units` shows up here as uncovered rather than being
    quietly left out of the audit.
    """

    # A literal entry-point definition, not the escaped form a regex uses to
    # scan someone else's source -- ``cuda_ftz`` audits the other modules'
    # translation units and defines none of its own.
    definition = re.compile(r'extern +"C" +__global__ +void +')
    found: set[str] = set()
    for path in sorted(_REPO_SRC.rglob("*.py")):
        if path.name == "cuda_fp32.py":
            continue  # the shared prelude; it defines no kernel of its own
        text = path.read_text(encoding="utf-8", errors="replace")
        if definition.search(text):
            found.add(path.relative_to(_REPO_SRC).as_posix())
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--expect-clean", action="store_true",
        help="exit non-zero when any translation unit still has a rewrite site",
    )
    args = parser.parse_args(argv)

    from cupy.cuda import nvrtc, runtime

    from concurrent.futures import ThreadPoolExecutor
    requests = {}
    covered: set[str] = set()
    for module_key, anchor, source in translation_units():
        covered.add(anchor.split("::", 1)[0].removeprefix("src/"))
        identity = (str(source), tuple(getattr(source, "options", SHIPPED_OPTIONS)))
        requests.setdefault(identity, (module_key, anchor, source))
    def measure(item):
        key, anchor, source = item
        record = rewrite_sites(source)
        record["python_anchor"] = anchor
        record["options"] = list(getattr(source, "options", SHIPPED_OPTIONS))
        return key + "::" + record["source_sha256"], record
    # Compiler work stays on bounded CPU workers, including cold audits.
    with ThreadPoolExecutor(max_workers=8) as pool:
        units = dict(pool.map(measure, requests.values()))
    total = sum(record["count"] for record in units.values())

    uncovered = sorted(source_bearing_files() - covered)

    document = {
        "schema": "mpas-port.nvrtc-reciprocal-census/v1",
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "method": (
            "each translation unit compiled to PTX twice with identical "
            f"options, for compute_{BELOW_BOUNDARY_ARCH} and "
            f"compute_{ABOVE_BOUNDARY_ARCH}; a site is counted only where the "
            "higher target emits mul.rn.f32 by the reciprocal of a constant "
            "the lower target divides by"
        ),
        "compiler": {
            "nvrtc_get_version": list(nvrtc.getVersion()),
            "cuda_runtime_version": int(runtime.runtimeGetVersion()),
            "cuda_driver_version": int(runtime.driverGetVersion()),
            "options": list(SHIPPED_OPTIONS),
            "host": __import__("hexcore.host_identity", fromlist=["host_record"]).host_record(),
        },
        "total_rewritten_sites": total,
        "cuda_bearing_modules_not_censused": uncovered,
        "translation_units": units,
    }
    text = json.dumps(document, indent=1)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    if args.expect_clean and (total or uncovered):
        print(
            f"AUDIT FAILED: {total} rewrite sites remain; "
            f"uncensused CUDA-bearing modules {uncovered}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
