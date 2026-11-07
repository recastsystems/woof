"""No CUDA translation unit may divide by a float literal.

THE BREAKAGE THIS PREVENTS, measured 2026-08-26 on the desktop RTX 3080
(sm_86, NVRTC 13.0.48 ``CL-36260728``, driver 13030) and recorded in
``tree/evidence/nvrtc-reciprocal-20260826/``:

NVRTC compiles ``x / <float literal>`` to a real ``div.rn.f32`` for every
target up to and including ``compute_90``, and to ``mul.rn.f32`` by the
literal's float32 reciprocal for every target from ``compute_100`` up --
which covers every card this port runs production work on.  The reciprocal of
12 is not exactly representable, so the multiply is not the correctly-rounded
quotient the CPU authority computes.  On the reference regional bytes that
moved:

===========================  =========  ==========================
kernel                       values     values the rewrite changes
===========================  =========  ==========================
``transport_vertical_flux``    166,376    51,258  (30.81 %)
``vertical_u_flux_f32``        474,032   157,710  (33.27 %)
``theta_vertical_flux_f32``    154,492    51,639  (33.43 %)
``w_vertical_flux_f32``        154,492    51,318  (33.22 %)
===========================  =========  ==========================

The 51,258 figure is the whole of a divergence that stood on the books
unexplained: ``transport_vertical_flux`` against the CPU authority's
``_atmosphere_vertical_flux`` on the reference cull.

The remedy is a divisor the compiler is not allowed to fold -- a
``__constant__`` translation-unit symbol or a runtime kernel argument, both of
which the host can write.  ``mpas_div`` must keep carrying the division so
its FTZ subnormal guard stays on the path; ``__fdiv_rn`` at a call site would
resist the rewrite but drop that guard.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import sys

import pytest
from _layout import PACKAGE_DIR


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TOOLS = ROOT / "tools"
DRIVERS = PACKAGE_DIR / "drivers"
AUDIT_PATH = TOOLS / "audit_nvrtc_reciprocal_rewrite.py"

#: The kernels the rewrite reached, and the divisions each one must keep.
AFFECTED_KERNELS = {
    "hexcore.cuda_transport": ("transport_vertical_flux",),
    "hexcore.cuda_driver": (
        "vertical_u_flux_f32",
        "theta_vertical_flux_f32",
        "w_vertical_flux_f32",
    ),
}
DIVISIONS_PER_AFFECTED_KERNEL = 2

# A PTX virtual register, whatever this NVRTC calls one -- see the note on
# `_REG` in `tools/audit_nvrtc_reciprocal_rewrite.py`. NVRTC 13.0 spells a
# float32 operand `%f`; NVRTC 13.3 at `compute_120` spells the same operand
# `%r`, and this row asserted the divisions were GONE when they were there
# (measured on the proving RTX 5090, 2026-08-27, the libm-identity finding).
#
# The distinction the guard actually rests on survives untouched: a division
# by a LOADED operand ends in a register, a division by an IMMEDIATE ends in
# `0f........`, and `%` never begins a `0f` literal.
_REG = r"%\w+"
_DIV_BY_REGISTER = re.compile(
    rf"^\s*div\.rn(?:\.ftz)?\.f32\s+{_REG},\s*{_REG},\s*{_REG};"
)
_DIV_BY_IMMEDIATE = re.compile(
    rf"^\s*div\.rn(?:\.ftz)?\.f32\s+{_REG},\s*{_REG},\s*0f"
)
_ENTRY = re.compile(r"^\s*(?:\.\w+\s+)*\.entry\s+([A-Za-z_]\w*)")


def _load_audit() -> object:
    name = "_test_audit_nvrtc_reciprocal_rewrite"
    sys.modules.pop(name, None)
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    spec = importlib.util.spec_from_file_location(name, AUDIT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _entry_body(ptx: str, kernel: str) -> list[str]:
    lines = ptx.splitlines()
    for index, line in enumerate(lines):
        match = _ENTRY.match(line)
        if match and match.group(1) == kernel:
            break
    else:  # pragma: no cover - a missing entry fails the caller's assert
        return []
    depth = 0
    body: list[str] = []
    for line in lines[index:]:
        body.append(line)
        depth += line.count("{") - line.count("}")
        if depth == 0 and "}" in line and len(body) > 1:
            break
    return body


def _literal_divisors(source: str) -> list[str]:
    """Every float-literal divisor in one CUDA translation unit.

    Both spellings are exposed: the helper the port uses everywhere, and the
    plain operator, which the compiler rewrites just the same.
    """

    found: list[str] = []
    for start in (match.end() for match in re.finditer(r"mpas_div\(", source)):
        depth = 1
        comma: int | None = None
        index = start
        while index < len(source) and depth:
            character = source[index]
            if character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
            elif character == "," and depth == 1 and comma is None:
                comma = index
            index += 1
        if comma is not None:
            found.append(source[comma + 1: index - 1].strip())
    literal = re.compile(r"-?\d+(\.\d*)?f")
    offenders = [
        divisor for divisor in found if literal.fullmatch(divisor)
    ]
    offenders.extend(
        match.group(0) for match in re.finditer(r"/\s*-?\d+(\.\d*)?f", source)
    )
    return offenders


# ---------------------------------------------------------------------------
# The source guard.  No card, no compiler -- it runs in every battery.
# ---------------------------------------------------------------------------
def test_no_cuda_translation_unit_divides_by_a_float_literal() -> None:
    # The census compiles with CuPy's bundled headers and captures CuPy RawModule requests (ddedd76d).
    pytest.importorskip("cupy")
    audit = _load_audit()
    offenders: dict[str, list[str]] = {}
    units = 0
    for module_key, _anchor, source in audit.translation_units():
        units += 1
        found = _literal_divisors(source)
        if found:
            offenders[module_key] = found
    assert units >= 15, "the census stopped enumerating translation units"
    assert offenders == {}, (
        "a float-literal divisor is compiled to a reciprocal multiply on "
        "every compute_100-or-above target and is then one ulp off the CPU "
        "authority's quotient; use a __constant__ translation-unit symbol or "
        f"a runtime kernel argument instead: {offenders}"
    )


def test_the_ptx_reader_sees_a_division_whatever_the_registers_are_called() -> None:
    """Teeth for the reader itself, in both directions and both spellings.

    THE BREAKAGE THIS PREVENTS, measured 2026-08-27 on the proving RTX 5090 (the libm-identity finding):
    these patterns spelled a float32 register `%f`, NVRTC 13.3 spells it `%r`
    at `compute_120`, and the whole family went blind on that toolchain -- the
    census reported a SILENT GREEN (no rewrites found because none could be
    seen) while the positive row reported a false red against two divisions
    that were measurably present. Only a teeth test caught it, so the teeth
    now cover the register spelling too. This runs anywhere: it reads PTX
    text, it does not compile any.
    """

    for reg in ("%f", "%r"):
        surviving = f"\tdiv.rn.ftz.f32 \t{reg}114, {reg}113, {reg}1;"
        rewritten_div = f"\tdiv.rn.ftz.f32 \t{reg}114, {reg}113, 0f3DAAAAAB;"
        assert _DIV_BY_REGISTER.match(surviving), (
            f"the reader cannot see a surviving division whose registers are "
            f"named {reg}, so on a toolchain that names them that way it "
            f"reports the guard broken when the guard is holding"
        )
        assert not _DIV_BY_REGISTER.match(rewritten_div), (
            f"a division by an IMMEDIATE ({reg} spelling) matched the "
            f"loaded-operand pattern; the guard would then accept a folded "
            f"divisor as if the division had survived"
        )
        assert _DIV_BY_IMMEDIATE.match(rewritten_div)
        assert not _DIV_BY_IMMEDIATE.match(surviving)

    # f64 divisions use a third spelling again and must not be counted as
    # float32 traffic: the opcode, not the register, is what selects.
    assert not _DIV_BY_REGISTER.match("\tdiv.rn.f64 \t%rd34, %rd475, %rd5;")

    audit = _load_audit()
    for reg in ("%f", "%r"):
        assert audit._MUL_CONST.match(
            f"\tmul.rn.ftz.f32 \t{reg}9, {reg}8, 0f3DAAAAAB;"
        ), (
            f"the rewrite census cannot see a reciprocal multiply whose "
            f"registers are named {reg}; on that toolchain it reports zero "
            f"offenders because it is blind, which reads exactly like a pass"
        )
        assert audit._DIV_CONST.match(f"\tdiv.rn.ftz.f32 \t{reg}9, {reg}8, 0f41400000;")
        assert not audit._MUL_CONST.match(f"\tmul.rn.ftz.f32 \t{reg}9, {reg}8, {reg}7;")


def test_the_literal_divisor_scanner_sees_both_spellings() -> None:
    """Teeth for the guard above: it must fail on what it claims to catch."""

    assert _literal_divisors("x = mpas_div(a, 12.0f);") == ["12.0f"]
    assert _literal_divisors("x = a / 12.0f;") == ["/ 12.0f"]
    assert _literal_divisors("x = mpas_div(a, denominator);") == []
    assert _literal_divisors("x = mpas_div(a, mpas_mul(b, 12.0f));") == []


def test_the_pin_table_covers_the_files_that_supply_the_pinned_bytes() -> None:
    """A pinned translation unit does not compile its own file.

    ``cuda_transport_v841`` builds its CUDA source from
    ``cuda_transport._CUDA_SOURCE``, and every unit in the port prepends
    ``cuda_fp32.CUDA_FTZ_HELPERS``.  While those two files were unpinned, an
    edit to either changed the compiled bytes of pinned units with every
    pinned digest still matching -- which is how this lane's own remedy first
    landed with the frozen-source proof reporting green.
    """

    name = "_test_pin_table_for_nvrtc_reciprocal"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(
        name, DRIVERS / "run_cuda_v841_full_physics_x4.py"
    )
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    sys.modules[name] = runner
    spec.loader.exec_module(runner)

    pins = runner.EXECUTION_SOURCE_PINS
    for relative in (
        "src/hexcore/cuda_transport.py",
        "src/hexcore/cuda_fp32.py",
    ):
        assert relative in pins, (
            f"{relative} supplies CUDA bytes to a pinned translation unit "
            "and must be pinned itself, or the pinned digest certifies "
            "source it does not contain"
        )
        assert pins[relative], f"{relative} is pinned to nothing"


# ---------------------------------------------------------------------------
# The compiler guard.  Needs NVRTC, not a card of the affected architecture:
# the rewrite happens while compiling, so a target below the boundary can
# still be asked what the compiler does above it.
# ---------------------------------------------------------------------------
def test_no_translation_unit_is_rewritten_into_a_reciprocal_multiply() -> None:
    import cupy  # noqa: F401  (the gpu-tier gate reads this import)

    audit = _load_audit()
    offenders: dict[str, list[dict[str, object]]] = {}
    uncovered = sorted(
        audit.source_bearing_files()
        - {
            anchor.split("::", 1)[0].removeprefix("src/")
            for _key, anchor, _source in audit.translation_units()
        }
    )
    from concurrent.futures import ThreadPoolExecutor
    # Contributor anchors can share one exact compile request. Keep every
    # anchor in the coverage check above, but compile each source/options pair
    # once. Concurrent lru_cache misses otherwise repeat large translation units.
    units = {}
    for unit in audit.translation_units():
        source = unit[2]
        identity = (str(source), tuple(getattr(source, "options", audit.SHIPPED_OPTIONS)))
        units.setdefault(identity, unit)
    def measure(unit):
        module_key, _anchor, source = unit
        return module_key, audit.rewrite_sites(source)
    # The expanded inventory compiles independent requests on bounded workers.
    with ThreadPoolExecutor(max_workers=8) as pool:
        for module_key, record in pool.map(measure, units.values()):
            if record["count"]:
                offenders[module_key] = record["rewritten_sites"]
    assert uncovered == [], (
        "a CUDA-bearing module is not in the census, so nobody is asking the "
        f"compiler what it does with that unit: {uncovered}"
    )
    assert offenders == {}, (
        f"NVRTC rewrites a division into a reciprocal multiply at "
        f"compute_{audit.ABOVE_BOUNDARY_ARCH} in: {offenders}"
    )


def test_the_census_detects_a_planted_literal_divisor() -> None:
    """Teeth: the compiler guard must fail on the shape it was written for."""

    import cupy  # noqa: F401

    audit = _load_audit()
    from woof.hex.cuda_fp32 import CUDA_FTZ_HELPERS

    planted = CUDA_FTZ_HELPERS + """
extern "C" __global__ void planted(const int n, const float *x, float *o)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    o[i] = mpas_div(x[i], 12.0f);
}
"""
    record = audit.rewrite_sites(planted)
    assert record["count"] == 1, record
    site = record["rewritten_sites"][0]
    assert site["divisor"] == 12.0
    assert site["reciprocal_is_exact"] is False
    assert site["entry"] == "planted"

    # And the control on the control: the same kernel with a divisor the host
    # can write is not rewritten.
    exempt = CUDA_FTZ_HELPERS + """
__constant__ float planted_denominator = 12.0f;

extern "C" __global__ void planted(const int n, const float *x, float *o)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    o[i] = mpas_div(x[i], planted_denominator);
}
"""
    assert audit.rewrite_sites(exempt)["count"] == 0


@pytest.mark.parametrize("arch", ["86", "120"])
def test_the_affected_kernels_still_divide_on_every_target(arch: str) -> None:
    """The positive form: the division survives, it was not merely not-found.

    A census that reports zero because it stopped looking would pass the test
    above.  This one requires the two ``div.rn.f32`` instructions per kernel
    to be present, against a loaded operand rather than an immediate, on a
    target either side of the rewrite boundary.
    """

    import cupy  # noqa: F401
    import importlib

    audit = _load_audit()
    for module_name, kernels in AFFECTED_KERNELS.items():
        module = importlib.import_module(module_name)
        ptx = audit.compile_ptx(module._CUDA_SOURCE, arch)
        for kernel in kernels:
            body = _entry_body(ptx, kernel)
            assert body, f"{kernel} is not an entry point of {module_name}"
            by_register = sum(bool(_DIV_BY_REGISTER.match(l)) for l in body)
            by_immediate = sum(bool(_DIV_BY_IMMEDIATE.match(l)) for l in body)
            assert by_register == DIVISIONS_PER_AFFECTED_KERNEL, (
                f"compute_{arch} {module_name}::{kernel} emits {by_register} "
                f"float32 divisions by a loaded operand, expected "
                f"{DIVISIONS_PER_AFFECTED_KERNEL}"
            )
            assert by_immediate == 0, (
                f"compute_{arch} {module_name}::{kernel} divides by an "
                "immediate, which is the shape the rewrite consumes"
            )


def test_census_does_not_pair_an_unchanged_division_with_an_unrelated_multiply():
    """A rewrite must lose a division and gain its reciprocal in the same entry."""
    import cupy  # noqa: F401
    audit = _load_audit()
    source = r'''
extern "C" __global__ void unchanged(const float *x, float *out) {
    float quotient;
    asm("div.rn.f32 %0, %1, 0f41400000;" : "=f"(quotient) : "f"(x[0]));
    out[0] = quotient;
    out[1] = x[1] * 0.0833333358168602f;
}
'''
    record = audit.rewrite_sites(source)
    assert record['div_by_constant_below_boundary'] == 1
    assert record['count'] == 0
