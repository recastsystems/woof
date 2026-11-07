"""A146 gate, compiler layer: no float division NVRTC rewrites on Blackwell.

The breakage this prevents: NVRTC's NVVM, for compute_100 and compute_120
(every Blackwell card: RTX 5090, 5070 Ti, PRO 4500, PRO 6000) under
``-ftz=true`` (which CuPy appends to every ``RawModule``), compiles
``x / C`` for a compile-time constant ``C`` as ``x * RN(1/C)``.  That is not
IEEE division: on an RTX 5070 Ti with NVRTC 13.4, exhaustively over every
float in [1, 2), ``x / 3.0f`` is one ULP off for 33 percent of inputs,
``x / 60.0f`` (the advection flux) for 65 percent, ``x / 9.81f`` for 0.08
percent, while compute_90 and older keep ``div.rn`` and round correctly.
NVRTC 12.9, 13.0, 13.3 and 13.4 all do it; ``-prec-div=true``,
``-fmad=false`` and ``-Xptxas -O0`` do not stop it.  ``__fdiv_rn(x, C)`` is
never rewritten and is the IEEE quotient on every card, with the module's
own FTZ mode, so it is the one spelling a kernel may use for such a
division.  A power-of-two, zero or infinite divisor has an exact reciprocal
and either spelling is IEEE.

Two layers, because a source scan cannot see every such division (a
divisor can arrive through a local constant, an integer literal or a helper
inlined with a literal argument) and the compiler census needs NVRTC:

* the source scan (tests/test_literal_division_source_gate.py, no CuPy)
  runs on every leg and names the file, line and divisor;
* this file, on the GPU shard, compiles every production -ftz=true
  translation unit.  For compute_89, where a constant division is still a
  ``div.rn``, it requires no ``div.rn`` with an immediate divisor.  Then it
  compares compute_120 under ``-ftz=false`` (where a constant division stays
  a ``div.rn``) with compute_120 under ``-ftz=true`` line by line and
  requires no line that loses a ``div.rn`` under ``-ftz=true`` while its
  multiplies gain an immediate with an inexact reciprocal: that catches a
  divisor compute_120 alone constant-folds (``tgammaf(4.0f)`` under NVRTC
  13.4.92) and constant divisions compute_89 merges into one ``div.rn`` of
  a selected register, which the first census could not see.  A line whose
  divisions are all ``__fdiv_rn`` is not listed: compute_120 may fold such
  a call on constants, and the folded value is the IEEE quotient.

The line comparison's reference is the same architecture (A174).  It was
compute_89, and NVRTC 12.9.86 (the default ``recast-woof[gpu]`` compiler) compiles
``powf`` and ``logf`` differently for the two architectures (noah's as
separate functions on compute_120, inlined on compute_89), so per-line counts
moved on seven lines of noah, nssl2_nucond, shinhong and thompson that hold
no constant division, and this file failed twice on every host under the
default install.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pytest

#: The tree under test: the census reads every source from it (A193), never
#: from wherever woof happens to be imported.
REPO_ROOT = Path(__file__).resolve().parents[1]


def _nvrtc():
    try:
        from cupy.cuda import nvrtc
        nvrtc.getVersion()
    except Exception as exc:  # pragma: no cover - host without CuPy's NVRTC
        pytest.skip(f"CuPy's NVRTC is not importable here ({exc}); the "
                    "compiler census needs it, no GPU")
    return nvrtc


def test_no_production_unit_divides_by_a_compile_time_constant():
    """A146, compiler layer: compile every production -ftz=true unit for
    compute_89 and require no ``div.rn`` with a non-exact immediate divisor.
    Catches divisors that arrive through locals, integer literals and
    inlined helpers, which the source scan cannot see."""
    _nvrtc()
    from tools.literal_division_census import census_tree
    offenders, compiled = [], 0
    # Every unit compiles with the flags the serial loop used, one unit per
    # worker process, and the rows come back in production_units order, so
    # the count and the listing are the serial loop's.  A unit NVRTC
    # refuses alone (row None) is a fragment compiled through another unit
    # (p3 via p3_device) or an --ftz=false RRTMG source; the census covers
    # the fragments through their real units.
    for key, row in census_tree(REPO_ROOT, "immediate"):
        if row is None:
            continue
        compiled += 1
        for site in row["constant_divisor_sites"]:
            offenders.append(f"{key}: {site['file']}:{site['line']} "
                             f"divides by {site['divisor']}")
    assert compiled >= 90, f"only {compiled} units compiled"
    assert not offenders, "A146:\n  " + "\n  ".join(offenders)


def test_the_census_sees_a_planted_constant_division():
    """The compiler layer fires on the three spellings a source scan misses."""
    _nvrtc()
    from tools.literal_division_census import Unit, census_unit
    src = """
__device__ __forceinline__ float helper(float a, float b) { return a / b; }
extern "C" __global__ void k(const float* x, float* y) {
    int i = threadIdx.x;
    float d = 3.0f;
    y[i] = x[i] / d + helper(x[i], 60.0f) + x[i] / 7 + __fdiv_rn(x[i], 9.81f)
         + x[i] / 4.0f;
}
"""
    row = census_unit(Unit("planted", src, ("-std=c++17",), []))
    divisors = sorted(float(s["divisor"]) for s in row["constant_divisor_sites"])
    assert divisors == [3.0, 7.0, 60.0], divisors


def test_no_production_line_becomes_a_reciprocal_multiply_on_compute_120():
    """A146, compiler layer, second pass: no source line of a production
    unit whose ``div.rn`` on compute_120 under ``-ftz=false`` is a multiply
    by an immediate under ``-ftz=true``.  The review of 4810400f1 found four
    such lines the compute_89 census passed (nssl2_fused_gs.cu's
    ``/ tgammaf(4.0f)`` and ``/ tgammaf(5.0f)``, thompson_aerosol_warm.cu's
    two lamc clamps).  The immediate census catches a division by 0 or by a
    power of two too, and exempts it, since its reciprocal is exact; this
    pass does the same."""
    _nvrtc()
    from tools.literal_division_census import census_tree
    offenders, compiled = [], 0
    for key, row in census_tree(REPO_ROOT, "rewrites"):
        if row is None:
            continue      # a fragment, compiled through its real unit
        compiled += 1
        for site in row["rewritten_sites"]:
            offenders.append(
                f"{key}: {site['file']}:{site['line']} IEEE quotients "
                f"{site['quotients'][0]} -> {site['quotients'][1]}, multiplies "
                f"by {', '.join(site['gained_immediates'])} on compute_120")
    assert compiled >= 90, f"only {compiled} units compiled"
    assert not offenders, "A146:\n  " + "\n  ".join(offenders)


#: (NVRTC major, minor) -> whether it constant-folds ``tgammaf(4.0f)`` on
#: compute_120 (A174, measured on test host 4's CPU with CUDA hidden): 13.4.92
#: divides by the folded 6.0 under either flush mode, 12.9.86 calls tgammaf.
_GAMMA_FOLDED_ON_COMPUTE_120 = {(12, 9): False, (13, 4): True}


def test_the_line_comparison_sees_what_the_immediate_census_misses():
    """Both shapes the compute_89 census passed, planted: a division by a
    math function of a literal, and two constant divisions in the arms of
    a branch.  The ``__fdiv_rn`` spellings of the same lines pass.

    The two literal divisions are rewritten on compute_120 by every NVRTC
    measured and must be listed under each.  ``/ tgammaf(4.0f)`` is a
    divisor only some compilers fold (A174): NVRTC 13.4.92 folds the call
    to 6.0 on compute_120 and rewrites the quotient; 12.9.86 calls it at run
    time there and keeps a ``div.rn`` of a register, which is the IEEE
    quotient.  So the line must be listed exactly when the reference build
    divides it by the constant 6.0, read from that build's PTX, and the two
    measured compilers must read as measured."""
    nvrtc = _nvrtc()
    from tools.literal_division_census import (_DIV_IMM, _LOC, Unit,
                                               compile_ptx, rewrite_sites)
    template = """
__constant__ float table[16] = {1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f,
    8.5f, 9.5f, 10.5f, 11.5f, 12.5f, 13.5f, 14.5f, 15.5f, 16.5f};
extern "C" __global__ void k(const float* x, const int* n, double* y) {
    int i = threadIdx.x;
    float v = x[i] * tgammaf(4.0f + x[i]) DIVA;
    double lam = 0.0;
    if (v < 1.0e-6f) {
        lam = (double)(table[n[i]] DIVB);
    } else if (v > 50.0e-6f * 2.0f) {
        lam = (double)(table[n[i]] DIVC);
    }
    y[i] = lam + v;
}
"""
    plain = (template.replace("x[i] * tgammaf(4.0f + x[i]) DIVA",
                              "x[i] * tgammaf(4.0f + x[i]) / tgammaf(4.0f)")
             .replace("table[n[i]] DIVB", "table[n[i]] / 1.0e-6f")
             .replace("table[n[i]] DIVC", "table[n[i]] / (50.0e-6f * 2.0f)"))
    fixed = (template.replace("x[i] * tgammaf(4.0f + x[i]) DIVA",
                              "__fdiv_rn(x[i] * tgammaf(4.0f + x[i]), "
                              "tgammaf(4.0f))")
             .replace("(table[n[i]] DIVB)",
                      "__fdiv_rn(table[n[i]], 1.0e-6f)")
             .replace("(table[n[i]] DIVC)",
                      "__fdiv_rn(table[n[i]], 50.0e-6f * 2.0f)"))
    literal = {n + 1 for n, text in enumerate(plain.splitlines())
               if "/ 1.0e-6f" in text or "/ (50" in text}
    (gamma,) = [n + 1 for n, text in enumerate(plain.splitlines())
                if "/ tgammaf" in text]
    assert len(literal) == 2
    # The reference build's constant divisors, by (file, line): the folded
    # tgammaf(4.0f) is a div.rn by 0f40C00000 (6.0) on the planted file's
    # line, or the call stays and the divisor is a register.
    reference = compile_ptx(
        plain, ("-std=c++17", "-lineinfo", "-arch=compute_120", "-ftz=false"))
    divisors, loc = set(), None
    for text in reference.splitlines():
        m = _LOC.match(text)
        if m:
            loc = (int(m.group(1)), int(m.group(2)))
            continue
        m = _DIV_IMM.match(text)
        if m and loc is not None:
            divisors.add((loc, int(m.group(1), 16)))
    (planted_file,) = {f for (f, line), _bits in divisors if line in literal}
    folded = ((planted_file, gamma), 0x40C00000) in divisors
    measured = _GAMMA_FOLDED_ON_COMPUTE_120.get(tuple(nvrtc.getVersion()))
    assert measured is None or folded == measured, (
        f"NVRTC {nvrtc.getVersion()} folds tgammaf(4.0f) on compute_120: "
        f"{folded}, measured {measured}")
    lines = literal | ({gamma} if folded else set())
    found = {site["line"] for site in rewrite_sites(
        Unit("planted", plain, ("-std=c++17",), []))["rewritten_sites"]}
    assert literal <= found, (found, literal)
    assert found == lines, (found, lines, folded)
    assert not rewrite_sites(Unit("fixed", fixed, ("-std=c++17",), []))[
        "rewritten_sites"]


def test_line_comparison_normalizes_constant_register_operands(monkeypatch):
    """A libdevice coefficient loaded by mov is not a gained reciprocal.

    A mov register and a direct immediate are the same coefficient across
    the flush modes. A real reciprocal added beside that coefficient must
    still be reported, including another occurrence of the same value.
    """
    from tools import literal_division_census as census

    unit = census.Unit("planted", "y[0] = logf(x[0] / z[0]);", (), [])
    old_template = """.visible .entry k() {
    .loc 1 1 0
    mov.f32 %f2, 0f3F317218;
    fma.rn.ftz.f32 %f3, %f1, %f2, %f0;
    div.rn.ftz.f32 %f4, %f3, %f1;
    div.rn.ftz.f32 %f5, %f4, %f1;
}
"""
    new_template = """.visible .entry k() {
    .loc 1 1 0
    fma.rn.ftz.f32 %f3, %f1, 0f3F317218, %f0;
    div.rn.ftz.f32 %f4, %f3, %f1;
}
"""
    def compile_fixture(_source, options):
        return old if "-ftz=false" in options else new

    monkeypatch.setattr(census, "compile_ptx", compile_fixture)
    for coefficient in ("3F317218", "3EAAAAAB"):
        old = old_template.replace("3F317218", coefficient)
        new = new_template.replace("3F317218", coefficient)
        assert not census.rewrite_sites(unit)["rewritten_sites"]
        # The second arm collides with an existing RN(1/3) coefficient:
        # set membership would hide this additional reciprocal multiply.
        new = new.replace("}", "mul.ftz.f32 %f5, %f4, 0f3EAAAAAB;\n}")
        found = census.rewrite_sites(unit)["rewritten_sites"]
        assert len(found) == 1 and found[0]["line"] == 1
        assert found[0]["gained_immediates"] == ["0.3333333432674408"]


def test_constant_registers_are_local_and_require_one_write():
    from tools.literal_division_census import _line_counts

    ptx = """.visible .entry first() {
    .loc 1 1 0
    mov.f32 %f2, 0f3F317218;
    mul.ftz.f32 %f3, %f1, %f2;
}
.visible .entry second() {
    .loc 1 2 0
    mov.f32 %f2, 0f3EAAAAAB;
    @%p1 mov.f32 %f2, 0f3F317218;
    mul.ftz.f32 %f3, %f1, %f2;
}
"""
    assert _line_counts(ptx) == {(1, 1): (0, Counter({0x3F317218: 1})),
                                (1, 2): (0, Counter())}


def test_an_existing_coefficient_cannot_hide_a_compiled_reciprocal():
    """A folded divisor adds RN(1/6) beside an existing RN(1/6).

    NVRTC 13.4.92 folds tgammaf(4.0f) to 6.0 on compute_120; 12.9.86 keeps
    the runtime call. The former must list the rewritten line even though
    the coefficient already occurs, and the latter must not invent it.
    """
    nvrtc = _nvrtc()
    from tools.literal_division_census import Unit, compile_ptx, rewrite_sites
    from tools.literal_division_scan import literal_divisions
    from tools.literal_division_census import census_unit

    source = """
extern "C" __global__ void k(const float* x, const float* z, float* y) {
    int i = threadIdx.x;
    float d = tgammaf(4.0f);
    y[i] = x[i] * 0.1666666716337204f + z[i] * tgammaf(4.0f + z[i]) / d;
}
"""
    fixed = source.replace("z[i] * tgammaf(4.0f + z[i]) / d",
                           "__fdiv_rn(z[i] * tgammaf(4.0f + z[i]), d)")
    unit = Unit("coefficient_collision", source, ("-std=c++17",), [])
    assert literal_divisions(source) == []
    assert census_unit(unit)["constant_divisor_sites"] == []
    measured = _GAMMA_FOLDED_ON_COMPUTE_120.get(tuple(nvrtc.getVersion()))
    probe = ('extern "C" __global__ void k(const float* x, float* y) '
             '{ y[0] = x[0] / tgammaf(4.0f); }')
    reference = compile_ptx(probe, ("-std=c++17", "-arch=compute_120",
                                    "-ftz=false"))
    body = reference[reference.index(".entry k("):]
    folded = bool(re.search(
        r"div\.rn\.f32\s+%\w+,\s*%\w+,\s*0f40C00000;", body))
    assert measured is None or folded == measured, (
        nvrtc.getVersion(), folded, measured)
    expected = {5} if folded else set()
    found = {s["line"] for s in rewrite_sites(unit)["rewritten_sites"]}
    assert found == expected, (found, expected, nvrtc.getVersion())
    assert not rewrite_sites(Unit("fixed", fixed, ("-std=c++17",), []))[
        "rewritten_sites"]


def test_compute_120_still_rewrites_and_fdiv_rn_still_holds():
    """The mechanism this gate exists for, measured on the installed NVRTC.

    If the first assertion ever fails, NVRTC stopped rewriting ``x / C`` on
    compute_120 and this gate (and the __fdiv_rn spelling it enforces) can
    be retired with that measurement; the second is the property the fix
    relies on."""
    nvrtc = _nvrtc()
    from tools.literal_division_census import compile_ptx
    src = ('extern "C" __global__ void k(const float* x, float* y)'
           '{ y[0] = x[0] / 3.0f; y[1] = __fdiv_rn(x[1], 3.0f); }')
    ptx = compile_ptx(src, ("-std=c++17", "-ftz=true", "-arch=compute_120"))
    body = ptx[ptx.index(".entry k("):]
    assert "mul.ftz.f32" in body and "0f3EAAAAAB" in body, (
        f"NVRTC {nvrtc.getVersion()} no longer rewrites x / 3.0f on "
        "compute_120; A146's gate may be retired with this measurement")
    assert len(re.findall(r"div\.rn\.ftz\.f32", body)) == 1
    # The line comparison's reference (A174): the same architecture under
    # -ftz=false keeps both divisions as div.rn, the constant one with its
    # divisor as an immediate.  If NVRTC ever rewrites there too, the
    # reference and the Blackwell build agree on every rewritten line, the
    # comparison lists nothing and passes, and this row is what says so.
    # It is read under every option set the census compiles a production
    # unit with (p3_device's -fmad=false among them), since each unit's
    # reference is built with that unit's own options.
    from tools.literal_division_census import production_units
    units = production_units(REPO_ROOT)
    option_sets = {tuple(u.options) for u in units}
    assert ("-std=c++17",) in option_sets, option_sets
    # A unit compiled with --prec-div=false has opted out of IEEE division
    # for EVERY quotient, not only constant ones (x / 3.0f is div.full), so
    # A146 has nothing to hold it to and no div.rn reference exists for it.
    # Only the named fast-division units may do so: EEPS's default runtime
    # (ea830c91a3, fast numerics; its strict replay keeps precise division
    # and is censused under the default option set).  Another unit turning
    # up here needs its own named reason.
    fast_division = {u.key for u in units if "--prec-div=false" in u.options}
    assert fast_division <= {"kernels:eeps"}, fast_division
    for options in sorted(option_sets):
        if "--prec-div=false" in options:
            continue
        ref = compile_ptx(src, options + ("-ftz=false", "-arch=compute_120"))
        ref_body = ref[ref.index(".entry k("):]
        assert re.search(r"div\.rn\.f32\s+%\w+,\s*%\w+,\s*0f40400000;",
                         ref_body), (
            f"NVRTC {nvrtc.getVersion()} rewrites x / 3.0f on compute_120 "
            f"under -ftz=false with {options} as well; the line comparison "
            "has no reference")
        assert len(re.findall(r"div\.rn\.f32", ref_body)) == 2, options
        assert "0f3EAAAAAB" not in ref_body, options
