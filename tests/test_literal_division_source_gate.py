"""A146 gate, source layer: no float division by a constant outside __fdiv_rn.

The breakage this prevents: NVRTC's NVVM, for compute_100 and compute_120
(every Blackwell card: RTX 5090, 5070 Ti, PRO 4500, PRO 6000) under
``-ftz=true`` (which CuPy appends to every ``RawModule``), compiles
``x / C`` for a compile-time constant ``C`` as ``x * RN(1/C)``.  That is not
IEEE division: on an RTX 5070 Ti with NVRTC 13.4, exhaustively over every
float in [1, 2), ``x / 3.0f`` is one ULP off for 33 percent of inputs,
``x / 60.0f`` (the advection flux) for 65 percent, ``x / tgammaf(4.0f)`` for
33 percent, while compute_90 and older keep ``div.rn`` and round correctly.
``__fdiv_rn(x, C)`` is never rewritten and is the IEEE quotient on every
card, so it is the one spelling a kernel may use for such a division.

This file imports neither CuPy nor NVRTC, so it runs on every leg, the CPU
legs that set ``GPUWM_NO_LOCAL_GPU`` included (tools/battery/always_files.txt
lists it).  The compiler census, which sees divisors a source scan cannot,
is tests/test_literal_division_gate.py on the GPU shard.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tools.literal_division_scan import (
    FTZ_FALSE_SOURCES,
    inline_kernel_offenders,
    inline_kernel_strings,
    kernel_file_offenders,
    literal_divisions,
)

#: The tree under test: the scan reads every source from it (A193), never
#: from wherever woof happens to be imported.
REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("strict", [False, True])
def test_port_constant_divisions_take_the_direct_non_ftz_route(monkeypatch, strict):
    """A source exception must follow the actual runtime compile dispatch.

    Check the integer-tier path as well: EEPS and CAM set their column depth
    there, and the ordinary loader alone would leave their runtime unchanged.
    """
    import sys
    from types import SimpleNamespace
    from woof import numerics, wrf_exact
    from woof.core import kernels

    monkeypatch.setattr(wrf_exact, "ENABLED", strict)
    monkeypatch.setattr(numerics, "mode", lambda: "strict" if strict else "fast")
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace())
    calls = []
    sentinel = object()

    def compile_unit(name, source, key):
        calls.append((name, source, key, kernels.module_options(name)))
        return sentinel

    monkeypatch.setattr(kernels, "compile_noftz_module", compile_unit)
    ports = {"gfdleta_init", "gfdleta_lw_helpers", "gfdleta_tables",
             "afwa_buoyancy", "afwa_severe", "misc_diag_print", "snow_papers",
             "gfdleta_column", "gfdleta_math_probe", "gfdleta_co2_init",
             "gfdleta_table_runtime", "cam3_emissivity", "eeps",
             "gsfcgce4ice_aux", "qnsepbl", "qnsesfc", "sas", "sstskin",
             "temf", "temf_sfclay", "idealscmsfclay",
             'pxsfclay', 'ssib', 'trajectory', 'tslist', 'wrf_stock_aer3',
             'afwa_diagnostics', 'dfi_reset', 'global_advection', 'global_geometry', 'global_hydrostatic', 'global_polar', 'level_diagnostics', 'nwp_windows', 'nssl2_shape_gs'}
    assert kernels._PORT_IEEE_DIVISION_MODULES == ports
    for name in sorted(ports | {"gsfcsw", "kfcup"}):
        assert kernels.load_module.__wrapped__(name) is sentinel
        assert calls[-1][:3] == (name, kernels.module_source(name),
                                f"{kernels.MODULE_KEY_ROOT}:{name}")
        assert "--ftz=false" in calls[-1][3]
        defines = (("COLUMN_DEPTH", 34),)
        assert kernels.load_module_int_defines.__wrapped__(name, defines) is sentinel
        assert calls[-1][:3] == (
            name, kernels.module_source_int_defines(name, defines),
            f"{kernels.MODULE_KEY_ROOT}:{name}[COLUMN_DEPTH=34]")
        assert "--ftz=false" in calls[-1][3]
    for name in ("cam3_emissivity.cu", "eeps.cu", "gfdleta_co2_init.cuh",
                 "gfdleta_column.cuh", "gsfcgce4ice_aux.cu", "qnsepbl.cu",
                 "sas.cu", "sstskin.cu", "temf.cu", "gsfcsw.cu",
                 "gsfcsw_fast.cuh", "kfcup.cu"):
        assert name in FTZ_FALSE_SOURCES
    assert "--fmad=false" not in kernels.module_options("eeps")
    assert "--fmad=false" not in kernels.module_options("sstskin")
    assert ("--fmad=false" in kernels.module_options("qnsepbl")) == strict


def test_other_port_source_exceptions_keep_their_existing_non_ftz_routes(monkeypatch):
    import sys
    from types import SimpleNamespace
    from woof.core import kernels

    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace())
    sentinel = object()
    calls = []

    def diffusion(name, defines=()):
        calls.append((name, defines))
        return sentinel

    monkeypatch.setattr(kernels, "_load_diffusion_module", diffusion)
    assert kernels.load_module.__wrapped__("km5") is sentinel
    assert kernels.load_module_int_defines.__wrapped__("km5", (("NZ", 34),)) is sentinel
    assert calls == [("km5", ()), ("km5", (("NZ", 34),))]
    assert "--ftz=false" in kernels.module_options("km5")
    assert "km5.cu" in FTZ_FALSE_SOURCES

    def ishmael(name, source, options):
        calls.append((name, source, options))
        return sentinel

    monkeypatch.setattr(kernels, "compile_ishmael_no_ftz", ishmael)
    assert kernels.load_module.__wrapped__("ishmael") is sentinel
    assert calls[-1] == ("ishmael", kernels.module_source("ishmael"),
                         kernels.module_options("ishmael"))
    assert "--ftz=false" in calls[-1][-1]
    assert "ishmael.cu" in FTZ_FALSE_SOURCES


def test_terrain_drag_divisions_use_the_direct_non_ftz_production_route():
    """Its source exception requires the loader that preserves IEEE division."""
    from woof.core import terrain_drag
    from tools.ftz_receipt import route_inventory

    source = (REPO_ROOT / "woof" / "core" / "terrain_drag.py").read_text(
        encoding="utf-8")
    sites = route_inventory.scan_source("woof/core/terrain_drag.py", source)
    assert len(sites) == 1
    assert sites[0]["constructor_kind"] == (
        "cupy.cuda.compiler.compile_using_nvrtc")
    assert sites[0]["options_expression"] == "MODULE_OPTIONS"
    assert terrain_drag.MODULE_OPTIONS == (
        "-std=c++17", "-fmad=false", "--ftz=false")
    assert "terrain_drag.cu" in FTZ_FALSE_SOURCES


def test_cumulus_and_lightning_exceptions_compile_without_ftz(monkeypatch):
    """2.9.0 round 2: g3.cu, nsas.cu and lightning.cu are exempt only because
    their loaders never append -ftz=true; if a route changes, the exception
    must go and their constant divisions must take __fdiv_rn."""
    from woof.core import kernels

    for module in ("g3", "nsas"):
        assert module in kernels._NO_FTZ_MODULES
        assert "--ftz=false" in kernels.module_options(module)
        assert "--fmad=false" in kernels.module_options(module)
        assert f"{module}.cu" in FTZ_FALSE_SOURCES
    import sys
    from types import SimpleNamespace
    from woof import wrf_exact, nvrtc_ptx_cache
    from importlib import import_module
    kernel_manifest = import_module("woof.certify.kernel_manifest")

    calls = []
    native = SimpleNamespace(load=lambda image: None)
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(
        cuda=SimpleNamespace(function=SimpleNamespace(Module=lambda: native))))
    def compile_unit(source, options, arch, filename):
        calls.append((source, options, filename))
        return b"measured-route-test", None
    monkeypatch.setattr(nvrtc_ptx_cache, "compile_using_nvrtc", compile_unit)
    monkeypatch.setattr(kernel_manifest, "record_module", lambda *a, **kw: None)
    for strict in (False, True):
        monkeypatch.setattr(wrf_exact, "ENABLED", strict)
        for name in ("lightning", "native_surface_diagnostics"):
            kernels.load_module.__wrapped__(name)
            assert calls[-1] == (kernels.module_source(name),
                                  kernels.module_options(name), f"{name}.cu")
            assert "--ftz=false" in calls[-1][1]
            assert ("--fmad=false" in calls[-1][1]) == strict
            assert f"{name}.cu" in FTZ_FALSE_SOURCES
    # Existing native loaders retain their explicit no-FTZ choices.
    for name in ("clwrf", "mskf", "mskf_aerosol_replay", "wind_legacy", "wind_mav"):
        assert "--ftz=false" in kernels.module_options(name)
    from woof.core import ntu
    assert "--ftz=false" in ntu.compile_options()
    assert "ntu.cu" in FTZ_FALSE_SOURCES and "ntu_parallel.cuh" in FTZ_FALSE_SOURCES


@pytest.mark.parametrize("strict", [False, True])
def test_3_0_port_source_exceptions_keep_non_ftz_routes(monkeypatch, strict):
    """The 3.0.0 FTZ_FALSE_SOURCES rows follow the real runtime dispatch.

    Each file is exempt from the __fdiv_rn spelling only because no runtime
    route compiles it with -ftz=true; if a route moves, its row must go.
    """
    import sys
    from types import SimpleNamespace
    from woof import numerics, wrf_exact
    from woof.core import kernels, ntu

    monkeypatch.setattr(wrf_exact, "ENABLED", strict)
    monkeypatch.setattr(numerics, "mode", lambda: "strict" if strict else "fast")
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace())
    calls = []
    sentinel = object()

    def compile_unit(name, source, key):
        calls.append(name)
        return sentinel

    monkeypatch.setattr(kernels, "compile_noftz_module", compile_unit)
    for name in ("pxsfclay", "mskf", "mskf_aerosol_replay", "wind_mav",
                 "wind_legacy", "clwrf"):
        assert kernels.load_module.__wrapped__(name) is sentinel
        assert calls[-1] == name
        assert f"{name}.cu" in FTZ_FALSE_SOURCES or name == "mskf_aerosol_replay"
    for name in ("mskf", "mskf_aerosol_replay"):
        assert "mskf_aerosol.cuh" in kernels.EXTRA_HEADERS[name]
    assert {name for name, headers in kernels.EXTRA_HEADERS.items()
            if "mskf_aerosol.cuh" in headers} <= {"mskf", "mskf_aerosol_replay"}
    assert "mskf_aerosol.cuh" in FTZ_FALSE_SOURCES
    # ntu: its own direct NVRTC route appends ntu_parallel.cuh.
    assert "--ftz=false" in ntu.MODULE_OPTIONS
    assert "--ftz=true" not in ntu.compile_options()
    ntu_py = (REPO_ROOT / "woof/core/ntu.py").read_text(encoding="utf-8")
    assert "compile_using_nvrtc(module_source(kmax), compile_options()," in ntu_py
    assert '"ntu_parallel.cuh"' in ntu_py
    assert {"ntu.cu", "ntu_parallel.cuh"} <= FTZ_FALSE_SOURCES
    # dfi_reset: woof.core.dfi compiles it preserving subnormals.
    dfi_py = (REPO_ROOT / "woof/core/dfi.py").read_text(encoding="utf-8")
    assert "options = ('-std=c++17', '--ftz=false')" in dfi_py
    assert ("compile_module_preserving_subnormals(n, module_source(n), options)"
            in dfi_py)
    assert "dfi_reset.cu" in FTZ_FALSE_SOURCES
    # global_*: every production load is kernels.load_module_no_ftz.
    import re
    for name in ("global_advection", "global_geometry", "global_hydrostatic",
                 "global_polar"):
        loads = []
        for path in sorted((REPO_ROOT / "woof").rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            loads += re.findall(
                rf"(\w+)\(\s*['\"]{name}['\"]", text)
        assert loads and set(loads) <= {"load_module_no_ftz"}, (name, loads)
        assert f"{name}.cu" in FTZ_FALSE_SOURCES


def test_nextgen_reservoir_divisions_use_non_ftz():
    import ast
    from woof.core import ieee_module
    source = (REPO_ROOT / "woof/hydro/reservoirs.py").read_text()
    loader = next(node for node in ast.parse(source).body
                  if isinstance(node, ast.FunctionDef) and node.name == "compile_unit")
    block = ast.get_source_segment(source, loader)
    assert 'compile_module(kernel_source(strict), tuple(options), "nextgen_reservoirs")' in block
    assert 'options = ["--std=c++17", "--ftz=false"]' in block
    assert "nextgen_reservoirs.cuh" in FTZ_FALSE_SOURCES
    assert "--ftz=false" in ieee_module._effective_options(("--ftz=false",))


def test_kernel_sources_divide_by_float_constants_only_through_fdiv_rn():
    """A146: every -ftz=true kernel source spells a constant divisor as
    ``__fdiv_rn(x, C)``; a plain ``x / C`` is rounded one ULP off on
    Blackwell cards for a large share of inputs."""
    offenders = kernel_file_offenders(REPO_ROOT)
    assert not offenders, (
        "A146: NVRTC compiles a float division by a compile-time constant "
        "as a multiply by the rounded reciprocal on compute_100/120 under "
        "-ftz=true, which is not IEEE on Blackwell cards; spell these as "
        "__fdiv_rn(x, C):\n  " + "\n  ".join(offenders))


def test_inline_python_kernels_divide_by_float_constants_only_through_fdiv_rn():
    """A146: the same rule for CUDA source held in woof's Python modules
    (RawKernel/RawModule strings and ElementwiseKernel operations), which
    CuPy compiles with the same appended -ftz=true."""
    offenders = inline_kernel_offenders(REPO_ROOT)
    assert not offenders, (
        "A146: spell these constant divisions as __fdiv_rn(x, C):\n  "
        + "\n  ".join(offenders))


def test_the_source_scan_catches_every_spelling_it_claims():
    """The scan is only a gate if it fires: each plain spelling below is
    refused and each allowed one is not."""
    header = {"THOMPSON_AA_D0C": 1.0e-6, "THOMPSON_AA_D0R": 50.0e-6}
    refused = [
        "y = x / 3.0f;", "y = a * b / 60.0f;", "y = x / G;", "y /= 12.0f;",
        "y = x / 1.718e-5f;", "#define KD 7.0f\ny = x / KD;",
        "const float c = 9.81f; y = x / c;", "y = (a - b) / 0.075f;",
        # The four sites the first A146 repair missed (review of 4810400f1):
        # a header constant, a parenthesized constant, and a C math
        # function of a literal, which compute_120 folds and compute_89
        # calls at run time.
        "lamc = (double)(cce2[nu] / THOMPSON_AA_D0C);",
        "lamc = (double)(cce2[nu] / (THOMPSON_AA_D0R * 2.0f));",
        "v = d * c * powf(x, e) * tgammaf(4.0f + e) / tgammaf(4.0f);",
        "v = d * c * powf(x, e) * tgammaf(5.0f + e) / tgammaf(5.0f);",
        # A constant numerator built from math calls is not folded before
        # the rewrite can see it: tgammaf(a) * powf(b, c) / tgammaf(d).
        "c1 = tgammaf(0.53f) * powf(0.2f, -1.0f / 3.0f) / tgammaf(0.2f);",
        "z = zm + (t - tm) / (-9.81f / 1004.5f);",
    ]
    allowed = [
        "y = __fdiv_rn(x, 3.0f);", "y = x / 2.0f;", "y = x / 0.5f;",
        "y = x / 3.0;", "y = x / dz;", "y = (1.0f / 6.0f) * q;",
        "c = -9.81f / 1004.5f;", "y = x / 3.0f; // comment",
        "// y = x / 3.0f;", "y = x / G_fn(z);", "y = x / (2.0 * 1.0e-12);",
        "y = x / (float)n;", "y = x / (2.0f * 4.0f);", "y = x / (dt * 2.0f);",
        "y = x / tgamma(4.0);", "y = x / tgammaf(e);",
        "y = __fdiv_rn(a * tgammaf(4.0f + e), tgammaf(4.0f));",
    ]
    for src in refused:
        assert literal_divisions(src, header, header), src
    for src in allowed:
        hits = literal_divisions(src, header, header)
        if src.endswith("// comment"):
            assert hits == [(1, "3.0f")], src
        else:
            assert not hits, (src, hits)


def test_the_inline_scan_reads_every_kernel_string_form():
    """A RawKernel body, an f-string body and an ElementwiseKernel
    operation each reach the scan; a docstring does not."""
    module = textwrap.dedent('''
        """Docs may say y = x / 3.0f; freely."""
        import cupy as cp
        BODY = r"""
        extern "C" __global__ void k(const float* x, float* y)
        { y[0] = x[0] / 3.0f; }
        """
        OP = cp.ElementwiseKernel("float32 x", "float32 y",
                                  "y = x / 60.0f", "op")
        TIER = 64
        FMT = f"""__device__ float h(float a) {{ return a / 12.0f; }}
        // {TIER}"""
    ''')
    divisors = sorted(token for _lineno, text in inline_kernel_strings(module)
                      for _line, token in literal_divisions(text))
    assert divisors == ["12.0f", "3.0f", "60.0f"], divisors
