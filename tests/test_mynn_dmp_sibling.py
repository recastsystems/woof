"""The shared DMP source keeps the frozen default and plume-export arms.

Source pins come from a24a0c478e3, before the sibling fold. GPU exactness and
whole-forecast BASE/TIP receipts are required before this fold lands.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import re

import pytest

_KDIR = Path(__file__).resolve().parents[1] / "woof" / "core" / "kernels"
_MARKER = "// MF-EXPORT"


def _read(name: str) -> str:
    return (_KDIR / name).read_text(encoding="utf-8")


def _export_arm(source, enabled):
    return re.sub(r"#ifdef MYNN_EXPORT_PLUME_EDGE\n(.*?)#endif\n",
                  lambda match: match[1] if enabled else "", source, flags=re.S)


def _dmp_entry(source):
    start = source.index('extern "C" __global__\nvoid mynn_dmp_mf_columns(')
    return source[start:source.index("\n}\n", start) + 3]


def _without_gsd41(source):
    """The source the default build compiles: MYNN_GSD41 undefined.

    The GSD MYNN v4.1 rows (bl_mynn_version = "gsd_41") sit behind
    ``#if defined(MYNN_GSD41)`` / ``#if !defined(MYNN_GSD41)`` and reach a
    run only through the integer-define loader; this evaluates those
    conditionals with the define absent, then undoes the remaining spellings the
    gsd_41 rows needed outside them (the length macros, the kernel argument
    lists' closing parenthesis).
    """
    out, stack = [], []
    for line in source.splitlines(True):
        text = line.strip()
        if text in ("#if defined(MYNN_GSD41)", "#if !defined(MYNN_GSD41)"):
            stack.append(text == "#if !defined(MYNN_GSD41)")
            continue
        if stack and text == "#else":
            stack[-1] = not stack[-1]
            continue
        if stack and text == "#endif":
            stack.pop()
            continue
        if all(stack):
            out.append(line)
    assert not stack
    source = "".join(out)
    edits = (
        ("#define MYNN_GSD41_LENGTH_PARAMS\n"
         "#define MYNN_GSD41_LENGTH_ARGS(flt_, unsq_)\n", ""),
        ("int mixlength MYNN_GSD41_LENGTH_PARAMS)", "int mixlength)"),
        ("\n                               MYNN_GSD41_LENGTH_ARGS("
         "0.0f, unsquared_qtke));", ");"),
        ("\n        MYNN_GSD41_LENGTH_ARGS(flt_c, unsquared_qtke));", ");"),
    )
    for current, historical in edits:
        assert source.count(current) == 1, current
        source = source.replace(current, historical)
    # Kernels that take a gsd_41 argument close their parameter list on a
    # line of its own; the historical source never does.
    assert source.count("\n    )\n{") >= 1
    return source.replace("\n    )\n{", ")\n{")


def test_default_source_pin_unchanged():
    source = _export_arm(_read("mynn_pbl.cu"), False)
    # RE-PINNED for 3.0.0: 6d79b6e9c5d added MYNN_GSD41-only DMP cloud-base
    # outputs inside #if defined(MYNN_GSD41); the default arm compiles the
    # same program (test_spp_consumers records the same re-pin). Was 176d00a6.
    assert hashlib.sha256(source.encode()).hexdigest() == "a22ef22ada5cd82facb1ef7651935ef49f4a06f820cb48d2d5fdfd9d391b61f6"


def test_export_arm_is_the_frozen_sibling_dmp_entry():
    source = _without_gsd41(_export_arm(_read("mynn_pbl.cu"), True))
    assert hashlib.sha256(_dmp_entry(source).encode()).hexdigest() == "a83ed1f766e20c39b2e86eaaa71b3ce4e71ea8e062567ffa4316809817ad0085"


def test_the_marker_is_actually_exercised():
    source = _read("mynn_pbl.cu")
    tagged = [line for line in source.splitlines() if _MARKER in line]
    assert len(tagged) == 12
    assert _MARKER not in _export_arm(source, False)
    for name in ("up_a_pre", "psig_w_o", "plume_active_o", "limiter_adjustment_o"):
        assert name in "\n".join(tagged)


def test_logical_sibling_keeps_its_compiler_policy_and_shared_source():
    from woof.core import kernels

    source = kernels.module_source("mynn_dmp_sibling")
    assert "#define MYNN_EXPORT_PLUME_EDGE 1\n" in source
    assert source.endswith(_read("mynn_pbl.cu"))
    assert kernels.module_options("mynn_dmp_sibling") == ("-std=c++17", "--fmad=false")
    assert "mynn_dmp_sibling" not in kernels._NO_FTZ_MODULES
    assert "mynn_pbl" in kernels._NO_FTZ_MODULES


@pytest.mark.parametrize("shape", [(), (3, 2, 4)])
def test_default_dmp_still_launches_the_original_module(shape):
    """The DMP dispatch uses the sibling only for scalar-mixing exports."""
    # Execute the production dispatch and its validator without importing the
    # device-array driver. This guard tests loader routing, not CUDA execution.
    path = _KDIR.parent / "mynn_pbl_gpu.py"
    core = path.read_text(encoding="utf-8")
    tree = ast.parse(core)
    dispatch = ast.Module(body=[node for node in tree.body if (
        isinstance(node, ast.FunctionDef)
        and node.name in {"mynn_pbl_kernel", "_mynn_version_check"}
    ) or (
        isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "MYNN_VERSIONS"
                for target in node.targets)
    )], type_ignores=[])

    calls, original = [], object()

    def capture(module, function, shape=()):
        calls.append((module, function, shape))
        return original

    def refuse_defines(*args, **kwargs):
        raise AssertionError("generic MYNN must use the original module without defines")

    namespace = {"get_kernel": capture, "get_kernel_int_defines": refuse_defines}
    exec(compile(dispatch, str(path), "exec"), namespace)
    for args in (("mynn_dmp_mf_columns",), ("mynn_dmp_mf_columns", "wrf_461")):
        assert namespace["mynn_pbl_kernel"](*args, shape=shape) is original
    assert calls == [("mynn_pbl", "mynn_dmp_mf_columns", shape)] * 2

    assert ('mynn_pbl_kernel("mynn_dmp_mf_columns", bl_mynn_version, '
            '(nz, 1, ncol))') in core
    assert ('get_kernel("mynn_dmp_sibling", "mynn_dmp_mf_columns", '
            '(nz, 1, ncol))') in core
