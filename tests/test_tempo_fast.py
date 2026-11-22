"""TEMPO v3.2.5 fast numerics (woof.core.tempo_fast): what it may and may not touch.

Breakage these tests prevent:

* the strict column (the one the native v3.2.5 oracle holds bitwise) picking
  up the binary32 transform, or the pinned v3.0.3 generation being transformed;
* a header edit silently escaping the transform (a struct or maths block the
  transform no longer finds would leave binary64 or untiled code in the fast
  module, or a mis-anchored replacement would compile something else);
* the fast kernel and the runtime kernel drifting apart in entry points (the
  runtime looks every kernel up by name);
* a table value outside binary32 range reaching the fast blob as inf.
"""
from __future__ import annotations

import re

import numpy as np
import pytest

from woof.core import tempo_fast as tf


def _runtime():
    return pytest.importorskip("woof.core.tempo_runtime")


def test_fast_float_is_fast_numerics_and_v325_only(monkeypatch):
    tr = _runtime()
    from woof import numerics
    monkeypatch.delenv("WOOF_TEMPO_FAST_FLOAT", raising=False)
    monkeypatch.setattr(numerics, "mode", lambda: "fast")
    assert tr.fast_float_selected("v3.2.5")
    assert not tr.fast_float_selected("v3.0.3")
    monkeypatch.setenv("WOOF_TEMPO_FAST_FLOAT", "0")
    assert not tr.fast_float_selected("v3.2.5")
    monkeypatch.delenv("WOOF_TEMPO_FAST_FLOAT")
    monkeypatch.setattr(numerics, "mode", lambda: "strict")
    assert not tr.fast_float_selected("v3.2.5")


def test_fast_ctx_matches_the_binary32_context_struct():
    tr = _runtime()
    fast = tr.ctx_dtype("v3.2.5", fast_float=True)
    assert fast.names == tr.ctx_dtype("v3.2.5").names
    assert fast["global_dt"] == np.float32 and fast["global_inverse_dt"] == np.float32
    assert fast.itemsize == tr.ctx_dtype("v3.2.5").itemsize - 8


def test_fast_source_is_binary32_tiled_and_carries_no_glibc_words():
    tr = _runtime()
    src = tf.kernel_source(tr.kernel_dir("v3.2.5"))
    # every TEMPO header is expanded; the binary32 exner library is the only other text
    for marker in ("struct TempoWork {", "#define TEMPO_WA_MEMBER(name) TCol<float> name;", "struct TempoTend {",
                   "TEMPO_HDN inline void tempo_main_column(", "__expf(x)"):
        assert marker in src, marker
    nocomment = re.sub(r"//[^\n]*", "", src)
    assert "glibc_exp(" not in nocomment and '#include "../glibc_flt64.cuh"' not in src
    assert "TCol<double>" not in src and "TSP<double>" not in src
    for header in sorted(tr.kernel_dir("v3.2.5").glob("*.cuh")):
        code = re.sub(r"//[^\n]*", "", tf._float_header(header.read_text(encoding="utf-8"), header.name))
        assert not re.search(r"\bdouble\b", code), f"binary64 left in {header.name}"


def test_fast_and_runtime_kernels_define_the_same_entry_points():
    tr = _runtime()
    for name in tf.ENTRY_POINTS:
        assert f'extern "C" __global__ void {name}(' in tf.KERNEL, name
        assert f'extern "C" __global__ void {name}(' in tr._KERNEL, name
    assert set(re.findall(r'extern "C" __global__ void (\w+)\(', tr._KERNEL)) == set(tf.ENTRY_POINTS)


@pytest.mark.parametrize("name,anchor,drifted", [
    ("tempo_common.cuh", "struct TempoTend {", "struct TempoTend  {"),
    ("tempo_common.cuh", '#include "../glibc_flt64.cuh"', '#include "../glibc_flt64x.cuh"'),
    ("tempo_common.cuh", "#define TEMPO_NZP (TEMPO_NZ_MAX + 3)", "#define TEMPO_NZP (TEMPO_NZ_MAX+3)"),
    ("tempo_column.cuh", "struct TempoWork {", "struct TempoWork  {"),
    ("tempo_column.cuh", "    if (l[k]) return true;", "    if (l[k] != 0) return true;"),
])
def test_header_drift_is_refused(name, anchor, drifted):
    tr = _runtime()
    text = (tr.kernel_dir("v3.2.5") / name).read_text(encoding="utf-8")
    assert text.count(anchor) == 1
    tf._float_header(text, name)  # the shipped header transforms
    with pytest.raises(ValueError, match="TEMPO fast"):
        tf._float_header(text.replace(anchor, drifted), name)


def test_literals_become_binary32_and_identifiers_do_not():
    sub = lambda s: tf._FLOAT_LITERAL.sub(lambda m: m.group(1) + "f", s)
    assert sub("x = 1.e-4 * y + .5 - 2.0E3 + 1e10;") == "x = 1.e-4f * y + .5f - 2.0E3f + 1e10f;"
    assert sub("PA1(t1_qr_qc, 2) + 0x1.8p3 + k2") == "PA1(t1_qr_qc, 2) + 0x1.8p3 + k2"
    assert tf._POINTER.sub(lambda m: f"TSP<{m.group(1) or ''}{m.group(2)}> {m.group(3)}",
                           "const double* rho, bool* l_qr, int* ktop, double x") == \
        "TSP<const double> rho, TSP<bool> l_qr, TSP<int> ktop, double x"


def test_blob_overflow_is_refused_by_name():
    layout = [("small", 2), ("huge", 1)]
    assert tf.float_blob(np.array([1.0, 2.0, 3.0]), layout).dtype == np.float32
    with pytest.raises(ValueError, match="huge"):
        tf.float_blob(np.array([1.0, 2.0, 1.0e300]), layout)


def test_fast_tempo_any_is_branch_free_and_strict_is_untouched():
    """Breakage: the early-return tempo_any cost 34 % of a TEMPO call on sm_120 (tempo_fast._ANY_OLD)."""
    tr = _runtime()
    d = tr.kernel_dir("v3.2.5")
    assert tf._ANY_OLD in (d / "tempo_column.cuh").read_text(encoding="utf-8")  # strict: TEMPO's own loop
    fast = tf.kernel_source(d)
    assert "if (l[k]) return true;" not in fast
    assert "for (int k = 1; k <= nz; ++k) a = a | l[k];" in fast
