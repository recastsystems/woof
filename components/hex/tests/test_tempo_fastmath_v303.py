"""Fail closed on source drift and exercise the fast helpers' fallback guards.

The host witness checks dispatch and range handling, not CUDA accuracy or
forecast acceptance. Real capture replay supplies those measurements.
"""
import ctypes
import math
from pathlib import Path
import shutil
import subprocess

import pytest

from woof.hex.mpasphys.rrfs import tempo_fastmath_v303 as fast
from woof.hex.mpasphys.rrfs import tempo_v303 as tempo


def test_transform_covers_helpers_without_changing_header_or_callers():
    source = "// external maths header\n" + tempo._COLUMN
    result = fast.fast_source(source)
    assert result.startswith("// external maths header\n")
    assert "g64_pow::" not in result
    for old, new in reversed(list(fast._replacements())):
        assert result.count(new) == 1
        result = result.replace(new, old)
    assert result == source


@pytest.mark.parametrize("old,new", list(fast._replacements()))
@pytest.mark.parametrize("mutation", ["missing", "duplicate"])
def test_changed_or_duplicated_helper_fails(old, new, mutation):
    source = tempo._COLUMN.replace(old, "" if mutation == "missing" else old + old)
    with pytest.raises(ValueError, match="TEMPO fast maths: expected one"):
        fast.fast_source(source)


def test_double_transform_fails():
    with pytest.raises(ValueError, match="TEMPO fast maths: expected one"):
        fast.fast_source(fast.fast_source(tempo._COLUMN))


@pytest.fixture(scope="module")
def host_helpers(tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required for the fallback dispatch witness")
    directory = tmp_path_factory.mktemp("tempo-fast-guards")
    helpers = fast.fast_source(tempo._COLUMN)
    helpers = helpers[helpers.index(fast._CORE):helpers.index("// RRFS MPAS-Model-rrfs", helpers.index(fast._CORE))]
    aliases = "\n".join(
        f"#define mpas_{name}{suffix} fallback" for name in ("exp", "log", "log10", "pow", "tgamma")
        for suffix in ("", "f"))
    source = r'''
#include <cmath>
using namespace std;
#define TEMPO_HD inline
static int fallback_count;
template<class T> T fallback(T x) { ++fallback_count; return T(-917); }
template<class T> T fallback(T x,T y) { ++fallback_count; return T(-917); }
''' + aliases + "\n" + helpers + r'''
extern "C" double evaluate(int op, double x, double y, int narrow) {
  fallback_count=0;
  if (op==4) return tempo_pow_with(tempo_pow_base(x),y);
  if (narrow) {
    if(op==0) return tempo_exp((float)x);
    if(op==1) return tempo_log((float)x);
    if(op==2) return tempo_log10((float)x);
    return tempo_pow((float)x,(float)y);
  }
  if(op==0) return tempo_exp(x);
  if(op==1) return tempo_log(x);
  if(op==2) return tempo_log10(x);
  return tempo_pow(x,y);
}
extern "C" int fallbacks() { return fallback_count; }
'''
    path = directory / "guards.cpp"
    path.write_text(source)
    library = directory / "guards.so"
    subprocess.run([compiler, "-std=c++17", "-O2", "-shared", "-fPIC", str(path), "-o", str(library)],
                   check=True, capture_output=True, timeout=30)
    lib = ctypes.CDLL(str(library))
    lib.evaluate.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_int]
    lib.evaluate.restype = ctypes.c_double
    return lib


@pytest.mark.parametrize("narrow", [0, 1])
def test_exceptional_operands_reach_exact_replica(host_helpers, narrow):
    lib = host_helpers
    cases = [(op, x, 2.0) for op in range(5) for x in (math.nan, math.inf, -math.inf)]
    cases += [(op, x, .5) for op in (1, 2, 3, 4) for x in (0.0, -0.0, -1.0)]
    cases += [(op, 2.0, y) for op in (3, 4) for y in (math.nan, math.inf, -math.inf, 129.0)]
    cases += [(0, x, 0.0) for x in (-10000.0, 10000.0)]
    cases += [(op, 1e-310, 2.0) for op in (1, 2, 3, 4)]
    for op, x, y in cases:
        assert lib.evaluate(op, x, y, narrow) == -917, (op, x, y, narrow)
        assert lib.fallbacks() == 1


def test_wide_normal_results_do_not_narrow_to_zero_or_infinity(host_helpers):
    lib = host_helpers
    for op, x, y, expected in ((0, 500., 0., math.exp(500.)),
                              (0, -500., 0., math.exp(-500.)),
                              (1, 1e200, 0., math.log(1e200)),
                              (2, 1e-200, 0., -200.),
                              (3, 1e100, 2., 1e200),
                              (4, 1e100, 2., 1e200)):
        assert lib.evaluate(op, x, y, 0) == pytest.approx(expected, rel=5e-6)
        assert lib.fallbacks() == 0
    for op in (3, 4):
        assert lib.evaluate(op, 1e300, 2., 0) == -917
        assert lib.fallbacks() == 1
        assert lib.evaluate(op, 1e-300, 2., 0) == -917
        assert lib.fallbacks() == 1
