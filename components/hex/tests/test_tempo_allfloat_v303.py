"""rrfs/tempo-yolo-allfloat: the fast TEMPO module names no binary64 type; strict is untouched (CPU, source level).

The SASS proof (0 binary64 arithmetic and 0 binary64 conversions in every kernel of the fast warp module,
nvcc 13.4, sm_120) and the 3 km replay are in the lane receipts; this pins the source properties.
"""
import math
import re

import pytest

from woof.hex.mpasphys.rrfs import tempo_v303 as tempo
from woof.hex.mpasphys.rrfs.tempo_allfloat_v303 import all_float_mpas, all_float_source, scalar_names
from woof.hex.mpasphys.rrfs.tempo_fastdp_v303 import fast_dp_source

MATHS = "// device maths\n"


def test_strict_module_is_the_proved_source():
    assert tempo._column_source("strict") is tempo._COLUMN
    assert tempo._mpas_source("strict") is tempo._MPAS
    assert tempo.warp_source(MATHS, max_levels=59, numerics="strict").endswith(tempo._COLUMN + tempo._WARP + tempo._MPAS)
    assert tempo.cuda_source(MATHS, max_levels=59, numerics="strict").endswith(tempo._COLUMN + tempo._DRIVER + tempo._MPAS)


def test_fast_module_names_no_binary64():
    for source in (tempo.warp_source(MATHS, max_levels=59, numerics="fast"),
                   tempo.cuda_source(MATHS, max_levels=59, numerics="fast")):
        code = re.sub(r"//[^\n]*", "", source)
        assert not re.search(r"\bdouble\b", code)
        assert not re.search(r"\bmpas_\w+\(", code), sorted(set(re.findall(r"\bmpas_\w+\(", code)))
        assert "__constant__ TempoScalarsF tempo_k;" in source and "__ldg(&P.L->" in source


def test_constant_block_matches_the_table_layout():
    names = scalar_names(tempo._COLUMN)
    assert names == [n for n, shape in tempo.TABLE_LAYOUT if math.prod(shape) == 1]
    fast = tempo._column_source("fast")
    block = fast[fast.index("struct TempoScalarsF {"):fast.index("};", fast.index("struct TempoScalarsF {"))]
    assert re.findall(r"float (\w+);", block) == names
    assert len(tempo._scalar_offsets()) == len(names)
    offsets = dict(zip(names, tempo._scalar_offsets()))
    assert offsets["av_r"] == 0 and offsets["bv_r"] == 1


def test_all_float_refuses_drift_and_reapplication():
    once = all_float_source(fast_dp_source(tempo._COLUMN))
    with pytest.raises(ValueError):
        all_float_source(once)
    with pytest.raises(ValueError):
        all_float_source(tempo._COLUMN)  # needs the binary32 arithmetic layer under it
    with pytest.raises(ValueError):
        all_float_mpas(all_float_mpas(tempo._MPAS))
