"""rrfs/tempo3km-a1: TEMPO's binary64 arithmetic narrows to binary32 in fast numerics only (CPU, source level)."""
import re

import pytest

from woof.hex.mpasphys.rrfs import tempo_v303 as tempo
from woof.hex.mpasphys.rrfs.tempo_fastdp_v303 import fast_dp_source

MATHS = "// device maths\n"


def test_strict_column_is_the_proved_source():
    assert tempo._column_source("strict") is tempo._COLUMN
    assert tempo._COLUMN in tempo.warp_source(MATHS, max_levels=59, numerics="strict")
    assert tempo._COLUMN in tempo.cuda_source(MATHS, max_levels=59, numerics="strict")


def test_fast_column_keeps_binary64_only_in_storage_and_maths_shim():
    from woof.hex.mpasphys.rrfs.tempo_allfloat_v303 import all_float_source
    composed = tempo._column_source("fast")
    assert composed in tempo.warp_source(MATHS, max_levels=59, numerics="fast")
    # rrfs/tempo-yolo: fast = tiny mass over the all-binary32 layer (blob, parameters, intrinsics) over the binary32
    # arithmetic (tempo3km-a1) over the fixed-speed sedimentation; the storage-only property of the a1 layer is
    # checked here, the all-float property in test_tempo_allfloat_v303.
    from woof.hex.mpasphys.rrfs.tempo_fastsed_v303 import fast_sed_source
    from woof.hex.mpasphys.rrfs.tempo_tinymass_v303 import tiny_mass_source
    fast = fast_dp_source(fast_sed_source(tempo._COLUMN))
    assert composed == tiny_mass_source(all_float_source(fast))
    left = [line.strip() for line in fast.splitlines() if re.search(r"\bdouble\b", line)]
    allowed = ("#define TEMPO_TX_MEMBER", "TEMPO_HD int tnint(double", "TEMPO_HD double tempo_", "struct TempoPowBase {",
               "TEMPO_HD TempoPowBase tempo_pow_base(double", "TEMPO_HD double tempo_pow_with(", "const double ehi",
               "const double elo")
    assert left and all(line.startswith(allowed) for line in left), [l for l in left if not l.startswith(allowed)]
    assert "using TK_tcg_racg = float;" in fast and "#define TEMPO_TT_MEMBER(name) float name" in fast
    body = fast[fast.index("TEMPO_HD float calc_rslf"):]
    assert "TempoPowBase " not in body and "tempo_pow_with(" not in body
    assert not re.search(r"(?<![\w.])(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?(?![\w.])", body)


def test_fast_dp_refuses_drift_and_reapplication():
    once = fast_dp_source(tempo._COLUMN)
    with pytest.raises(ValueError):
        fast_dp_source(once)
    with pytest.raises(ValueError):
        fast_dp_source(tempo._COLUMN.replace("module_mp_tempo_utils.F90:256-282", "module_mp_tempo_utils.F90:256-283"))


def test_fast_sedimentation_is_fast_warp_only_and_refuses_drift():
    from woof.hex.mpasphys.rrfs import tempo_fastsed_v303 as sed
    once = sed.fast_sed_source(tempo._COLUMN)
    assert "tempo_fsed<3>" in once and "tempo_fsed" not in tempo._column_source("strict")
    # the serial column shape keeps the native sedimentation text inside the #else branch
    native = tempo._COLUMN[tempo._COLUMN.index(sed._START):tempo._COLUMN.index(sed._END)]
    assert native in once
    with pytest.raises(ValueError):
        sed.fast_sed_source(once)
    with pytest.raises(ValueError):
        sed.fast_sed_source(tempo._COLUMN.replace(sed._START, "  // sedimentation (rain)" + chr(10)))
