"""The direct exporter writes real.exe's own CF1-3, CFN/CFN1 and MF_VX_INV words.

What breaks without this: wrf.exe reads all six from wrfinput and never
recomputes them.  The exporter left CF1-3/CFN/CFN1 to a template prototype
(another vertical grid's values) and rounded MF_VX_INV from a float64
reciprocal, so wrf.exe 4.6.1 ran the UAE 9 km d01 with a foreign model-top
extrapolation and, with every physics scheme off, differed from strict WOOF
in 495,278 U words after one step (2026-10-29).  The expected words
are gfortran's (-O2 -fno-fast-math -ffp-contract=off, the pinned strict
recipe) evaluation of module_initialize_real.F:3734-3742 on that column.
"""
import numpy as np

from woof.wrf_direct import _real_exe_extrapolation_constants


def _word(hex_word):
    return np.array([int(hex_word, 16)], dtype=np.uint32).view(np.float32)[0]


def test_extrapolation_constants_are_real_exe_words():
    nz = 44
    dn = np.zeros(nz, np.float32)
    dnw = np.full(nz, _word("BBCAAE00"), np.float32)
    fnp = np.zeros(nz, np.float32)
    fnm = np.zeros(nz, np.float32)
    dn[1], dn[2], dn[-1] = _word("BBD41FC0"), _word("BBE7D240"), _word("BC1D26AA")
    dnw[0], dnw[-1] = _word("BBCAAE00"), _word("BC10F745")
    fnp[1], fnm[1] = _word("3F05B2EC"), _word("3EF49A28")
    got = _real_exe_extrapolation_constants(dn, dnw, fnp, fnm)
    words = {name: format(int(np.asarray(value, np.float32).view(np.uint32)), "08X")
             for name, value in got.items()}
    assert words == {"CF1": "3FF79683", "CF2": "BFAD0F5F", "CF3": "3ED5E36F",
                     "CFN": "3FBB099F", "CFN1": "BEEC267D"}


def test_mf_vx_inv_is_the_fp32_reciprocal_real_exe_stores():
    import inspect

    from woof import wrf_direct
    source = inspect.getsource(wrf_direct._wrfinput_fields)
    assert ('"MF_VX_INV": np.float32(1.0) / np.asarray(static["MAPFAC_V"], '
            'dtype=np.float32)') in source
    msfv = _word("3F7AF1EA")
    # real.exe's word for the stored factor; the old writer's reciprocal of
    # the unrounded float64 factor wrote 3F829413 here (UAE d01, 4,560 of
    # 12,120 words one ULP off).
    assert format(int((np.float32(1.0) / msfv).view(np.uint32)), "08X") == "3F829414"
