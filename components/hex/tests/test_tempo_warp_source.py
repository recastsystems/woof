"""TEMPO's warp shape carries the column transcription unchanged (CPU tier).

The warp kernel (TEMPO_WARP 1, woof.hex.mpasphys.rrfs.tempo_v303._WARP) runs one column per warp with level k on
lane (k-1)%32. Five routines whose levels read each other (four fall speeds, sedimentation, cloud fall) carry a
warp block beside their serial text. THE BREAKAGE THIS PREVENTS: an edit made to one copy and not the other,
which would silently make fast and strict forecasts (both run the warp shape) leave the native transcription.
Every computing statement of each copy must be a statement of the other, apart from the named
synchronisation/chain/maximum bookkeeping. The word-for-word proof on real inputs is the module replay in
tools/tempo_rrfs_replay (strict: 0 of 132,977,368 words differ on each of three captured hrrrv5 calls).
"""
import re

import pytest

from woof.hex.mpasphys.rrfs import tempo_v303 as tempo

ROUTINES = ("rain_fallspeed", "graupel_fallspeed", "snow_fallspeed", "ice_fallspeed", "cloud_fallspeed",
            "sedimentation")
# Statements only the warp copy has: synchronisation, ballots, chain ends, warp maxima, the lead lane.
WARP_ONLY = ("for (", "TEMPO_", "tempo_next_active", "tempo_warp_", "__shfl", "rows", "ktop", "substeps", "[j]",
             "const int j", "return", "tempo_active_count", "tempo_active_level",
             "int rank=TEMPO_LANE", "rank+=32")
# Statements only the serial copy has: the chain step and the in-place maxima.
ASSIGNMENT = re.compile(r"(?<![<>=!])=(?!=)")
SERIAL_ONLY = ("ktop_sedi", "substeps_sedi", "vt[k + 1]", "vtn[k + 1]", "int k = nz", "for (", "return")


def _body(name):
    text = tempo._COLUMN
    start = text.index(f"inline void {name}(")
    end = text.index("\n}\n", start)
    return text[start:end]


def _split(body):
    warp = re.search(r"#if TEMPO_WARP\n(.*?)\n#(?:else|endif)\n", body, re.S)
    assert warp, "warp block missing"
    # The declarations before the block are shared; the serial copy is the text after it.
    return warp.group(1), body[warp.end():]


def _statements(text, skip):
    text = re.sub(r"//[^\n]*", "", text)
    out = set()
    for piece in re.sub(r"\s+", " ", text).split(";"):
        piece = re.split(r"[{}]", piece)[-1].strip()
        piece = re.sub(r"^(if|else if) \(.*?\) ", "", piece)
        piece = re.sub(r"^(?:--|\+\+)\w+\) ", "", piece)  # the tail of a for (...; ...; --k) header
        piece = re.sub(r"^TEMPO_LEVELS\(\w+, ?[\w +-]+\) ", "", piece)  # a lane-strided level loop
        if ASSIGNMENT.search(piece) and not any(token in piece for token in skip):
            out.add(piece)
    return out


@pytest.mark.parametrize("name", ROUTINES)
def test_warp_and_serial_copies_compute_the_same_statements(name):
    warp, serial = _split(_body(name))
    warp_statements = _statements(warp, WARP_ONLY)
    serial_statements = _statements(serial, SERIAL_ONLY)
    assert warp_statements, name
    assert warp_statements <= serial_statements, sorted(warp_statements - serial_statements)
    assert serial_statements <= warp_statements, sorted(serial_statements - warp_statements)


def test_column_shape_is_the_default_and_the_warp_shape_is_named():
    column = tempo.cuda_source("// maths\n", max_levels=55, numerics="strict")
    warp = tempo.warp_source("// maths\n", max_levels=55, numerics="strict")
    assert "#define TEMPO_WARP 1" not in column and "tempo_columns" in column and "tempo_warp_steps" not in column
    assert "#define TEMPO_WARP 1" in warp and "tempo_warp_steps" in warp and "void tempo_columns" not in warp
    # Both shapes compile the same routine text and the same MPAS slice kernels.
    assert tempo._COLUMN in column and tempo._COLUMN in warp and tempo._MPAS in column and tempo._MPAS in warp
