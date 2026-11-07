"""Goddard GCE (mp_physics=97) constant table: the host half of the port.

CPU-only.  The column kernel reads one packed float32 table; these tests
hold the table to the values the WRF 4.6.1 referee object (the scalar
wrf.exe build's phys/module_mp_gsfcgce.o) stores, and hold the kernel's
slot enum to the table's order.  The device half is graded against the
Fortran by tools/gsfcgce_wrf461_oracle.
"""

from __future__ import annotations

import pathlib
import re

import numpy as np
import pytest

from woof.core import gsfcgce_constants as g

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Immediates the referee object's consat_s stores (objdump of
#: module_mp_gsfcgce.o, gfortran 13.3.0 -O2 -fno-fast-math
#: -ffp-contract=off): the values gcc folded at compile time, including the
#: correctly rounded constant powers.  Shared by both gsfcgce_hail values.
REFEREE_IMMEDIATES = {
    "zrc": 0x3F354258, "rn1": 0x282955D9, "rn8": 0x3E015CE6,
    "rn10": 0x3F80ADFD, "rn10a": 0x571E3C72, "rn10b": 0x4B1CFBC2,
    "rn10c": 0x4B9BC9A9, "rn11": 0x2FA5AB80, "rn11a": 0x3C4DA28A,
    "rn17b": 0x4B9FC270, "rn17c": 0x4C1FB8AC, "rn18": 0x3A4EFB0A,
    "rn18a": 0x3F28F5C3, "rn21": 0x3233B391, "rn23": 0x3F00ADFD,
    "rn23b": 0x56F6578E, "rn30a": 0x516B54F7, "rn30b": 0x4B1CFBC2,
    "rn30c": 0x4B9BC9A9, "rn31": 0x233877AA, "rn32": 0x3CA8E714,
    "rnn30a": 0x514FA050, "rn33": 0x3F23D70A, "rn331": 0x3F266666,
    "cp": 0x4B1932C0, "avc": 0x451BA0A3, "afc": 0x43A622AD,
    "asc": 0x453064F9, "beta": 0xBEEB851F, "tns": 0x3E23D70A,
    "roqs": 0x3DCCCCCD, "as_": 0x429D428F, "bs": 0x3DE147AE,
    "bsh": 0x3D6147AE, "bsq": 0x3CE147AE, "bwq": 0x3E4CCCCD,
}


def _bits(x) -> int:
    return int(np.asarray(x, dtype=np.float32).view(np.uint32))


@pytest.mark.parametrize("ihail", [0, 1])
def test_table_matches_the_referee_objects_folded_constants(ihail):
    scalars = g.constants(ihail)["scalars"]
    wrong = {k: (hex(_bits(scalars[k])), hex(v))
             for k, v in REFEREE_IMMEDIATES.items() if _bits(scalars[k]) != v}
    assert not wrong, wrong


def test_table_layout_matches_the_kernel_enum():
    src = (ROOT / "woof/core/kernels/gsfcgce.cu").read_text(encoding="utf-8")
    body = re.search(r"enum GceSlot \{(.*?)\};", src, re.S).group(1)
    names = [n.strip() for n in body.replace("\n", " ").split(",")
             if n.strip()]
    assert names[-1] == "S_NSCALAR"
    expected = ["S_" + s.rstrip("_").upper() for s in g.SCALAR_SLOTS]
    assert names[:-1] == expected
    offsets = dict(re.findall(r"#define (T_[A-Z0-9]+) +\(S_NSCALAR \+ (\d+)\)",
                              src))
    for n, name in enumerate(g.ARRAY_SLOTS):
        assert int(offsets["T_" + name.upper()]) == n * g.ARRAY_LEN
    for ihail in (0, 1):
        t = g.table(ihail)
        assert t.dtype == np.float32 and t.shape == (g.TABLE_LEN,)
        assert not t.flags.writeable


def test_the_hail_switch_moves_only_the_dense_ice_constants():
    a, b = g.constants(0)["scalars"], g.constants(1)["scalars"]
    moved = {k for k in g.SCALAR_SLOTS if _bits(a[k]) != _bits(b[k])}
    dense = {"ag", "bg", "bgh", "bgq", "tng", "roqg", "zgc", "vgc", "rn9",
             "rn14", "rn15", "rn15a", "rn16", "rn17", "rn17a", "rn19",
             "rn19a", "rn20", "rn20b", "rn30", "rnn192"}
    assert moved and moved <= dense, sorted(moved - dense)


def test_fall_flux_gammas_are_the_runtime_ggamma():
    # ggamma's own fixed points: Gamma(4.5) = 11.6317..., and the B series
    # is a fit good to a few float32 ulps, not an exact gamma.
    assert abs(float(g.ggamma(np.float32(4.5))) - 11.631728396567448) < 2e-5
    assert abs(float(g.gammagce(np.float32(3.8))) - 4.694174205740421) < 2e-5


def test_declared_divergences_are_the_oracle_fix_arm():
    import importlib.util
    path = ROOT / "tools/gsfcgce_wrf461_oracle/make_fixed_source.py"
    spec = importlib.util.spec_from_file_location("gce_fix", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Only witnessed undefined reads are repaired; deterministic ice2=2
    # graupel fallout follows native WRF unchanged.
    assert len(g.DECLARED_DIVERGENCES) == 2
    assert {r[0] for r in mod.REPAIRS} == {
        "warm-transport-coefficients", "refl-2ice-snow-qg1d"}
    assert g.refl_species(1) == (1, 0)
