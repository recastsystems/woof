"""Exact mode drives each scalar slot from the boundary species native applies there (runtime_v842.native_lbc_scalar_order).

THE BREAKAGE THESE PREVENT: native RRFS-MPAS copies lbc_scalars into scalars by position and its Registry declares the two
arrays in different orders (scalars ni, nr; lbc_scalars lbc_nr, lbc_ni), so an exact limited-area run that drove each
species by name differed from the native reference in every specified-zone nr and ni word from the first step.
"""
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import woof.hex
from woof.hex.mpasphys.runtime_v842 import native_lbc_scalar_order

DATA = Path(woof.hex.__file__).parent / "data"


def _layout(scalars, lbc):
    return SimpleNamespace(fields=[{"pool": "state", "name": "scalars", "components": list(scalars)},
                                   {"pool": "lbc", "name": "lbc_scalars", "components": list(lbc)}])


def _order(registry, array):
    root = ET.parse(registry).getroot()
    for node in root.iter("var_array"):
        if node.get("name") == array and node.find("var") is not None:
            return [v.get("name") for v in node.findall("var")]
    raise AssertionError(array)


NAMES = ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "nc")
LBC = ("lbc_qv", "lbc_qc", "lbc_qr", "lbc_qi", "lbc_qs", "lbc_qg", "lbc_nr", "lbc_ni", "lbc_nc")


def test_exact_mode_takes_the_native_positional_order():
    names, swapped = native_lbc_scalar_order(_layout(NAMES, LBC), SimpleNamespace(mode="exact"), NAMES)
    assert names == LBC
    assert swapped == {"ni": "nr", "nr": "ni"}


def test_correct_mode_drives_each_species_by_name():
    names, swapped = native_lbc_scalar_order(_layout(NAMES, LBC), SimpleNamespace(mode="correct"), NAMES)
    assert names == tuple("lbc_" + n for n in NAMES) and swapped is None


def test_same_order_or_other_sets_change_nothing():
    same = tuple("lbc_" + n for n in NAMES)
    assert native_lbc_scalar_order(_layout(NAMES, same), SimpleNamespace(mode="exact"), NAMES) == (same, None)
    assert native_lbc_scalar_order(_layout(NAMES, LBC[:-1]), SimpleNamespace(mode="exact"), NAMES) == (same, None)


def test_the_packaged_registries_hold_the_orders_this_relies_on():
    for fork in ("mpasphys-framework-rrfs", "mpasphys-framework-rrfs-44c0849"):
        registry = DATA / fork / "src/core_atmosphere/Registry.xml"
        scalars, lbc = _order(registry, "scalars"), _order(registry, "lbc_scalars")
        assert scalars.index("ni") < scalars.index("nr")
        assert lbc.index("lbc_nr") < lbc.index("lbc_ni")
    upstream = DATA / "mpasphys-framework/src/core_atmosphere/Registry.xml"
    scalars, lbc = _order(upstream, "scalars"), _order(upstream, "lbc_scalars")
    assert [s for s in scalars if "lbc_" + s in lbc] == [l[4:] for l in lbc if l[4:] in scalars]
