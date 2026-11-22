# SPDX-License-Identifier: Apache-2.0
"""Run the ordinary hex forward at chosen values of the differentiable physics constants (default off).

``WOOF_HEX_AD_PARAMS`` names a parameter file in the window runner's ``--params`` format (what
:func:`woof.hex.ad.params.carrier_document` writes: ``{"params": [{"name", "p0_bits"}, ...]}``, e.g. constants fitted
with ``woof hex param-grad``). When it is set, each named constant's literal in its scheme's CUDA source string is
replaced by the float32 value as a hexadecimal float literal, at the same occurrence the parameter carrier reads
(:mod:`woof.hex.ad.carrier`), so the forecast does the same float operation on the same value the carrier forward does.

Unset (the default), :func:`patch_module` returns before reading anything: every source string, image and output word
is the shipped one.

The constants are carried for the scheme versions whose sources contain them (MYNN-SFC v6.0.0 for the surface-layer
constants; GSL drag; MYNN-EDMF 44c0849, c942418 and v6.1.1 for B1). A module of another version that is asked for one
of its family's constants refuses (:func:`refuse_family`), so a run never goes ahead silently at the shipped value.
"""
from __future__ import annotations

import json
import os
import struct

ENV = "WOOF_HEX_AD_PARAMS"


def _values() -> dict[str, float] | None:
    path = os.environ.get(ENV)
    if not path:
        return None
    from woof.hex.ad.params import BY_NAME
    with open(path, encoding="utf-8") as stream:
        doc = json.load(stream)
    out = {}
    for row in doc["params"]:
        bits = int(str(row["p0_bits"]), 16)
        out[row["name"]] = struct.unpack("<f", struct.pack("<I", bits))[0]
    unknown = sorted(set(out) - set(BY_NAME))
    if unknown:
        raise ValueError(f"{ENV}: unknown physics constants {unknown}")
    return out


def _literal(value: float) -> str:
    return float.hex(value) + "f"


def patch_module(namespace: dict, family: str) -> None:
    """Replace this family's requested constants in the module's upper-case source strings (no-op when unset).

    Each site must occur exactly once over the module's string constants, or the import fails: a source that no longer
    carries the pattern would otherwise run silently at the shipped constant.
    """
    values = _values()
    if values is None:
        return
    from woof.hex.ad.params import BY_NAME
    for name, value in values.items():
        constant = BY_NAME[name]
        if constant.family != family:
            continue
        for site in constant.sites:
            hits = [k for k, v in namespace.items() if isinstance(v, str) and k.isupper() and site in v]
            count = sum(namespace[k].count(site) for k in hits)
            if count != 1:
                raise RuntimeError(f"{ENV}: {name}: source pattern found {count} times in this {family} module, expected 1")
            if constant.literal in site:
                new = site.replace(constant.literal, _literal(value))
            else:
                # MYNN-EDMF B1: the qke-prediction occurrence reads the named constant b1, not the literal.
                new = site.replace("r6_mul(b1, 0.25f)", f"r6_mul({_literal(value)}, 0.25f)")
                if new == site:
                    raise RuntimeError(f"{ENV}: {name}: no replaceable token in its site")
            namespace[hits[0]] = namespace[hits[0]].replace(site, new)


def refuse_family(family: str, what: str) -> None:
    """For a scheme version that does not carry its family's constants: refuse when the run asks for one of them."""
    values = _values()
    if values is None:
        return
    from woof.hex.ad.params import BY_NAME
    asked = sorted(n for n in values if BY_NAME[n].family == family)
    if asked:
        raise RuntimeError(f"{ENV}: {', '.join(asked)} are carried for the pinned scheme version, not {what}; "
                           f"run with that version or drop them from the file")
