"""Validated scalar VMR operands shared by radiation adapters.

These are representation and implemented-operand checks, not a claim that a
coefficient table has been validated for every admitted concentration.
"""
from collections.abc import Mapping
import math

import numpy as np

RFMIP_GAS_NAMES = {
    "co2": "carbon_dioxide", "n2o": "nitrous_oxide",
    "co": "carbon_monoxide", "ch4": "methane", "o2": "oxygen",
    "n2": "nitrogen", "ccl4": "carbon_tetrachloride",
    "cfc11": "cfc11", "cfc12": "cfc12", "cfc22": "hcfc22",
    "hfc143a": "hfc143a", "hfc125": "hfc125", "hfc23": "hfc23",
    "hfc32": "hfc32", "hfc134a": "hfc134a", "cf4": "cf4",
}
CLASSIC_GASES = frozenset(("co2", "n2o", "ch4"))
LEGACY_SW_GASES = frozenset(("co2", "ch4", "n2o", "o2"))
# WRF's SW wrapper carries N2O through SETCOEF, but SPCVRT calls
# TAUMOL_SW with H2O/CO2/CH4/O2/O3/air only. It has no N2O absorption.
LEGACY_SW_ABSORPTION_GASES = LEGACY_SW_GASES - {"n2o"}
LEGACY_LW_GASES = LEGACY_SW_GASES | {"cfc11", "cfc12", "cfc22", "ccl4"}


class FileGases(dict):
    """Final mole fractions read from a gas file (WRF ``ghg_input = 1``,
    woof.core.camtr_gases).  Zero is a legal file value (the CAMtr table
    has no CFCs before 1944-1945) and reaches the solver as WRF passes it;
    every other override check still applies."""


def validate_trace_gas_overrides(overrides, *, supported=None, consumer=None):
    """Own positive scalar mole fractions representable by the FP32 solvers.

    Zero remains outside the existing explicit-override contract. The old
    blanket 1e-2 trace-gas ceiling excluded ordinary oxygen; a mole fraction
    instead has the representation bound (0, 1]. Default year policies are
    selected by the caller and never pass through this override function.
    """
    if overrides is None:
        return {}
    if not isinstance(overrides, Mapping):
        raise TypeError("trace-gas override must be a mapping or None")
    unknown = sorted(set(overrides) - set(RFMIP_GAS_NAMES), key=str)
    if unknown:
        raise ValueError(f"unknown trace gas(es) {unknown} in override; "
                         f"known well-mixed gases: {sorted(RFMIP_GAS_NAMES)}")
    result = {}
    for gas, raw in overrides.items():
        if isinstance(raw, (bool, np.bool_)):
            raise ValueError(f"trace-gas override[{gas!r}] must be a positive scalar mole fraction")
        value = float(raw)
        if isinstance(overrides, FileGases) and value == 0.0:
            result[gas] = value
            continue
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError(f"trace-gas override[{gas!r}] = {value!r} must be a "
                             "finite mole fraction in (0, 1]")
        if np.float32(value) == 0:
            raise ValueError(f"trace-gas override[{gas!r}] = {value!r} underflows "
                             "the solver's positive FP32 VMR representation")
        result[gas] = value
    if isinstance(overrides, FileGases):
        result = FileGases(result)
    if supported is not None:
        unused = sorted(set(result) - set(supported))
        if unused:
            raise ValueError(f"{consumer or 'selected radiation'} has no absorption "
                             f"operand for trace gas(es) {unused}; implemented "
                             f"operands: {sorted(supported)}")
    return result


def trace_gas_subset(overrides, supported):
    """Pass only a selected spectrum's operands after composition validation."""
    subset = {name: value for name, value in (overrides or {}).items()
              if name in supported}
    return FileGases(subset) if isinstance(overrides, FileGases) else subset
