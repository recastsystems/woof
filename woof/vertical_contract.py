"""Shared fail-closed contracts for explicit real-data eta grids.

The dynamical core has always allocated from ``RunConfig.nz``.  Native source
adapters and direct-WRF export additionally need to prove that their explicit
full-level eta array, serialized coordinate arrays, and declared WRF
``e_vert`` all describe that same grid.  Keeping those checks here prevents a
source-specific default profile from becoming a hidden dimension authority.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


WRF_REFERENCE_PRESSURE_PA = 100_000.0

MASS_COORDINATE_ARRAYS = (
    "znu", "dnw", "rdnw", "dn", "rdn", "fnp", "fnm",
    "c1h", "c2h", "c3h", "c4h",
)
INTERFACE_COORDINATE_ARRAYS = (
    "znw", "c1f", "c2f", "c3f", "c4f",
)


def planned_source_top_pressure_pa(source: str) -> float | None:
    """Read the planned atmosphere's top from its registry or level inventory.

    Extendable fetches can reach beyond their certified default inventory.
    Model-level products without a declared pressure bound, and caller-owned
    mappings, still need the decoded-data check during preparation.
    """
    import json

    from woof.source_adapters import get_source_adapter
    from woof.source_authorities import packaged_authorities

    adapter = get_source_adapter(source)
    if adapter.extendable_source_top_pa is not None:
        return float(adapter.extendable_source_top_pa)
    if adapter.certified_source_top_pa is not None:
        return float(adapter.certified_source_top_pa)
    if adapter.packaged_profile is None:
        return None
    top = _mapping_top_pa(
        source, json.loads(packaged_authorities(adapter.packaged_profile)["mapping"].read_text()))
    if top is None:
        return None
    # A composition whose companion extends the column above the primary top
    # (the RRFS ensemble member's 250 hPa column and its deterministic RRFS
    # companion, development branch): the composed atmosphere reaches the
    # companion's top, so the plan reads that one.  Breakage: every rrfs-ens
    # plan at the default 50 hPa model top was refused before acquisition.
    from woof.source_authorities import (packaged_composition,
                                          packaged_contributing_mappings)
    composition = packaged_composition(adapter.packaged_profile)
    contributing = packaged_contributing_mappings(adapter.packaged_profile)
    for binding in (composition.get("field_sources") or {}).values():
        companion = (binding or {}).get("companion") or {}
        path = contributing.get((binding or {}).get("mapping_role"))
        if companion.get("vertical_extension") != "above_primary_top" or path is None:
            continue
        reach = _mapping_top_pa(source, json.loads(Path(path).read_text()))
        if reach is not None:
            top = min(top, reach)
    return top


def _mapping_top_pa(source, mapping) -> float | None:
    """The lowest pressure (Pa) a mapping's pressure inventory reaches, or None."""
    vertical = mapping.get("coordinates", {}).get("vertical", {})
    if vertical.get("kind") != "pressure" or not vertical.get("levels"):
        return None
    units = str(vertical.get("units", "Pa")).strip().lower()
    if units in ("pa", "pascal", "pascals"):
        scale = 1.0
    elif units in ("hpa", "mb", "mbar", "millibar", "millibars"):
        scale = 100.0
    else:
        raise ValueError(f"source {source!r}: unknown pressure inventory units {units!r}")
    return min(float(level) for level in vertical["levels"]) * scale


def validate_config_source_top(raw: Mapping, *, context: str) -> str | None:
    """Refuse unsupported model tops before acquisition, decode or GPU sizing.

    This uses the same parsed configuration at the planning and check doors.
    Without a named source or declared pressure top there is no early verdict;
    the decoded-data check remains authoritative for the supplied atmosphere.
    """
    fetch = raw.get("fetch") or {}
    shared = raw.get("shared") or {}
    if not isinstance(fetch, dict) or not isinstance(shared, dict):
        return None  # the owning schema reports malformed tables
    source = fetch.get("source")
    p_top = shared.get("p_top", 0.0)
    if not source or isinstance(p_top, bool) or not isinstance(p_top, (int, float)):
        return None
    if not math.isfinite(p_top) or p_top <= 0.0:
        return None  # the vertical schema reports an invalid requested top
    source_top = planned_source_top_pressure_pa(source)
    if source_top is None:
        return None
    detail = (f"source {source!r} top is {source_top / 100:g} hPa "
              f"({source_top:g} Pa); requested p_top = {p_top:g} Pa "
              f"({p_top / 100:g} hPa)")
    if p_top < source_top:
        from woof.ingest.source_coverage import VerticalLadderRefusal

        raise VerticalLadderRefusal(
            f"{context}: model-top check REFUSED: {detail}. The model levels "
            "above the source atmosphere would require extrapolation past "
            "the available analysis, so this run cannot be initialized",
            remedy=f"raise [shared].p_top to {source_top:g} Pa or above, "
                   f"or choose a source reaching {p_top:g} Pa")
    return f"model-top check PASS: {detail}"


def validate_explicit_eta_grid(
    eta_levels,
    *,
    nz: int,
    p_top: float,
    context: str,
    source_top_pressure_pa: float | None = None,
) -> np.ndarray:
    """Validate and return one explicit ``nz + 1`` full-interface grid.

    ``source_top_pressure_pa`` is the smallest pressure available in a source
    product.  Supplying it refuses a requested model top above the source
    atmosphere instead of extrapolating an unsupported layer.
    """

    if isinstance(nz, bool) or not isinstance(nz, (int, np.integer)):
        raise TypeError(f"{context}: nz must be an integer, got {nz!r}")
    nz = int(nz)
    if nz < 1:
        raise ValueError(f"{context}: nz must be positive, got {nz}")
    eta = np.asarray(eta_levels, dtype=np.float64)
    if eta.shape != (nz + 1,):
        raise ValueError(
            f"{context}: explicit eta_levels has shape {eta.shape}; "
            f"nz={nz} requires ({nz + 1},) interfaces (WRF e_vert={nz + 1})"
        )
    if not np.isfinite(eta).all():
        raise ValueError(f"{context}: eta_levels must all be finite")
    if eta[0] != 1.0 or eta[-1] != 0.0:
        raise ValueError(
            f"{context}: eta_levels must run exactly from 1.0 to 0.0, "
            f"got {eta[0]!r} .. {eta[-1]!r}"
        )
    if not np.all(np.diff(eta) < 0.0):
        raise ValueError(f"{context}: eta_levels must be strictly decreasing")

    p_top = float(p_top)
    if not math.isfinite(p_top) or not 0.0 < p_top < WRF_REFERENCE_PRESSURE_PA:
        raise ValueError(
            f"{context}: p_top must be finite and inside "
            f"(0, {WRF_REFERENCE_PRESSURE_PA:g}) Pa, got {p_top!r}"
        )
    if source_top_pressure_pa is not None:
        source_top = float(source_top_pressure_pa)
        if not math.isfinite(source_top) or source_top <= 0.0:
            raise ValueError(
                f"{context}: source top pressure must be finite and positive"
            )
        if source_top > p_top:
            # NAME THE BREAKAGE AND THE WAY OUT.  "A stops at X, B is Y"
            # is two numbers and no consequence: the operator who meets it
            # cannot tell whether the grid is wrong, the source is wrong,
            # or the run is merely unusual.  What actually breaks is that
            # every mass level above the source top has no data under it.
            raise ValueError(
                f"{context}: source atmosphere stops at {source_top:g} Pa "
                f"but requested p_top is {p_top:g} Pa. The levels above "
                f"{source_top:g} Pa would be filled by extrapolating past "
                f"the top of the analysis, so the model top has to stay "
                f"inside it: raise p_top to {source_top:g} Pa or above, or "
                f"supply a source reaching {p_top:g} Pa"
            )
    return eta.copy()


def expected_coordinate_shapes(nz: int) -> dict[str, tuple[int, ...]]:
    """Return every serialized coordinate shape derived from ``nz``."""

    return {
        **{name: (int(nz),) for name in MASS_COORDINATE_ARRAYS},
        **{name: (int(nz) + 1,) for name in INTERFACE_COORDINATE_ARRAYS},
    }


def validate_coordinate_shapes(
    shapes: Mapping[str, Sequence[int]], *, nz: int, context: str
) -> None:
    """Require the complete prepared/restart coordinate shape inventory."""

    expected = expected_coordinate_shapes(nz)
    missing = sorted(set(expected) - set(shapes))
    if missing:
        raise ValueError(f"{context}: missing vertical coordinate arrays {missing}")
    drift = {
        name: {"actual": tuple(shapes[name]), "expected": shape}
        for name, shape in expected.items()
        if tuple(shapes[name]) != shape
    }
    if drift:
        raise ValueError(f"{context}: vertical coordinate shape drift: {drift}")


def explicit_vertical_from_wrf_namelist(
    path: str | Path,
    *,
    expected_nz: int | None = None,
    context: str = "WRF namelist",
):
    """Parse one shared explicit eta grid from ``namelist.input``.

    WRF repeats some domain values.  Repeated ``e_vert`` and
    ``p_top_requested`` entries are accepted only when uniform because the
    current native nesting contract shares one vertical grid.  The return
    value is the canonical :class:`woof.experiment.VerticalConfig` used by
    the initializer, rather than a second source-specific vertical type.
    """

    from woof.experiment import VerticalConfig
    from woof.namelist_import import parse_namelist

    sections = parse_namelist(path)
    try:
        domains = sections["domains"]
        dynamics = sections["dynamics"]
    except KeyError as exc:
        raise ValueError(
            f"{context}: namelist must contain &domains and &dynamics") from exc

    try:
        raw_eta = domains["eta_levels"]
        raw_p_top = domains["p_top_requested"]
        raw_hybrid = dynamics["hybrid_opt"]
        raw_etac = dynamics["etac"]
    except KeyError as exc:
        raise ValueError(
            f"{context}: explicit eta_levels, p_top_requested, hybrid_opt, "
            "and etac are required") from exc

    def uniform_numeric(values, label):
        if not values:
            raise ValueError(f"{context}: {label} has no values")
        converted = [float(value) for value in values]
        if any(not math.isfinite(value) for value in converted):
            raise ValueError(f"{context}: {label} must be finite")
        if any(value != converted[0] for value in converted[1:]):
            raise ValueError(
                f"{context}: shared vertical grid requires uniform {label}, "
                f"got {converted}")
        return converted[0]

    eta = tuple(float(value) for value in raw_eta)
    e_vert_values = domains.get("e_vert")
    if e_vert_values:
        e_vert_float = uniform_numeric(e_vert_values, "e_vert")
        if not e_vert_float.is_integer():
            raise ValueError(f"{context}: e_vert must be an integer")
        e_vert = int(e_vert_float)
    else:
        e_vert = len(eta)
    nz = e_vert - 1
    if expected_nz is not None and nz != int(expected_nz):
        raise ValueError(
            f"{context}: target nz={expected_nz} but namelist "
            f"e_vert={e_vert} implies nz={nz}")
    p_top = uniform_numeric(raw_p_top, "p_top_requested")
    hybrid_float = uniform_numeric(raw_hybrid, "hybrid_opt")
    if not hybrid_float.is_integer():
        raise ValueError(f"{context}: hybrid_opt must be an integer")
    hybrid_opt = int(hybrid_float)
    etac = uniform_numeric(raw_etac, "etac")
    validate_explicit_eta_grid(
        eta, nz=nz, p_top=p_top, context=context)
    return VerticalConfig(
        eta_levels=eta,
        p_top=p_top,
        hybrid_opt=hybrid_opt,
        etac=etac,
    )


__all__ = [
    "INTERFACE_COORDINATE_ARRAYS",
    "MASS_COORDINATE_ARRAYS",
    "WRF_REFERENCE_PRESSURE_PA",
    "expected_coordinate_shapes",
    "explicit_vertical_from_wrf_namelist",
    "validate_coordinate_shapes",
    "validate_explicit_eta_grid",
]
