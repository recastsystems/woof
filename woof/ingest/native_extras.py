"""Numeric gate declarations for optional native-level records.

Short names are transport labels only. Section 4 selectors are the decode
authority; a requested field requires fifty levels and manifest-bound bytes.
"""

EXTRA_GATES = {
    "MASSDEN": "PASS discipline=0 category=20 parameter=0 level_type=105",
    "PMTF": "PASS discipline=0 category=13 parameter=193 level_type=105",
    "PMTC": "PASS discipline=0 category=13 parameter=192 level_type=105",
    # The HRRR start's request-only number concentrations (fix/hrrr-start-fields): the decoder's QNC, QNI and QNR
    # (hrrr_grib2_bridge EXTRA_SPECS), which the public index spells NCONCD, NCCICE and SPNCR.
    "QNC": "PASS discipline=0 category=6 parameter=28 level_type=105",
    "QNI": "PASS discipline=0 category=6 parameter=29 level_type=105",
    "QNR": "PASS discipline=0 category=1 parameter=100 level_type=105",
}

#: Request-only single-level records (one record, not fifty levels): the decoder's RH2 (EXTRA_SURFACE_SPECS) is
#: HRRR's 2 m relative humidity, the index's ``RH:2 m above ground``.
SURFACE_EXTRA_GATES = {
    "RH2": "PASS discipline=0 category=1 parameter=1 level_type=103 level_value=2",
}


def requested_extra_fields(names=()):
    names = tuple(names)
    known = EXTRA_GATES.keys() | SURFACE_EXTRA_GATES.keys()
    if len(set(names)) != len(names) or any(name not in known for name in names):
        raise ValueError("unknown or duplicate native extra; prevents decoding a wrong tracer or duplicated inventory")
    return tuple(sorted(names))


def gate_extra_fields(gate):
    declared = gate.get("extra_fields")
    names = requested_extra_fields(declared.split(",")) if declared is not None else ()
    actual = {key for key in gate if key.startswith("extra_") and key != "extra_fields"}
    if actual != {"extra_" + name for name in names}:
        raise ValueError("native extra declarations differ from their gate rows; prevents unbound tracer decoding")
    for name in names:
        if name not in EXTRA_GATES:
            # A gate row counts 50 records per extra (hrrr.py, seal_hrrr_native_bridge.py); a single-level record
            # declared there would miscount its inventory.
            raise ValueError(f"native extra {name} is a single-level record; gate rows declare hybrid extras only")
        if gate["extra_" + name] != EXTRA_GATES[name]:
            raise ValueError(f"native extra {name} has an unknown code; prevents mapping the wrong physical quantity")
    return names


def map_boundary_rows(source_fields, rows, source_row, plan, *, cycle=None, cpu_bridge=None):
    """Return row-keyed source tracers mapped bilinearly to the mass grid.

    Boundary row conversions run on native source levels using their own P
    and T before interpolation. Only available fields on this source are
    selected; absent rows are left to the ordered source resolver.

    ``cycle`` is the source files' cycle.  A variable whose units changed
    between cycles (its ``units_epochs`` rows) is first brought to its
    declared units by that epoch's scale, folded into the row's weights;
    such a variable read without a cycle is refused
    (:func:`woof.chem_conversions.units_epoch_scale`).
    """
    from woof.chem_conversions import (
        convert_source, units_epoch_scale, weighted_source_fields)
    output = {}
    for row in rows:
        for boundary in row.boundary:
            if boundary["source"] != source_row.name:
                continue
            names = boundary["fields"]
            if not all(name in source_fields for name in names):
                continue
            weights = [float(weight) * units_epoch_scale(
                           source_row.variables[name], cycle,
                           source=f"source {source_row.name!r}", variable=name)
                       for name, weight in zip(names, boundary["weights"])]
            combined = weighted_source_fields(
                (source_fields[name] for name in names), weights,
                cpu_bridge=cpu_bridge)
            parameters = [source_row.variables[name].get("conversion_parameters") for name in names]
            if any(value != parameters[0] for value in parameters):
                raise ValueError("boundary terms use different density conventions; prevents applying one conversion to incompatible source units")
            value = convert_source(boundary["conversion"], combined,
                                   source_fields, parameters=parameters[0], cpu_bridge=cpu_bridge)
            output[row.name] = plan.apply(value, method="bilinear")
            break
    return output


def interpolate_boundary_tracers(fields, vertical_plan, ordered_levels,
                                 zero_surface, nz):
    """Map IC or one boundary time's row-keyed tracers onto model eta levels.

    Input fields: FP32 (source_level, ny, nx), already in row units on the
    target horizontal mass grid. ``ordered_levels`` uses initialize_real's
    pressure ordering; ``vertical_plan`` uses that call's source dry pressure,
    surface dry pressure and target eta dry pressure. Output: row-keyed FP32
    (nz, ny, nx), mass centered, unstaggered, in the same units (ug/kg dry
    air for the mass-density conversion). No state or boundary tables are
    written. IC and each boundary time use the same contract.

    WRF module_initialize_real.F:1862-1982 uses linear Q interpolation in
    log pressure with a surface pseudo-level. Native source has no surface
    tracer: supplied zero surface, constant extrapolation, vboundb=nz+1.
    This is the existing hydrometeor treatment, including its below-source
    surface ramp; it is not a claim of zero extrapolation at every altitude.
    """
    return {name: vertical_plan.apply(ordered_levels(value), zero_surface,
                interp_in_logp=True, extrap="constant", vboundb=nz + 1,
                values_are_finite=True) for name, value in fields.items()}


ABSENT_BOUNDARY_RECEIPT = "smoke from outside the domain is absent"


def choose_boundary_source(boundaries, available_sources):
    """Choose the first available boundary row, or return the absent receipt.

    Orchestration removes sources that fail coverage before calling. A None
    selection means zero IC and the species row's default inflow. This helper
    fetches nothing and does not write state, cache or lateral boundary tables.
    """
    for boundary in boundaries:
        if boundary["source"] in available_sources:
            return boundary, None
    return None, ABSENT_BOUNDARY_RECEIPT


def inside_native_grid(x, y, *, nx, ny):
    """Pure coverage check for zero-based source fractional mass indices.

    Matches the Lambert plan's floor(index)-1 through floor(index)+2 halo.
    Callers project all target points, including boundary strips, first.
    """
    import math
    pairs = tuple(zip(x, y, strict=True))
    return bool(pairs) and all(math.isfinite(a) and math.isfinite(b)
        and math.floor(a) >= 1 and math.floor(a) + 2 < nx
        and math.floor(b) >= 1 and math.floor(b) + 2 < ny for a, b in pairs)
