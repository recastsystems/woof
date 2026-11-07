"""Engine-side upper-ocean diagnostics. Contains no private DA implementation."""

def validate_snapshot_diagnostics(arrays):
    """Validate missing diagnostic values without weakening state checks.

    Land/inactive columns are missing. D26 is missing for a warm truncated
    column; ILD/MLD are missing when their thresholds are not crossed.
    The prognostic T/S arrays remain subject to the ordinary finite gate.
    """
    import numpy as np
    names = {"ocean_ohc26", "ocean_d26", "ocean_d26_censored",
             "ocean_ild", "ocean_mld", "ocean_barrier_layer"}
    present = names.intersection(arrays)
    if not present:
        return set()
    if present != names or "ocean_pwp_active" not in arrays:
        raise FloatingPointError("incomplete ocean diagnostic snapshot contract")
    wet = np.asarray(arrays["ocean_pwp_active"]) > .5
    values = {name: np.asarray(arrays[name]) for name in names}
    for name, value in values.items():
        if value.shape != wet.shape or np.any(np.isinf(value)):
            raise FloatingPointError(f"invalid ocean diagnostic {name}")
        if np.any(np.isfinite(value[~wet])):
            raise FloatingPointError(f"inactive ocean diagnostic {name} must be missing")
        if np.any(value[wet] < 0):
            raise FloatingPointError(f"negative ocean diagnostic {name}")
    for name in ("ocean_ohc26", "ocean_d26_censored"):
        if not np.all(np.isfinite(values[name][wet])):
            raise FloatingPointError(f"active ocean diagnostic {name} is non-finite")
    censored = values["ocean_d26_censored"][wet]
    if not np.all((censored == 0) | (censored == 1)):
        raise FloatingPointError("invalid ocean D26 censor flag")
    if not np.array_equal(np.isfinite(values["ocean_d26"][wet]), censored == 0):
        raise FloatingPointError("ocean D26 missingness disagrees with censor flag")
    ild, mld, barrier = (values[name][wet] for name in
                        ("ocean_ild", "ocean_mld", "ocean_barrier_layer"))
    defined = np.isfinite(ild) & np.isfinite(mld)
    if not np.array_equal(np.isfinite(barrier), defined):
        raise FloatingPointError("ocean barrier missingness disagrees with layer depths")
    if not np.allclose(barrier[defined], np.maximum(0, ild[defined]-mld[defined]),
                       rtol=2.e-5, atol=2.e-5):
        raise FloatingPointError("ocean barrier disagrees with layer depths")
    return names

def history_fields(ocean):
    """Output-cadence CUDA diagnostics from native hex PWP columns.

    Returned fields follow owned-cell order, just like the existing ocean
    history fields. The caller's existing history remapping restores file order.
    """
    import cupy as cp
    from woof.ocean.tc_diagnostics import diagnose
    columns=ocean.array('pwp_columns')
    p=ocean.options['pwp']
    bottom=(cp.arange(columns.shape[-1],dtype=cp.float64)+1)*p['layer_m']
    wet=ocean.array('pwp_diagnostics')[:,3]>0
    return {name.lower():value for name,value in diagnose(columns[:,0],columns[:,1],bottom,wet,
        density=p['density'],heat_capacity=p['heat_capacity'],
        thermal_expansion=p['thermal_expansion'],haline_contraction=p['haline_contraction']).items()}
