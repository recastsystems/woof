"""GPU output accumulators from pinned native and operational WRF diagnostics.

Bucket reset occurs at step start, independently of history writes.
Stock prec_acc_dt: WRF 4.7.1, phys/module_diag_misc.F:415-448.
Fork prec_acc_dt1: NOAA-EMC/HRRR 40ee6058, module_diag_gsd.F:724-813.
"""
from __future__ import annotations
import math
import numpy as np

PRECIP_HISTORY = {
    prefix + suffix: (prefix.upper() + suffix, "mm", description)
    for suffix in ("", "1")
    for prefix, description in (("prec_acc_c", "Interval convective precipitation"),
        ("prec_acc_nc", "Interval grid-scale precipitation"),
        ("snow_acc_nc", "Interval snow including frozen convective rain"),
        ("graup_acc_nc", "Interval grid-scale graupel"))}
from woof.core.gsd_diagnostics import HISTORY as GSD_HISTORY, PERSISTENT as GSD_PERSISTENT
from woof.core.climate_diagnostics import HISTORY as CLIMATE_HISTORY, PERSISTENT as CLIMATE_PERSISTENT
from woof.core.clwrf import HISTORY as CLWRF_HISTORY, PERSISTENT as CLWRF_PERSISTENT
from woof.core.nwp_diagnostics import HISTORY as NWP_HISTORY, PERSISTENT as NWP_PERSISTENT
from woof.core.rasm import HISTORY as RASM_HISTORY, PERSISTENT as RASM_PERSISTENT
from woof.core.solar_diagnostics import HISTORY as SOLAR_HISTORY, PERSISTENT as SOLAR_PERSISTENT
from woof.core.afwa_diagnostics import HISTORY as AFWA_HISTORY, PERSISTENT as AFWA_PERSISTENT
from woof.core.misc_buckets import (HISTORY as BUCKET_HISTORY, PERSISTENT as BUCKET_PERSISTENT,
    scratch_shapes as bucket_shapes, update as update_buckets)
from woof.core.physics_tendency_output import (HISTORY as TENDENCY_HISTORY,
    PERSISTENT as TENDENCY_PERSISTENT, scratch_shapes as tendency_shapes,
    update as update_tendencies)
from woof.core.misc_diag_print import (PERSISTENT as PRINT_PERSISTENT,
    scratch_shapes as print_shapes, update as update_print)
from woof.core.uh_layers import HISTORY as UH_LAYERS_HISTORY, PERSISTENT as UH_LAYERS_PERSISTENT
HISTORY = SOLAR_HISTORY | RASM_HISTORY | CLWRF_HISTORY | TENDENCY_HISTORY | BUCKET_HISTORY | AFWA_HISTORY | NWP_HISTORY | PRECIP_HISTORY | GSD_HISTORY | CLIMATE_HISTORY
# The fork's own UP_HELI_MAX02/03 rows (and their text) win where both name a slot.
HISTORY = HISTORY | {slot: row for slot, row in UH_LAYERS_HISTORY.items() if slot not in HISTORY}
PERSISTENT = tuple(dict.fromkeys((*PRINT_PERSISTENT, *TENDENCY_PERSISTENT, *BUCKET_PERSISTENT, *PRECIP_HISTORY, 'prec_acc_epoch', *GSD_PERSISTENT, *CLIMATE_PERSISTENT, *NWP_PERSISTENT, *AFWA_PERSISTENT, *CLWRF_PERSISTENT, *RASM_PERSISTENT, *SOLAR_PERSISTENT, *UH_LAYERS_PERSISTENT)))

def scratch_shapes(cfg):
    shapes = {slot: (cfg.ny, cfg.nx) for slot in PRECIP_HISTORY
              if getattr(cfg, "prec_acc_dt1" if slot.endswith("1") else "prec_acc_dt") > 0}
    if shapes:
        shapes["output_zero"] = (cfg.ny, cfg.nx)
    if cfg.prec_acc_dt > 0:
        shapes['prec_acc_epoch'] = (2,)
    from woof.core.gsd_diagnostics import scratch_shapes as gsd_shapes
    shapes.update(gsd_shapes(cfg))
    from woof.core.climate_diagnostics import scratch_shapes as climate_shapes
    shapes.update(climate_shapes(cfg))
    from woof.core.uh_layers import scratch_shapes as uh_layers_shapes
    shapes.update(uh_layers_shapes(cfg))
    from woof.core.clwrf import scratch_shapes as stock_shapes
    shapes.update(stock_shapes(cfg))
    from woof.core.nwp_diagnostics import scratch_shapes as nwp_shapes
    shapes.update(nwp_shapes(cfg))
    from woof.core.afwa_diagnostics import scratch_shapes as afwa_shapes
    shapes.update(afwa_shapes(cfg))
    shapes.update(bucket_shapes(cfg))
    shapes.update(tendency_shapes(cfg))

    from woof.core.rasm import scratch_shapes as rasm_shapes
    shapes.update(rasm_shapes(cfg))
    from woof.core.solar_diagnostics import scratch_shapes as solar_shapes
    shapes.update(solar_shapes(cfg))
    shapes.update(print_shapes(cfg))
    return shapes

def initialize(state, cfg):
    for slot, shape in scratch_shapes(cfg).items():
        if not slot.startswith(("clwrf_", "rasm_")):
            state.scratch(shape, slot)
    from woof.core.clwrf import initialize as initialize_stock
    initialize_stock(state, cfg)
    from woof.core.rasm import initialize as initialize_rasm
    initialize_rasm(state,cfg)
    if cfg.gsd_diagnostics:
        for slot in ("wspd10", "wspd80"):
            state.existing_scratch(slot)[...] = -99.

def boundary_due(seconds, interval_seconds):
    # The fork uses MOD(curr_secs2, interval)==0, at the beginning of a step.
    return seconds == 0 or math.isclose(seconds % interval_seconds, 0., abs_tol=1.e-6)

def update_precipitation(state, cfg):
    update_tendencies(state, cfg)
    update_buckets(state, cfg)
    update_print(state, cfg)
    if cfg.prec_acc_dt <= 0 and cfg.prec_acc_dt1 <= 0:
        return
    from woof.core.kernels import get_kernel
    physics = state.physics
    micro = getattr(physics, "microphysics", None)
    zero = state.scratch((cfg.ny, cfg.nx), "output_zero")
    rainc = getattr(physics, "cu_raincv", None)
    def amount(name):
        value = getattr(micro, name, None)
        return zero if value is None or cfg.mp_physics == 0 else value
    temperature = getattr(physics, "fields", {}).get("t2", zero)
    count = cfg.nx * cfg.ny
    for suffix, minutes in (("", cfg.prec_acc_dt), ("1", cfg.prec_acc_dt1)):
        if minutes <= 0:
            continue
        inputs = (zero if rainc is None else rainc, amount("rainncv"),
             amount("snowncv"), amount("graupelncv"), temperature,
             *(state.existing_scratch(name+suffix) for name in
               ("prec_acc_c", "prec_acc_nc", "snow_acc_nc", "graup_acc_nc")))
        if suffix:
            get_kernel("output_diagnostics", "precipitation_buckets")(((count+255)//256,), (256,),
                (*inputs, np.int32(count), np.int32(boundary_due(state.elapsed_seconds, minutes*60.))))
        else:
            epoch = float(getattr(state, 'domain_start_offset', 0.) or 0.)
            get_kernel("output_diagnostics", "native_precipitation_buckets")(((count+255)//256,), (256,),
                (*inputs, state.existing_scratch('prec_acc_epoch'), np.int32(count),
                 np.float64(state.elapsed_seconds-epoch), np.float32(minutes)))


def registry_rows(registry):
    from woof.core.misc_diag_print import registry_rows as print_registry_rows
    print_registry_rows(registry)
    from woof.core.physics_tendency_output import registry_rows as tendency_registry_rows
    tendency_registry_rows(registry)
    from woof.core.misc_buckets import registry_rows as bucket_registry_rows
    bucket_registry_rows(registry)
    registry['parameters'].setdefault('prec_acc_dt', {}).update(
        type='real', default=0.0, minimum=0.0, per_domain=True,
        source_pin='f52c197ed39d12e087d02c50f412d90d418f6186',
        consuming_read='woof/core/output_diagnostics.py')
    registry.setdefault('diagnostic_state', {})['precipitation'] = {
        name: {'description': d, 'units': u, 'history': True, 'restart': True}
        for name, (_, u, d) in PRECIP_HISTORY.items() if not name.endswith('1')}
    registry['diagnostic_state']['precipitation']['prec_acc_epoch'] = {
        'description': 'Native restart-relative accumulation clock origin (split real64)',
        'units': 's', 'history': False, 'restart': True}
