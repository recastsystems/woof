"""Native input names for the registered dynamical-core PBL scalars."""
from __future__ import annotations
from woof.core.pbl_scalar_inventory import PBL_TRANSPORTED_SCALARS, pbl_transported_scalars

ALL_PBL_WRFINPUT = frozenset(name for fields in PBL_TRANSPORTED_SCALARS.values() for name in fields)


def pbl_input_map(cfg):
    """WRF Registry names and state names are identical for these scalars."""
    if not getattr(cfg, "bl_pbl_physics", 0):
        return {}
    return {name: name for name in pbl_transported_scalars(cfg)}


def restore_pbl_scalars(state, raw, cfg, array_module):
    """Preserve supplied float32 words and synchronize the RK time-t copies."""
    mapping = pbl_input_map(cfg)
    supplied = mapping.keys() & raw.keys()
    if supplied and supplied != mapping.keys():
        raise ValueError(f"WRF input lacks part of the supplied PBL scalar inventory: {sorted(mapping.keys() - supplied)}")
    for wrf_name, state_name in mapping.items():
        target = getattr(state, state_name, None)
        if target is None:
            raise ValueError(f"DomainState lacks active PBL scalar {state_name}")
        if supplied:
            value = raw[wrf_name]
            if value.shape != target.shape:
                raise ValueError(f"WRF PBL scalar {wrf_name} shape {value.shape} != {target.shape}")
            target[...] = array_module.asarray(value, dtype=array_module.float32)
        else:
            # Registry i0 carriers start at zero when selecting a new PBL
            # package on a cold real.exe input from another package.
            target[...] = array_module.float32(0.0)
        initial = getattr(state, f"{state_name}0", None)
        if initial is not None:
            initial[...] = target
