"""Explicit optional sea-ice carrier coverage for global moving partitions.

The column engine allocates these stores only when a class contains ocean ice.
An ice-free class therefore has no ice volume or accumulated ice budgets to
carry. Its inactive skin is the native initialization value, min(tsk, 273.15).
Unknown partial sets and missing stores over ice are refused, never cold seeded.
"""
import numpy as np

from woof.ocean.seaice import COLD_SEEDED

# The engine's own list of the carriers it creates together, read rather than copied. THE BREAKAGE THIS FIXES: a copy here
# went stale when woof's sea ice gained ice_ocean_owed and ice_ocean_coupled, so a member holding polar ice stopped at its
# first move, "unknown lazy physics carrier: fields/ice_ocean_coupled" (run_84cabe12ce57b858), and with the flag dropped at
# the move its next segment stopped with KeyError 'ice_ocean_coupled' (run_3218e3dc972bb517).
SEAICE_FIELDS = frozenset('fields/' + name for name in COLD_SEEDED)


def complete_source_ice(payload):
    """Return a shallow carrier view with explicit inactive class stores."""
    sources = payload['backends']
    if not any(SEAICE_FIELDS & s['seam']['arrays'].keys() for s in sources):
        return payload
    result = []
    for source in sources:
        arrays = source['seam']['arrays']
        present = SEAICE_FIELDS & arrays.keys()
        if present and present != SEAICE_FIELDS:
            raise ValueError('incomplete prognostic sea-ice carrier fields')
        if not present:
            ice, lake, skin = (np.asarray(arrays['fields/' + k])
                               for k in ('xice', 'lakemask', 'tsk'))
            if any(a.shape != skin.shape or a.dtype != np.float32 or
                   not np.isfinite(a).all() for a in (ice, lake, skin)):
                raise ValueError('invalid inactive sea-ice column layout')
            if np.any((ice > 0) & (lake < .5)):
                raise ValueError('ocean ice has no carried prognostic stores')
            arrays = dict(arrays)
            for key in SEAICE_FIELDS:
                arrays[key] = np.zeros_like(skin)
            arrays['fields/ice_skin'] = np.minimum(skin, np.float32(273.15))
        result.append(dict(source, seam=dict(source['seam'], arrays=arrays)))
    return dict(payload, backends=result)


def prepare_restart_ice(seam, arrays):
    """Allocate authorized destinations before the engine's strict restoration.

    No cold initializer is called and no state is omitted. The engine still
    checks every manifest key, shape, identity and clock in restore_state.
    """
    present = SEAICE_FIELDS & arrays.keys()
    if not present:
        return
    if present != SEAICE_FIELDS:
        raise ValueError('incomplete prognostic sea-ice restart fields')
    fields = seam._driver.fields
    cp = seam._cp
    shape = fields['tsk'].shape
    for key in SEAICE_FIELDS:
        value = np.asarray(arrays[key])
        if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f'invalid prognostic sea-ice restart field: {key}')
    for key in SEAICE_FIELDS:
        name = key.split('/', 1)[1]
        if name not in fields:
            fields[name] = cp.zeros_like(fields['tsk'])


def remap_ocean_ice(payload, weights, target_mesh, target_landmask):
    """Carry native ice inventories only to wet cells of the new geography.

    Renormalize each donor's wet overlaps, preserving its extensive inventory.
    A coastal donor with no wet overlap goes to its nearest wet cell, recorded
    explicitly. This is a local coastline representation change, not ice drift.
    Ice area is capped at one by thickening, without discarding volume or snow.
    """
    from scipy.sparse import coo_matrix
    from scipy.spatial import cKDTree
    from .moving_physics import gather_owner_columns
    def gather(key):
        return gather_owner_columns([b['seam']['arrays'][key] for b in payload['backends']],
                                    payload['layouts'], len(weights.source_area))[0]
    old = {key: gather(key) for key in SEAICE_FIELDS | {'fields/xice'}}
    source_wet = (gather('fields/landmask').reshape(-1) < .5) & (
        gather('fields/lakemask').reshape(-1) < .5)
    target_wet = np.asarray(target_landmask).reshape(-1) < .5
    if len(target_wet) != len(weights.target_area) or not target_wet.any():
        raise ValueError('sea-ice remap requires complete target ocean geography')
    for key in ('fields/ice_volume', 'fields/ice_snow_mass'):
        if np.any(old[key] < 0) or np.any(old[key][..., ~source_wet] != 0):
            raise ValueError('sea-ice inventory is negative or lies outside source ocean')
    matrix = weights.overlap.copy()
    rows = np.repeat(np.arange(matrix.shape[0]), np.diff(matrix.indptr))
    matrix.data *= target_wet[rows] & source_wet[matrix.indices]
    matrix.eliminate_zeros()
    totals = np.asarray(matrix.sum(axis=0)).ravel()
    missing = np.flatnonzero(source_wet & (totals == 0))
    distance_max = 0.
    distance = np.empty(0)
    if len(missing):
        def xyz(mesh, ids):
            p = np.stack([mesh.arrays[k][ids] for k in ('xCell','yCell','zCell')], axis=-1)
            return p / np.linalg.norm(p, axis=-1, keepdims=True)
        wet_ids = np.flatnonzero(target_wet)
        distance, near = cKDTree(xyz(target_mesh, wet_ids)).query(
            xyz(payload['mesh'], missing), workers=-1)
        distance_max = float(2 * np.arcsin(np.minimum(distance / 2, 1)).max() *
                             payload['mesh'].attrs['sphere_radius'] / 1000)
        matrix += coo_matrix((weights.source_area[missing], (wet_ids[near], missing)),
                             shape=matrix.shape).tocsr()
        totals = np.asarray(matrix.sum(axis=0)).ravel()
    scale = np.divide(weights.source_area, totals, out=np.zeros_like(totals), where=totals > 0)
    matrix.data *= scale[matrix.indices]
    def carry(value):
        flat = np.asarray(value, dtype=np.float64).reshape(-1, len(weights.source_area))
        return ((matrix @ flat.T).T / weights.target_area).reshape(
            *value.shape[:-1], len(weights.target_area))
    mapped = {key: carry(value).astype(value.dtype) for key, value in old.items()
              if key not in ('fields/ice_skin', 'fields/ice_ocean_mask')}
    volume = mapped['fields/ice_volume']
    concentration_raw = mapped['fields/xice'].copy()
    mapped['fields/xice'] = a = np.where(volume > 0, np.minimum(concentration_raw, 1), 0).astype(np.float32)
    if np.any((volume > 0) & (a <= 0)):
        raise ValueError('carried ice volume has no carried ice area')
    skin_weight = carry(old['fields/ice_volume'])
    mapped['fields/ice_skin'] = np.divide(carry(old['fields/ice_skin'] * old['fields/ice_volume']),
        skin_weight, out=np.full_like(skin_weight, 273.15), where=skin_weight > 0).astype(np.float32)
    mapped['fields/ice_ocean_mask'] = np.broadcast_to(target_wet, a.shape).astype(np.float32).copy()
    # The ice step's 0/1 flag for the columns whose surface fluxes it set, not an inventory: an overlap average would leave
    # fractions, and it stays 0 or 1. The next ice step rewrites it before the ocean reads it.
    mapped['fields/ice_ocean_coupled'] = (mapped['fields/ice_ocean_coupled'] > 0).astype(np.float32)
    mapped['fields/ice_thickness'] = np.divide(volume, a, out=np.zeros_like(a), where=a > 0)
    mapped['fields/ice_snow_depth'] = np.divide(mapped['fields/ice_snow_mass'], 330 * a,
                                              out=np.zeros_like(a), where=a > 0)
    inventory = {}
    for key in ('fields/ice_volume', 'fields/ice_snow_mass'):
        before = float(np.sum(old[key].astype(float) * weights.source_area))
        after = float(np.sum(mapped[key].astype(float) * weights.target_area))
        inventory[key] = dict(before=before, after=after, residual=after-before,
                              relative_residual=(after-before)/before if before else 0.)
    fallback_ice = old['fields/ice_volume'].reshape(-1)[missing] > 0
    fallback_inventory = {}
    for key in ('fields/ice_volume', 'fields/ice_snow_mass'):
        total = float(np.sum(old[key].astype(float) * weights.source_area))
        amount = float(np.sum(old[key].reshape(-1)[missing].astype(float) * weights.source_area[missing]))
        fallback_inventory[key] = amount / total if total else 0.
    receipt = dict(method='donor-conservative overlap restricted to target ocean; nearest wet fallback',
        nearest_wet_donors=len(missing), nearest_wet_distance_max_km=distance_max,
        nearest_wet_donors_with_ice=int(fallback_ice.sum()),
        nearest_wet_ice_distance_max_km=(float(2*np.arcsin(np.minimum(distance[fallback_ice]/2,1)).max()*
            payload['mesh'].attrs['sphere_radius']/1000) if fallback_ice.any() else 0.),
        nearest_wet_inventory_fractions=fallback_inventory,
        concentration_capped_cells=int(np.count_nonzero(concentration_raw > 1)),
        concentration_raw_max=float(concentration_raw.max()), inventories=inventory,
        target_land_ice_cells=int(np.count_nonzero(volume[..., ~target_wet])),
        area_units='same as atmospheric physical overlap', snow_density_kg_m3=330.)
    return mapped, receipt
