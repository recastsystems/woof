"""Producer-specific native control inventory, separate from downstream file allocation.

MPAS's package-qualified Registry scalar block includes PBL constituents
(WSM6+MYNN has ni). WOOF-ArWen's selected species row owns its transported block;
zeros added for a downstream file schema do not enlarge either inventory.
This describes source/layout authority; it does not implement an increment.
"""
from .producer_cloud import cloud_authority, MYNN_CLOUD_FIELDS


def native_scalar_names(layout):
    rows = [r for r in layout.fields if r['pool'] == 'state' and r['name'] == 'scalars']
    if len(rows) != 1 or not rows[0]['components']:
        raise ValueError('native increment layout requires the active Registry scalars block')
    return tuple(rows[0]['components'])


def increment_layout(*, route, microphysics, pbl, native_scalars=None):
    """Return source names and staggering; never select from a bg header alone.

    Native MPAS callers must pass active Registry components. WOOF-ArWen callers
    use its actual scheme row, including declared number/volume constituents.
    Common thermodynamic variable transforms and balanced edge-wind updates
    remain the consuming adapter's responsibility.
    """
    authority = cloud_authority(pbl)
    if route == 'mpas':
        if not native_scalars:
            raise ValueError('native MPAS requires active Registry scalar components')
        scalars = tuple(native_scalars)
        scalar_authority = 'active MPAS Registry state.scalars components'
    elif route == 'arwen':
        from .species_row import species_row_for_scheme
        scalars = species_row_for_scheme(microphysics).names()
        scalar_authority = 'selected WOOF-ArWen microphysics species row'
    else:
        raise ValueError(f'unknown native producer route {route!r}')
    if len(set(scalars)) != len(scalars) or 'qv' not in scalars:
        raise ValueError('invalid native scalar inventory')
    return {
        'route': route, 'microphysics': microphysics, 'pbl': pbl,
        'scalar_authority': scalar_authority, 'scalars': list(scalars),
        'cell_mass_level': ['theta', 'rho', 'pressure', 'uReconstructZonal',
                            'uReconstructMeridional', *scalars],
        'cell_interface_level': ['w'],
        'edge_mass_level': ['u'],
        'sgs_authority': authority,
        'sgs_native_fields': [name for name, _, _ in MYNN_CLOUD_FIELDS]
                             if authority == 'native_mynn' else [],
        'sgs_role': 'parameterized cloud state; not extra transported species',
        'wind_note': 'Vr consumes cell earth-relative u/v plus interface w; '
                     'native edge u recoupling must be consistent with analysed cell winds',
    }
