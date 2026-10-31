"""RRFS Thompson: identical MMM scheme with full-column fork writeback.

MPAS-Model-rrfs 1d6e370e interface.F:1223,1291,1528 removes the
v8.4.2 mp_top_level cutoff. driver_microphysics.F:1234 seeds reflectivity
from the prior field, but MMM mp_thompson.F90 calc_refl10cm overwrites
all levels with -35 before using it, so the shared radar is identical.
"""
from .. import thompson_v842 as shared


def bind_native_pools(provider, call):
    # The fork has no mp_top_level Registry field. Select its full-column
    # writeback before the shared binder can request the v8.4.2 cutoff.
    return shared.bind_native_pools(
        provider, call, writeback_levels=provider.session.layout.dimensions['nVertLevels'])


run_slot = shared.run_slot
