"""The RRFS route's diagnostic stream fields on the device (the mpasphys_rrfs row's ``output_diagnostics``).

MPAS-Model-rrfs@1d6e370e src/core_atmosphere/mpas_atm_core.F:680-700,845-860 runs the same
update/compute/write/reset order as MPAS-A v8.4.2, with the fork's convective diagnostics
(src/core_atmosphere/diagnostics/mpas_convective_diagnostics.F:297-681: the hourly helicity bands,
w and 10 m wind means, wind02 maxima and the convective/grid-scale precipitation accumulations) and
``mpas_atm_diag_update(configs, stream_mgr, dt)``. The isobaric module is the shared one.
"""
from ...mpasdiag import convective_rrfs
from ...mpasdiag.isobaric import FIELDS as ISOBARIC
from ..resident_output_diagnostics_v842 import (ACOUSTIC, INITIALIZED, MOMENTUM, PRIMARY,
                                                ResidentOutputDiagnostics, _Pools)


class _ForkPools(_Pools):
    """The fork convective module's views: vertex fields span nVertices, as cellsOnVertex does here.

    The isobaric module keeps the shared views (it reads vorticity with its dummy vertex).
    """

    def __getitem__(self, name):
        value = super().__getitem__(name)
        if self.pool == 'diag' and name == 'vorticity':
            return value[:, :self.owner.session.layout.dimensions['nVertices']]
        return value

    def view(self, name):
        # THE BREAKAGE THIS PREVENTS: a same-host rank's resident allocations hold owned columns only
        # (device_pools_v842.field packs nCellsSolve), but the fork's helicity split reads w at every local cell,
        # halo ring included (mpas_convective_diagnostics.F:585-605). The in-place view would hand it the owned
        # allocation, so halo cells read other levels' words and the top level reads past the allocation. On a rank
        # w is the full-extent copy whose halo columns _update_convective fills with the dycore's owner truth.
        if name == 'w' and self.pool == 'state' and self.owner.halo is not None:
            return self[name]
        return super().view(name)


class _StreamIntervals:
    """mpas_stream_mgr_get_stream_interval on the writer's streams (seconds; 0 for a non-recurring one)."""

    def __init__(self, writer):
        self.writer = writer

    def get_stream_interval(self, name, direction):
        from ...mpasdiag.utils import Direction
        from ...streams_rrfs import interval_seconds
        stream = self.writer.streams[name]
        return interval_seconds(stream.interval('input' if direction is Direction.INPUT else 'output')) or 0


def _block_creator_order(voc, neo, cov, owned, ncell, nv, halo):
    """Vertices in a native MPAS rank's local order (mpas_block_creator.F:269-456 with mpas_block_decomp.F:302-422).

    ``voc`` (cells, maxEdges) and ``cov`` (vertices, degree) are zero-based; an index outside 0..ncell-1 (or
    0..nv-1 for a vertex) names no local element. ``owned`` cells come first in ascending global id; ``halo`` is the
    halo cells in native traversal order (layer, then global id). The 0-halo vertices in first-appearance order
    over the owned cells and their verticesOnCell slots, those whose first valid cellsOnVertex entry is owned kept
    at the front and the others written from the end, so reversed; then the halo's new vertices in first-appearance
    order; vertices no local cell names come last. Vectorised: no per-cell host loop at startup.
    """
    import numpy as np
    seen = np.zeros(nv, dtype=bool)

    def first_appearance(cells):
        slots = voc[cells]                                 # cell-major, then slot: native's loop order
        valid = ((np.arange(voc.shape[1])[None, :] < neo[cells][:, None]) & (slots >= 0) & (slots < nv))
        sequence = slots[valid]
        sequence = sequence[~seen[sequence]]
        _, at = np.unique(sequence, return_index=True)
        found = sequence[np.sort(at)]
        seen[found] = True
        return found
    first = first_appearance(np.arange(owned))
    tail = first_appearance(np.asarray(halo, dtype=np.int64))
    # Native mpas_block_decomp.F:593-600 (44c0849; also June fork) skips
    # zero neighbours before assigning ownership. Pools have already mapped
    # that file zero to nCells+1. c3-lam's first-slot shortcut reverses outer
    # vertices unnecessarily when their first neighbour is outside the cut.
    adjacent = cov[first].T
    valid = (adjacent >= 0) & (adjacent < ncell)
    head = adjacent[np.argmax(valid, axis=0), np.arange(first.size)]
    mine = (head >= 0) & (head < owned)
    return np.concatenate((first[mine], first[~mine][::-1], tail, np.flatnonzero(~seen)))


def native_rank_vertex_order(session):
    """This rank's local vertices in a native MPAS rank's local order, the order its vertex loops scatter in.

    The fork's helicity adds each cell's vertex terms in that order (mpas_convective_diagnostics.F:558-565), so
    seam cells match native N ranks only when the gather follows it (one-card order: 20 seam cells differed at
    f000 on 2 ranks, 2026-10-28). See _block_creator_order; each halo layer's cells are taken in ascending global
    id (mpas_block_creator.F:604,738-942).
    """
    import numpy as np
    n, d = session.native, session.layout.dimensions
    owned, ncell, nv = int(d['nCellsSolve']), int(d['nCells']), int(d['nVertices'])
    voc = np.asarray(n.get_field('mesh', 'verticesOnCell'))[:, :ncell].astype(np.int64).T - 1
    neo = np.asarray(n.get_field('mesh', 'nEdgesOnCell'))[:ncell].astype(np.int64)
    cov = np.asarray(n.get_field('mesh', 'cellsOnVertex')).astype(np.int64).T - 1
    ids = np.asarray(n.get_field('mesh', 'indexToCellID'))[:ncell].astype(np.int64)
    ends = np.asarray(tuple(getattr(session.layout, 'halo_ends', {}).get('nCells', ())), dtype=np.int64)
    halo = np.arange(owned, ncell, dtype=np.int64)
    layer = np.searchsorted(ends, halo, side='right') if ends.size else np.zeros_like(halo)
    order = _block_creator_order(voc, neo, cov, owned, ncell, nv, halo[np.lexsort((ids[halo], layer))])
    return session.device_pools.cp.asarray(order)


def one_card_vertex_order(mesh, *, limited_area):
    """The global mesh's vertices in the order one card's helicity gather adds them (ForkOutputDiagnostics.bind).

    A limited-area cut on one card takes native's single-rank block-creator order (every cell owned, no halo); a
    closed mesh on one card keeps file order (vertex_order None). ``mesh`` is the hexcore Mesh (canonical zero-based
    connectivity, -1 for no element). Cached on the mesh: the ranks of one process share it.
    """
    import numpy as np
    key = '_one_card_vertex_order_' + ('lam' if limited_area else 'closed')
    held = getattr(mesh, '__dict__', {}).get(key)
    if held is not None:
        return held
    nv = int(np.asarray(mesh.cellsOnVertex).shape[0])
    if limited_area:
        voc = np.asarray(mesh.verticesOnCell, dtype=np.int64)
        neo = np.asarray(mesh.nEdgesOnCell, dtype=np.int64)
        ncell = int(voc.shape[0])
        order = _block_creator_order(voc, neo, np.asarray(mesh.cellsOnVertex, dtype=np.int64), ncell, ncell, nv,
                                     np.zeros(0, dtype=np.int64))
    else:
        order = np.arange(nv, dtype=np.int64)
    try:
        setattr(mesh, key, order)
    except Exception:  # noqa: BLE001 - a read-only mesh only loses the cache
        pass
    return order


def one_card_rank_vertex_order(vertex_l2g, global_order):
    """This rank's local vertices sorted by their place in one card's gather order (``global_order``, global ids).

    Each cell then adds its vertex terms in the order one card adds them, so a rank's updraft helicity holds one
    card's words at every cell, seam cells included.
    """
    import numpy as np
    global_order = np.asarray(global_order, dtype=np.int64)
    place = np.empty(global_order.size, dtype=np.int64)
    place[global_order] = np.arange(global_order.size, dtype=np.int64)
    return np.argsort(place[np.asarray(vertex_l2g, dtype=np.int64)], kind='stable').astype(np.int64)


#: WOOF_HEX_UH_RANK_ORDER: the vertex order a same-host rank's helicity gather adds each cell's terms in.
#: ``one-card`` (default, unset): one card's order, so N ranks write one card's updraft-helicity words. ``native``:
#: a native MPAS rank's own local order, the proof mode against native RRFS-MPAS at the same rank count (whose seam
#: cells differ in the last place from its own one-rank run, as hex's then do).
UH_RANK_ORDER_WORD = 'WOOF_HEX_UH_RANK_ORDER'


def uh_rank_order():
    import os
    value = os.environ.get(UH_RANK_ORDER_WORD, '').strip().lower() or 'one-card'
    if value not in ('one-card', 'native'):
        raise ValueError(f'{UH_RANK_ORDER_WORD} must be one-card or native (got {value!r}): a misspelt proof word '
                         'would otherwise run the default order unseen')
    return value


class ForkOutputDiagnostics(ResidentOutputDiagnostics):
    """ResidentOutputDiagnostics with the fork's convective module and its update arguments."""
    produced = (PRIMARY | frozenset(ISOBARIC) | frozenset(convective_rrfs.FIELDS)
                | {'t_iso_levels', 'z_iso_levels'} | INITIALIZED | MOMENTUM.keys() | ACOUSTIC)
    convective_maxima = convective_rrfs.ACCUMULATORS
    pool_names = ('mesh', 'state', 'diag', 'diag_physics')

    def bind(self, writer):
        super().bind(writer)
        if self.halo is None and self.session.layout.configs.get('config_apply_lbcs', False):
            # A limited-area cut on one card: native's single rank still renumbers its vertices in block-creator order
            # (first appearance over the cells, vertices whose cellsOnVertex(1) is the garbage cell reversed at the
            # end), which a cut's parent-subset file order is not.  THE BREAKAGE THIS FIXES: lam7k f000 helicity
            # differed from native 1 rank in about 1,100-1,500 of 9,957 cells at 1-ulp level; on 2 ranks, where the
            # rank order was already used, 12-18.
            self.convective.driver.vertex_order = native_rank_vertex_order(self.session)
        if self.halo is not None:
            # A same-host rank gathers helicity in one card's vertex order, so N ranks write one card's words.  THE
            # BREAKAGE THIS FIXES: in a native rank's own order (the default until 20 Nov 2026) every seam cell added
            # its vertex terms in another order than one card, and the updraft-helicity maxima and minima differed
            # in the last place at the seams (about 1,300 of 12 million diag words a frame, 1 against 2 ranks,
            # 15 Mar 2025 north Alabama 3 km circle): a box's 8 ranks wrote other UH words than one
            # card. WOOF_HEX_UH_RANK_ORDER=native keeps a native rank's order (the native-N-ranks proof mode).
            if uh_rank_order() == 'native':
                self.convective.driver.vertex_order = native_rank_vertex_order(self.session)
            else:
                execution = self.halo.execution
                source = getattr(execution.inputs[0], 'partition_source_mesh', None)
                if source is None:
                    raise ValueError("this rank's mesh does not name the global mesh it was cut from; one card's "
                                     f"helicity order needs it ({UH_RANK_ORDER_WORD}=native runs without it)")
                order = one_card_rank_vertex_order(execution.partition.vertex_l2g, one_card_vertex_order(
                    source, limited_area=bool(self.session.layout.configs.get('config_apply_lbcs', False))))
                self.convective.driver.vertex_order = self.session.device_pools.cp.asarray(order)
            ends = tuple(getattr(self.session.layout, 'halo_ends', {}).get('nCells', ()))
            owned = int(self.session.layout.dimensions['nCellsSolve'])
            # Native's w halo: layers 1 and 2 (the last layer end also holds deeper rings).
            self._halo_w = (owned, int(ends[min(2, len(ends)) - 1]) if ends else
                            int(self.session.layout.dimensions['nCells']))

    def _convective_pools(self):
        return {name: _ForkPools(self, name) for name in self.pool_names}

    def _new_convective(self):
        return convective_rrfs.ConvectiveDiagnosticsModule()

    def _update_convective(self):
        if self.halo is not None:
            # Owner-truth halo w. The fork smooths seam-cell helicity with ring-1 halo helicity, which reads halo w
            # (mpas_convective_diagnostics.F:585-605); native exchanges w at halo layers 1,2 after every RK stage
            # (mpas_atm_halos.F:112-118, atm_srk3:1466-1471). A rank's physics pools are synchronised on owned
            # columns only, so their halo w kept stale words (up to 0.13 m/s after step 1, 2026-10-28). The
            # dycore's halo w is owner truth (its B rounds): copied on the device, no host sync.
            w = _ForkPools(self, 'state')['w']
            source = self.dycore.atmosphere.saved.vertical_velocity
            owned, end = self._halo_w
            end = min(end, int(w.shape[-1]), int(source.shape[-1]))
            levels = min(int(w.shape[0]), int(source.shape[0]))
            if end > owned:
                w[:levels, owned:end] = source[:levels, owned:end]
        # mpas_atm_core.F:696,858: mpas_atm_diag_update(configs, streamManager, dt).
        self.convective.update(self.session.layout.configs, _StreamIntervals(self.writer),
                               self.dycore.config.config_dt)
