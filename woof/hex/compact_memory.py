"""Whole-run memory rows for the bounded, single-card allocation policy.

Rows are specific to the route, suite, vertical extent and card measured.
The intercept of a whole-run fit is not a measurement of idle CUDA memory.
Other configurations retain the conservative legacy admission model.
"""
from dataclasses import dataclass
from functools import lru_cache
import json
import os
from pathlib import Path
from .device_admission import ShapedFootprintModel, MIB


@dataclass(frozen=True)
class CompactFootprintModel(ShapedFootprintModel):
    route: str = 'mpas'
    suite: str = 'mesoscale_reference'
    peak_knots: tuple[tuple[int, int], ...] = ()

    def predict_bytes(self, cells: int) -> float:
        if cells <= 0:
            raise ValueError('cells must be positive')
        points = self.peak_knots
        if len(points) == 1:
            peak = points[0][1] * cells / points[0][0]
        else:
            index = next((i for i in range(1, len(points)) if cells <= points[i][0]), len(points)-1)
            (x0, y0), (x1, y1) = points[index-1:index+1]
            peak = y0 + (y1-y0)*(cells-x0)/(x1-x0)
        return max(self.core_bytes, peak) + self.rollback_bytes_per_cell*cells

    def max_cells(self, free_bytes: int, margin_bytes: int | None = None) -> int:
        # A nearly flat last segment is not evidence for larger meshes.
        return min(self.peak_knots[-1][0], super().max_cells(free_bytes, margin_bytes))

    def tiled_bytes(self, cells: int) -> int:
        return 0  # The measured whole-run peaks already include every tile.

    def tiled_breakdown(self, cells: int) -> dict[str, int]:
        return {'included_in_whole_run_peak': 0}

    def margin_terms(self):
        return {'sampling_and_placement': {
            'bytes': 256*MIB,
            'absorbs': '100 ms sampling, allocator placement and whole-device versus process convention',
            'scales_with': 'nothing',
            'measured': 'reserve in addition to the sampled whole-device peak; not another peak measurement'}}

    def as_dict(self):
        row = super().as_dict()
        row.update(shape='piecewise upper envelope of whole-run peaks by route and suite', route=self.route,
            suite=self.suite, peak_knots=[list(p) for p in self.peak_knots],
            tiled_workspace_knee_cells={},
            intercept_note='fit intercept, not measured fixed overhead',
            scope=f'one card, {self.levels} levels, compact defaults; interpolation within measured column range only')
        return row


@lru_cache(maxsize=1)
def measured_rows():
    return json.loads((Path(__file__).parent/'data'/'compact-memory.v1.json').read_text())['profiles']


def compact_model(card, configuration, levels, *, route, suite=None, rollback_armed=False,
                  cells=None):
    """Return a qualified row, never borrow it for another card or suite."""
    from .device_memory_release import shared_card_memory
    from .device_admission import ROLLBACK_ARMED_BYTES_PER_CELL
    if not shared_card_memory() or os.environ.get('WOOF_HEX_MEMBER_MEMORY', 'compact').strip().lower() != 'compact':
        return None
    from . import numerics
    arithmetic = numerics.mode()
    # Explicit workspace or allocator experiments have a different footprint.
    overrides = ('WOOF_HEX_THOMPSON_WORKSPACE', 'WOOF_HEX_THOMPSON_TILE_COLUMNS',
        'WOOF_HEX_THOMPSON_WORKSPACE_MIB', 'WOOF_HEX_TRANSPORT_WORKSPACE_BYTES',
        'WOOF_HEX_LIMITER_SPECIES_PER_PASS', 'GPUWM_HEX_POOL_RELEASE_MIN_BYTES',
        'GPUWM_HEX_RELEASE_HEADROOM', 'WOOF_HEX_POOL_RELEASE', 'GPUWM_HEX_LMEM_RELEASE')
    if any(os.environ.get(key) for key in overrides):
        return None
    if suite is None:
        from .physics_default import default_suite
        suite = 'arwen_mpas_column' if route == 'arwen' else default_suite()
    matches = [row for row in measured_rows()
        if row['route'] == route and row['suite'] == suite
        and row.get('numerics', 'fast') == arithmetic
        and row['configuration'] == configuration and row['levels'] == int(levels)
        and row['card_sm'] == card.multiprocessors
        and row['card_threads_per_sm'] == card.max_threads_per_sm
        and row['card_name_contains'] in card.name
        and (row.get('only_cells') is None or cells == row['only_cells'])
        and cells is not None
        and row['samples'][0]['cells'] <= cells <= row['samples'][-1]['cells']]
    if not matches:
        return None
    row, = matches
    peaks = tuple((point['cells'], point['peak_bytes']) for point in row['samples'])
    slope = ((peaks[-1][1]-peaks[0][1])/(peaks[-1][0]-peaks[0][0])
             if len(peaks) > 1 else peaks[0][1]/peaks[0][0])
    intercept = peaks[0][1]-slope*peaks[0][0]
    return CompactFootprintModel(core_bytes=intercept, bytes_per_cell=slope,
        card=card, configuration=configuration, levels=int(levels), measured=False,
        rollback_bytes_per_cell=ROLLBACK_ARMED_BYTES_PER_CELL if rollback_armed else 0,
        route=route, suite=suite, peak_knots=peaks,
        provenance=row['provenance'] + '; data/compact-memory.v1.json; '
            'interpolation within measured shapes; conservative fallback outside')
