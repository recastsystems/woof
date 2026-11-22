"""Observations for WOOF-ArWen: acquisition, geometry, superobbing, and the
``gpuwm-obs.radar-grid`` contract the DA lanes consume.

The pipeline is four stages with a provable artifact between each pair:

``rw_nexrad fetch``   S3 -> Level-II volume on disk (+ sha256)
``rw_nexrad decode``  volume -> ``gpuwm-obs.radar-sweeps.v1`` pack
:mod:`woof.obs.superob`   pack + :class:`~woof.obs.target_grid.TargetGrid`
                           -> gridded accumulators
:mod:`woof.obs.radar_grid` -> ``gpuwm-obs.radar-grid`` NetCDF (v2 when
                           the set is windowed, v1 when it is not)

Nothing in here names a case, a campaign, or a particular radar: site ids,
bucket names and time windows are data that arrive through arguments.
"""

from __future__ import annotations

from woof.obs.radar_grid import (RADAR_GRID_SCHEMA, RADAR_GRID_STATUS,
                                  RadarGridSchemaError, read_radar_grid,
                                  write_radar_grid)
from woof.obs.superob import (GriddedObservations, SuperobParams,
                               SuperobParamsError, merge_contributions,
                               superob_volume)
from woof.obs.sweeps import (SWEEPS_SCHEMA, RadarSweepPackError,
                              RadarVolume, read_sweep_pack)
from woof.obs.target_grid import (TARGET_GRID_SCHEMA, GridMismatchError,
                                   TargetGrid)

__all__ = [
    "GriddedObservations",
    "GridMismatchError",
    "RADAR_GRID_SCHEMA",
    "RADAR_GRID_STATUS",
    "RadarGridSchemaError",
    "RadarSweepPackError",
    "RadarVolume",
    "SWEEPS_SCHEMA",
    "SuperobParams",
    "SuperobParamsError",
    "TARGET_GRID_SCHEMA",
    "TargetGrid",
    "merge_contributions",
    "read_radar_grid",
    "read_sweep_pack",
    "superob_volume",
    "write_radar_grid",
]
