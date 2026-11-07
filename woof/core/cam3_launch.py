"""Stream-ordered CAM absorption schedule, shared by production and replay."""
from __future__ import annotations
import numpy as np
from woof.core.cam3_memory import BLOCK_THREADS, expression_count, layout


def launch_columns(nz, state, buffers, workers, *, refresh, shortwave=True, longwave=True):
    import cupy as cp
    from woof.numerics import is_strict
    from woof.core.kernels import get_kernel_int_defines
    metadata, _ = layout()
    columns = buffers[0].shape[0]
    defines = (('CAM_LEVELS', nz),('CAM_LONGWAVE_MODE',1 if longwave else 2),('CAM_SHORTWAVE_MODE',1 if shortwave else 2))
    workspace = cp.empty((workers, expression_count(metadata['workspace_expression'], nz)), cp.uint8)
    if not refresh:
        kernel = get_kernel_int_defines('cam3_emissivity', 'cam_driver', defines)
        kernel((workers//BLOCK_THREADS,), (BLOCK_THREADS,),
               (np.int32(columns), state, workspace, *buffers))
        return
    stage = get_kernel_int_defines('cam3_emissivity', 'cam_driver_stage', defines)
    descriptors = cp.empty((workers, metadata['absorption_operands']), cp.uint64)
    absorption = cp.empty((workers, (nz+1)**2+nz*4+nz+1), cp.float64 if is_strict() else cp.float32)
    absorb = get_kernel_int_defines('cam3_emissivity', 'cam_absorption', defines)
    lanes = expression_count(metadata['absorption_lanes_expression'], nz)
    absorb_work = cp.empty((workers*lanes, expression_count(
        metadata['absorption_workspace_expression'], nz)), cp.uint8)
    for first in range(0, columns, workers):
        last = min(first+workers, columns)
        count = last-first
        views = [value[first:last] for value in buffers]
        args = (np.int32(count), state, workspace)
        absorption.fill(0)
        stage((workers//BLOCK_THREADS,), (BLOCK_THREADS,),
              (*args,np.int32(1),descriptors,absorption,*views))
        absorb(((count*lanes+63)//64,), (64,),
               (np.int32(count),workspace,descriptors,absorption,absorb_work))
        stage((workers//BLOCK_THREADS,), (BLOCK_THREADS,),
              (*args,np.int32(2),descriptors,absorption,*views))
