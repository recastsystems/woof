"""Check the cooperative schedule against the original scalar equations."""
import numpy as np
import pytest
from tests.conftest import requires_gpu


@requires_gpu
def test_cooperative_dependencies_keep_the_scalar_words_at_extent_boundaries():
    import cupy as cp
    from woof import nvrtc_ptx_cache as compiler
    from woof.core import gbmpbl
    from woof.core.gbmpbl_fast import warp_source, COOPERATIVE_SLOTS
    from woof.verify.gbmpbl_oracle import read_cases, ORACLE_DIR

    # Compile both schedules without contraction to isolate dependency order
    # from the default compiler's legal FMA choices. Use the actual WRF fixture
    # including merging, cloud-top cooling, land/water and calm columns.
    source = gbmpbl.module_source()
    modules, kernels = [], []
    for text in (source, warp_source(source)):
        image, _ = compiler.compile_using_nvrtc(
            text, gbmpbl.MODULE_OPTIONS, None, 'gbmpbl-schedule-proof.cu')
        module = cp.cuda.function.Module()
        module.load(image.encode() if isinstance(image, str) else image)
        modules.append(module)
        kernels.append(module.get_function('gbmpbl_column'))
    case = read_cases(ORACLE_DIR / 'gbm-cases.bin')[0]
    ncol = case['ncol']  # Partial old-layout warp as well as full warps.
    for nz in (2, 31, 32, 33, 49, 100):
        index = np.rint(np.linspace(0, case['nz'] - 1, nz)).astype(int)
        def dev(name):
            return cp.asarray(np.ascontiguousarray(case[name][index].reshape(nz, 1, ncol)))
        columns = {k: dev(k) for k in ('u','v','qv','qc','qi','p','dz8w','rthraten')}
        columns.update(t=dev('th')*dev('pi'), exner=dev('pi'))
        surface = {k: cp.asarray(case[k].reshape(1, ncol)) for k in gbmpbl.GBMPBL_SURFACE_INPUTS}
        outputs = []
        for cooperative, kernel in enumerate(kernels):
            tke = dev('tke')
            out = {k: cp.empty(tke.shape, cp.float32) for k in gbmpbl.GBMPBL_COLUMN_OUTPUTS}
            out.update(pblh=cp.empty((1,ncol),cp.float32),ust=cp.empty((1,ncol),cp.float32),
                       kpbl=cp.empty((1,ncol),cp.int32))
            ws = cp.empty(gbmpbl.gbmpbl_workspace_floats(nz,ncol),cp.float32)
            args = (tuple(columns[k] for k in gbmpbl.GBMPBL_COLUMN_INPUTS)+(tke,)
                    +tuple(surface[k] for k in gbmpbl.GBMPBL_SURFACE_INPUTS)
                    +tuple(out[k] for k in gbmpbl.GBMPBL_COLUMN_OUTPUTS)+(tke,)
                    +tuple(out[k] for k in gbmpbl.GBMPBL_SURFACE_OUTPUTS)
                    +(case['dt'],np.int32(nz),np.int64(ncol),ws,np.int32(nz+3),np.int64(0)))
            kernel((ncol if cooperative else (ncol+31)//32,), (32,), args,
                   shared_mem=COOPERATIVE_SLOTS*(nz+3)*4 if cooperative else 0)
            out['tke_pbl'] = tke
            outputs.append(out)
        for name in outputs[0]:
            a,b=(cp.asnumpy(out[name]) for out in outputs)
            assert np.all(np.isfinite(a)), (nz,name)
            assert np.array_equal(a.view(np.uint32), b.view(np.uint32)), (nz,name)
