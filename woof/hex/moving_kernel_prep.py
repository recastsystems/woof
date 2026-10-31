"""Compile moving-only kernels beside the running segment, under its lease."""
import time
import numpy as np
from .moving_cuda_work import independent_stream


@independent_stream
def prepare(weights):
    from . import moving_vertical_cuda as vertical
    from . import moving_momentum_cuda as momentum
    from . import moving_balance_cuda as balance
    from . import moving_velocity_cuda as velocity
    from . import moving_saturation_cuda as saturation
    from . import moving_surface_cuda as surface
    from . import moving_horizontal_cuda as horizontal
    from .mpasphys import moving_rates_batch_v842 as rates
    from . import moving_math_cuda as maths
    groups = (
        ('mass', lambda: (vertical._kernel(), *vertical._profile_kernels())),
        ('rates', lambda: (vertical._interface_kernel(), *vertical._rate_kernels(),
                           rates._pack_kernel(4), rates._pack_kernel(8))),
        ('momentum', momentum._kernels),
        ('balance', lambda: (balance._kernel(len(weights.target_dzw)),)),
        ('powers', maths.kernels),
        ('velocity', lambda: (velocity._kernel(np.dtype('float32').str),)),
        ('saturation', lambda: (saturation._kernel(), saturation._adjust_kernel())),
        ('surface', lambda: (surface._kernel(), horizontal._kernel())),
    )
    rows = []
    for name, make in groups:
        began = time.perf_counter()
        kernels = make()
        for kernel in kernels:
            compile_ = getattr(kernel, 'compile', None)
            if compile_ is not None:
                compile_()
        rows.append(dict(group=name, kernels=len(kernels), seconds=time.perf_counter()-began))
    from .moving_pool_projection_cuda import warm
    began=time.perf_counter()
    warm()
    rows.append(dict(group='class-word-gathers', kernels=0, seconds=time.perf_counter()-began))
    began=time.perf_counter()
    size=rates.prepare_uploads(weights)
    rows.append(dict(group='rate-staging',kernels=0,bytes=size,seconds=time.perf_counter()-began))
    return rows
