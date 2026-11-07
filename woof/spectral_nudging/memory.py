"""CPU-only inventory of the prepared nudging adapter and captured operator."""


def memory_inventory(run, controls):
    """Yield (name, category, shape, itemsize, dtype), without loading CUDA.

    forecast.attach packs four physical grids even when a coefficient is zero.
    qv shares temperature's height/PBL buffers. The captured graph retains its
    private pool, FFT outputs and one plan per active field. Price each plan at
    cuFFT's documented worst-case eight complex elements per input element:
    https://docs.nvidia.com/cuda/cufft/index.html#fourier-transform-setup
    This is a bound, not a measurement of a particular card's FFT algorithm.

    The forcing seam can hold both decoded frames beside the old endpoints;
    setup copies these into the graph's persistent endpoint buffers. Price the
    raw frame uploads and returned physical fields for both times, including
    conversion temporaries. These are transient, never shared across domains.
    """
    if controls is None or not controls.enabled:
        return
    nz, ny, nx = int(run.nz), int(run.ny), int(run.nx)
    mass = (nz, ny, nx)
    shapes = dict(u=(nz, ny, nx + 1), v=(nz, ny + 1, nx),
                  temperature=mass, geopotential=(nz + 1, ny, nx), qv=mass)

    def row(name, shape, itemsize=4, dtype='float32', category='nudging'):
        return ('nudging/' + name, category, shape, itemsize, dtype)

    for name in ('temperature', 'geopotential'):
        yield row('physical/' + name, shapes[name])
        yield row('previous/' + name, shapes[name])
    if getattr(run, 'use_theta_m', 0):
        yield row('previous/qv', mass)
    for name in ('u', 'v', 'temperature', 'geopotential'):
        yield row('height/' + name, shapes[name])
        yield row('pbl/' + name, shapes[name][-2:])
        if controls.convention == 'wrf':
            yield row('weight/' + name, shapes[name])
    # Conservative if physics supplies the live plane; exact for the fallback.
    yield row('pbl_height', (ny, nx))
    active = [name for name in shapes if controls.coefficient(name) > 0]
    if 'qv' in active and controls.convention == 'wrf':
        yield row('weight/qv', mass)
    for name in active:
        shape = shapes[name]
        for endpoint in ('start', 'end'):
            yield row('analysis_' + endpoint + '/' + name, shape)
        # Either the state owns cadence storage or the operator allocates it.
        yield row('nudging_tendency_' + name, shape)
        yield row('scratch/' + name, shape, 8, 'complex64')
        if controls.mode == 'spectral':
            for output in ('spectrum', 'filtered'):
                yield row(output + '/' + name, shape, 8, 'complex64')
            yield row('fft_workspace_bound/' + name, (8, *shape), 8, 'complex64')
    yield row('clock', (4,), 8, 'float64')
    yield row('error', (1,), 4, 'uint32')
    yield row('stage_times', (2 * len(active),), 8, 'uint64')
    yield row('stage_sums', (len(active),), 8, 'uint64')
    yield row('stage_count', (1,), 8, 'uint64')
    for time in ('start', 'end'):
        # Full-grid base fields bound the 1-D bases too. Frame() decodes all
        # carriers before selecting positive coefficients in endpoints().
        raw = dict(u=shapes['u'], v=shapes['v'], thp=mass, thb=mass,
                   pressure=mass, php=shapes['geopotential'],
                   phb=shapes['geopotential'], qv=mass)
        for name, shape in raw.items():
            yield row('refresh/' + time + '/raw/' + name, shape, category='transient')
        for name, shape in shapes.items():
            yield row('refresh/' + time + '/physical/' + name, shape, category='transient')
        yield row('refresh/' + time + '/pressure_power', mass, category='transient')
