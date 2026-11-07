"""Wave state consumed by spray, with a declared parametric fallback.

The fallback has no swell, currents, propagation or storm-memory. It is a
local fetch-limited deep-water JONSWAP approximation, not an ocean model.
"""
import numpy as np

SPECTRUM = dict(min_hz=.01, max_hz=1., bins=96)


class WaveState:
    def __init__(self, cfg, shape):
        import cupy as cp
        from woof.core.kernels import get_kernel
        self.cfg = cfg
        self.n = int(np.prod(shape))
        self.hs, self.phase, self.dissipation, self.slope, self.roughness = (
            cp.zeros(shape, cp.float32) for _ in range(5))
        self.fetch_kernel = get_kernel("tc_waves", "tc_fetch_waves")
        self.spectral_kernel = get_kernel("tc_waves", "tc_spectral_waves")
        self.solver = None

    def bind(self, ocean):
        if self.cfg.tc_wave_source == "coupled":
            self.solver = getattr(ocean, "waves", None)
            if self.solver is None or not hasattr(self.solver, "model"):
                raise ValueError("TC coupled spray requires the resident spectral wave solver")

    def update(self, fields):
        f = fields
        if self.cfg.tc_wave_source == "fetch":
            args = [f[k] for k in ("u10", "v10", "ust", "xland")]
            args += [self.hs, self.phase, self.dissipation, self.slope, self.roughness,
                     np.float32(self.cfg.tc_fetch_m), np.float32(SPECTRUM["min_hz"]),
                     np.float32(SPECTRUM["max_hz"]), np.int32(SPECTRUM["bins"]), np.int32(self.n)]
            self.fetch_kernel(((self.n+127)//128,), (128,), tuple(args))
        else:
            if self.solver is None:
                raise RuntimeError("TC wave state must be bound before stepping")
            m, o = self.solver.model, self.solver.outputs
            self.hs = o.hs
            self.spectral_kernel(((self.n+127)//128,), (128,), (
                m.spectrum, m.frequency, m.df, m.k, m.moments, m.depth,
                o.hs, o.tp, f["ust"], self.phase, self.dissipation, self.slope,
                self.roughness, np.int32(m.nf), np.int32(m.nd), np.int32(self.n)))
