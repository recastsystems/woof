"""WRF's physics-tendency mass coupling and A-grid-to-C-grid faces, fused.

Every PBL scheme ends its step in :func:`woof.core.physics.
couple_ysu_tendencies` and every cumulus and radiation call in
:func:`~woof.core.physics.couple_column_tendencies`: multiply each A-grid rate
by ``c1h*mut+c2h`` (``calculate_phy_tend``), zero the boundary ring on a
specified or nested domain (``add_a2a``), interpolate momentum onto the C-grid
faces (``add_a2c_u``/``add_a2c_v``) and divide by the map factors
(``rk_addtend_dry``).  As CuPy array expressions that was about 20 full-volume
launches per call, every step, for work that reads each rate once.

Two kernels do the same writes:

* :func:`couple_mass` computes ``chm`` and every coupled scalar rate, the
  boundary ring and the theta map-factor division in one pass, and optionally
  the mass-coupled momentum ``chm*du`` and ``chm*dv`` for the faces;
* :func:`couple_faces` is ``_couple_momentum_to_faces`` for those two arrays:
  the periodic face average, its duplicated last face, the open and specified
  face masks, the staggered map-factor division and the measured closure of
  the duplicated face after that division.

Both are exact transcriptions of the expressions they replace, operation by
operation.  Every multiply, add and divide is its ``_rn`` intrinsic, so NVRTC
cannot contract ``c1h*mut + c2h`` into an FMA where CuPy rounded the product
first, and ``chm`` is recomputed per element with the same two roundings as
the broadcast array.  A zeroed ring element is still divided by its map
factor, as the expressions did.  Callers holding anything other than
C-contiguous float32 CuPy arrays keep the expressions; :func:`ready` is the
test.
"""

from __future__ import annotations

import numpy as np

from woof.wrf_exact import ENABLED as WRF_EXACT

_SOURCE = r"""
extern "C" __global__
void couple_mass_rates(const float* c1h, const float* c2h, const float* mut,
                       const float* msft,
                       const float* in0, const float* in1, const float* in2,
                       const float* in3, const float* in4, const float* in5,
                       float* out0, float* out1, float* out2,
                       float* out3, float* out4, float* out5,
                       const unsigned int present, const unsigned int wanted,
                       const int nslot, const int theta_slot,
                       const float* du, const float* dv,
                       float* mass_u, float* mass_v, const int momentum,
                       const int mask_ring, const int has_msf,
                       const long long nz, const long long ny,
                       const long long nx)
{
    const long long plane = ny * nx;
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nz * plane) return;
    const long long k = idx / plane;
    const long long cell = idx - k * plane;
    const long long j = cell / nx;
    const long long i = cell - j * nx;
    // chm = c1h[:, None, None] * mut[None] + c2h[:, None, None]
    const float chm = __fadd_rn(__fmul_rn(c1h[k], mut[cell]), c2h[k]);
    if (momentum) {
        mass_u[idx] = __fmul_rn(chm, du[idx]);
        mass_v[idx] = __fmul_rn(chm, dv[idx]);
    }
    const bool ring = mask_ring
        && (j == 0 || j == ny - 1 || i == 0 || i == nx - 1);
    const float* const in[6] = {in0, in1, in2, in3, in4, in5};
    float* const out[6] = {out0, out1, out2, out3, out4, out5};
    for (int s = 0; s < nslot; ++s) {
        if (!((wanted >> s) & 1u)) continue;
        // chm * rate, or the zero stack a missing rate stands for.
        float value = ((present >> s) & 1u)
            ? __fmul_rn(chm, in[s][idx]) : 0.0f;
        if (ring) value = 0.0f;
        if (s == theta_slot && has_msf) value = __fdiv_rn(value, msft[cell]);
        out[s][idx] = value;
    }
}

extern "C" __global__
void couple_faces(const float* mass_u, const float* mass_v,
                  const float* msfu, const float* msfv,
                  float* ru, float* rv,
                  const int mask_ring, const int open_x, const int open_y,
                  const int has_msf,
                  const long long nz, const long long ny, const long long nx)
{
    const long long nu = nz * ny * (nx + 1);
    const long long nv = nz * (ny + 1) * nx;
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nu) {
        const long long row = nx + 1;
        const long long k = idx / (ny * row);
        const long long r = idx - k * ny * row;
        const long long j = r / row;
        const long long f = r - j * row;
        // ru = 0.5 * (mass_u + roll(mass_u, 1, axis=2)); face nx repeats 0.
        const long long e = (f == nx) ? 0 : f;
        const long long w = (e == 0) ? nx - 1 : e - 1;
        const long long base = (k * ny + j) * nx;
        float value = __fmul_rn(0.5f,
                                __fadd_rn(mass_u[base + e], mass_u[base + w]));
        if (mask_ring) {
            if (j == 0 || j == ny - 1 || f == 0 || f == nx) value = 0.0f;
        } else if (open_x) {
            if (f == 0 || f == nx) value = 0.0f;
        }
        if (has_msf) {
            // The duplicated face is closed to face 0 after the division.
            const bool closed = !(mask_ring || open_x) && f == nx;
            value = __fdiv_rn(value, msfu[j * row + (closed ? 0 : f)]);
        }
        ru[idx] = value;
        return;
    }
    const long long t = idx - nu;
    if (t >= nv) return;
    const long long k = t / ((ny + 1) * nx);
    const long long r = t - k * (ny + 1) * nx;
    const long long g = r / nx;
    const long long i = r - g * nx;
    // rv = 0.5 * (mass_v + roll(mass_v, 1, axis=1)); face ny repeats 0.
    const long long e = (g == ny) ? 0 : g;
    const long long s = (e == 0) ? ny - 1 : e - 1;
    const long long base = k * ny * nx;
    float value = __fmul_rn(0.5f, __fadd_rn(mass_v[base + e * nx + i],
                                            mass_v[base + s * nx + i]));
    if (mask_ring) {
        if (g == 0 || g == ny || i == 0 || i == nx - 1) value = 0.0f;
    } else if (open_y) {
        if (g == 0 || g == ny) value = 0.0f;
    }
    if (has_msf) {
        const bool closed = !(mask_ring || open_y) && g == ny;
#if GPUWM_WRF_EXACT_C_BIGSTEP
        value = __fmul_rn(value,
            __fdiv_rn(1.0f, msfv[(closed ? 0 : g) * nx + i]));
#else
        value = __fdiv_rn(value, msfv[(closed ? 0 : g) * nx + i]);
#endif
    }
    rv[t] = value;
}
"""

_THREADS = 256
#: Rate slots one mass launch couples.
MAX_SLOTS = 6
_MODULE = None


def _kernel(name):
    global _MODULE
    if _MODULE is None:
        import cupy as cp
        from woof.certify.kernel_manifest import record_module
        from woof.core.kernels import _compile_observed

        # A forecast translation unit, so the kernel manifest records it
        # (tests/test_kernel_manifest.py SITE_FILES).  CuPy's default
        # options, stated so the record says what was compiled.
        module = cp.RawModule(code=_SOURCE, options=())
        _compile_observed(module, "woof.core.tendency_coupling")
        record_module("woof.core.tendency_coupling", source=_SOURCE, options=(),
                      module=module)
        _MODULE = module
    return _MODULE.get_function(name)


def ready(state, *arrays, mut=None) -> bool:
    """True when the state's coupling arrays and ``arrays`` suit the kernels.

    Every array must be a C-contiguous float32 CuPy array; the rates must be
    ``(nz, ny, nx)``, ``mut`` (when given) ``(ny, nx)`` and the metric arrays
    their documented shapes.
    """
    try:
        import cupy as cp
    except ImportError:  # pragma: no cover - the driver imports CuPy
        return False

    def ok(a, shape):
        return (type(a) is cp.ndarray and a.dtype == np.float32
                and a.flags.c_contiguous and a.shape == shape)

    nz, ny, nx = (int(n) for n in state.p.shape)
    metrics = [(state.c1h, (nz,)), (state.c2h, (nz,))]
    if mut is not None:
        metrics.append((mut, (ny, nx)))
    if state.has_msf:
        metrics += [(state.msft, (ny, nx)), (state.msfu, (ny, nx + 1)),
                    (state.msfv, (ny + 1, nx))]
    return (all(ok(a, shape) for a, shape in metrics)
            and all(ok(a, (nz, ny, nx)) for a in arrays))


def _output(cp, shape, previous):
    """Reuse an owned, fully overwritten rate buffer when its layout fits."""
    if (type(previous) is cp.ndarray and previous.shape == shape
            and previous.dtype == cp.float32 and previous.flags.c_contiguous):
        return previous
    return cp.empty(shape, dtype=cp.float32)


def couple_mass(state, mut, slots, *, theta_slot, mask_ring, momentum=None,
                outputs=None):
    """Coupled scalar rates, and ``chm*du``/``chm*dv`` when ``momentum``.

    ``slots`` is a sequence of ``(rate, wanted)``: a wanted slot with no rate
    returns the coupled zero stack, an unwanted one returns ``None``.
    ``mut`` is ``state.total_mu()`` as the caller computed it.  Returns
    ``(outputs, mass_u, mass_v)``.
    """
    import cupy as cp

    nz, ny, nx = (int(n) for n in state.p.shape)
    slots = list(slots)
    if len(slots) > MAX_SLOTS:
        raise ValueError(f"at most {MAX_SLOTS} rate slots, got {len(slots)}")
    shape = (nz, ny, nx)
    previous = [None] * len(slots) if outputs is None else list(outputs)
    if len(previous) != len(slots):
        raise ValueError("one output buffer is required for each rate slot")
    outputs = [_output(cp, shape, old) if wanted else None
               for (_rate, wanted), old in zip(slots, previous)]
    present = sum(1 << s for s, (rate, _w) in enumerate(slots)
                  if rate is not None)
    wanted = sum(1 << s for s, (_r, w) in enumerate(slots) if w)
    # Unused pointer slots name a real array and are never dereferenced.
    anchor = mut
    ins = [rate if rate is not None else anchor for rate, _w in slots]
    outs = [o if o is not None else anchor for o in outputs]
    ins += [anchor] * (MAX_SLOTS - len(ins))
    outs += [anchor] * (MAX_SLOTS - len(outs))
    if momentum is None:
        du = dv = mass_u = mass_v = anchor
    else:
        du, dv = momentum
        mass_u = cp.empty(shape, dtype=cp.float32)
        mass_v = cp.empty(shape, dtype=cp.float32)
    msft = state.msft if state.has_msf else anchor
    total = nz * ny * nx
    _kernel("couple_mass_rates")(
        ((total + _THREADS - 1) // _THREADS,), (_THREADS,),
        (state.c1h, state.c2h, mut, msft, *ins, *outs,
         np.uint32(present), np.uint32(wanted), np.int32(len(slots)),
         np.int32(theta_slot), du, dv, mass_u, mass_v,
         np.int32(momentum is not None), np.int32(bool(mask_ring)),
         np.int32(bool(state.has_msf) and not WRF_EXACT),
         np.int64(nz), np.int64(ny), np.int64(nx)))
    if momentum is None:
        mass_u = mass_v = None
    return outputs, mass_u, mass_v


def couple_faces(state, mass_u, mass_v, *, mask_ring, open_x, open_y,
                 outputs=None):
    """``_couple_momentum_to_faces`` in one launch; returns ``(ru, rv)``."""
    import cupy as cp

    nz, ny, nx = (int(n) for n in state.p.shape)
    old_u, old_v = (None, None) if outputs is None else outputs
    ru = _output(cp, (nz, ny, nx + 1), old_u)
    rv = _output(cp, (nz, ny + 1, nx), old_v)
    if state.has_msf:
        msfu, msfv = state.msfu, state.msfv
    else:
        msfu = msfv = mass_u
    total = ru.size + rv.size
    _kernel("couple_faces")(
        ((total + _THREADS - 1) // _THREADS,), (_THREADS,),
        (mass_u, mass_v, msfu, msfv, ru, rv,
         np.int32(bool(mask_ring)), np.int32(bool(open_x)),
         np.int32(bool(open_y)), np.int32(bool(state.has_msf) and not WRF_EXACT),
         np.int64(nz), np.int64(ny), np.int64(nx)))
    return ru, rv
