"""Parent->child nest interpolation operators (Phase 5 Task 10, lane L3).

WRF v4.6.1 transliterations, with the horizontal operator authority being
``share/sint.F`` ITSELF (architecture doc section D, F6 amendment): SINT
(sint.F:2-198) / SINTB (:203-347) are the Smolarkiewicz positive-definite
monotonic transport with field-dependent DONOR/TR4 flux statement functions
(sint.F:36-44) and nonlinear overshoot/undershoot limiters (:66-192);
``interp_fcn_sint`` (share/interp_fcn.F:874-993) and ``bdy_interp1``
(:2423-2626) are only the tile/stagger wrappers around it.  A fixed
precomputed weighted stencil is NOT SINT -- the flux/limiter arithmetic
depends on the field values and is evaluated per field at force time.

What IS precomputed, once at setup, is GEOMETRY ONLY (F6): the donor index
maps and the XIG/XJG offset coefficient tables (sint.F:46-59), built in
FP64 on host (nests are static) and stored FP32 on device.  WRF constructs
XIG/XJG in REAL, i.e. FP32, on every call (sint.F:13-14, :31, :46-57): the
FP64-build/FP32-store is a REGISTERED DEVIATION (architecture deviations
list) whose numeric consequence is tested by N1.5.  The FP64 mirrors in
``woof/verify/npref.py`` consume the SAME FP32-rounded tables.

Device-buffer policy (F4 nest allocation manifest): this module never
registers scratch slots itself.  Every device-side entry point accepts an
optional ``alloc(shape, dtype)`` callable; the owning lanes (L4 preflight
manifest, L6 coupler) pass allocators backed by the child state's scratch
pool with manifest-declared ``nest_*`` slot names.  The default is a plain
CuPy allocation, appropriate for tests only.  BIND-BEFORE-FIRST-USE
(REQUIREMENT on the owning lanes, enforced): manifest consumers MUST call
``NestRegistration.device_tables(alloc=manifest_alloc)`` before the first
driver call on that registration -- drivers upload lazily into
non-manifest allocations otherwise, and a later manifest bind on a
default-bound registration raises rather than silently leaving the
geometry outside the N0 footprint (see ``device_tables``).

This module is importable without CuPy (host geometry + validation); the
CUDA module is compiled on first device use, with ``-fmad=false`` so the
summation paths round like the FP64 mirrors (plan Task 10).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from functools import lru_cache

import numpy as np

from woof.core import constants as c
from woof.core import portable_math as pm
from woof.core.kernels import _preamble

_THREADS = 256

#: Wrapper-specific stagger offsets (the ONLY difference between the two
#: WRF entry points' geometry):
#:
#: - ``interp``: interp_fcn_sint, interp_fcn.F:920-929 --
#:   ``ioff = (nri-1)/2`` when x-staggered, else 0;
#: - ``bdy``: bdy_interp1, interp_fcn.F:2504-2510 --
#:   ``ioff = MAX((nri-1)/2, 1)`` when x-staggered, else 0.
#:
#: The two agree for every ratio >= 3; they differ at nri in {1, 2}, where
#: bdy_interp1's MAX(...,1) shifts the staggered donor lookup by one child
#: index.  Transliterated faithfully, anomaly included (pinned in tests;
#: flagged for the N2 ratio-1 identity-nest fixtures, which exercise
#: exactly this corner).
WRAPPERS = ("interp", "bdy")


def _wrapper_offset(wrapper: str, staggered: bool, ratio: int) -> int:
    if wrapper == "interp":
        return (ratio - 1) // 2 if staggered else 0    # interp_fcn.F:922-925
    if wrapper == "bdy":
        return max((ratio - 1) // 2, 1) if staggered else 0    # :2505-2507
    raise ValueError(f"unknown wrapper {wrapper!r}; expected one of "
                     f"{WRAPPERS}")


def sint_offsets(ratio: int, staggered: bool) -> np.ndarray:
    """The XIG/XJG offset coefficient table for one direction (sint.F:46-59).

    WRF builds, for refinement ratio ``rr`` and sub-cell position ``ip``
    (0-based; XIG index J-1 = ip per the psca pickup ``ip+1 + jp*nri``,
    interp_fcn.F:985/:2583)::

        rioff = 1 if staggered and rr even else 0            (sint.F:49-52)
        XIG   = (rr - 1 - rioff)/(2*rr) - ip/rr              (sint.F:56)

    Built here in FP64, returned FP32 (the registered deviation; WRF
    builds in REAL).  The offset is the SINT advection displacement: the
    child point samples the parent field at ``donor_position - XIG``.
    """
    rr = int(ratio)
    if rr < 1:
        raise ValueError("refinement ratio must be >= 1")
    rioff = 1.0 if (staggered and rr % 2 == 0) else 0.0
    table = np.array([(rr - 1.0 - rioff) / (2.0 * rr) - ip / rr
                      for ip in range(rr)], dtype=np.float64)
    return table.astype(np.float32)


def bdy_width(spec_zone: int, relax_zone: int, spec_bdy_width: int) -> int:
    """LBC strip width, interp_fcn.F:2517 (= 5 for the ratified bundle)::

        sz = MIN(MAX( spec_zone, relax_zone + 1 ),spec_bdy_width)
    """
    return min(max(int(spec_zone), int(relax_zone) + 1), int(spec_bdy_width))


@dataclass
class NestRegistration:
    """Static parent->child SINT geometry for one stagger/wrapper variant.

    Donor maps follow the WRF pickup arithmetic (interp_fcn.F:975-985 /
    :2562-2569): for child 1-based index ``n`` and lookup index
    ``n1 = n + ioff``::

        ci = ipos + (n1-1)/nri        ! donor coarse cell (1-based)
        ip = mod(n1-1, nri)           ! sub-cell position, XIG index

    stored 0-based for woof arrays (``ci0 = ci - 1``).  ``ipos``/``jpos``
    are WRF ``i_parent_start``/``j_parent_start`` (1-based namelist
    semantics; the child's origin cell sits inside parent cell ipos).
    ``xig``/``xjg`` are the FP32-stored offset tables (:data:`sint_offsets`).
    """

    nri: int
    nrj: int
    i_parent_start: int
    j_parent_start: int
    nxc: int                 # child x extent for this stagger
    nyc: int                 # child y extent for this stagger
    nxp: int                 # parent x extent for this stagger
    nyp: int                 # parent y extent for this stagger
    xstag: bool
    ystag: bool
    wrapper: str
    ioff: int
    joff: int
    ci: np.ndarray           # (nxc,) int32, 0-based donor parent i
    ip: np.ndarray           # (nxc,) int32, XIG index
    cj: np.ndarray           # (nyc,) int32
    jp: np.ndarray           # (nyc,) int32
    xig: np.ndarray          # (nri,) float32
    xjg: np.ndarray          # (nrj,) float32
    _device: dict = field(default_factory=dict, repr=False)
    _binding: str = field(default="", repr=False)   # "" | "default" | "manifest"

    def device_tables(self, alloc=None):
        """Upload (once) and return the device geometry tables.

        BIND-BEFORE-FIRST-USE CONTRACT (F4 nest allocation manifest;
        review finding): runtime consumers (lane L6's coupler, lane L4's
        ``--alloc`` preflight) MUST call ``device_tables(alloc=...)``
        with a manifest-backed allocator BEFORE any driver
        (:func:`sint`/:func:`bdy_interp1`/...) touches this
        registration.  Drivers upload lazily with ``alloc=None`` into
        plain CuPy allocations that the F4 registry-equality test cannot
        see -- acceptable for tests ONLY.  The binding is explicit and
        idempotent-checked:

        - first call wins and records its mode (``default`` vs
          ``manifest``);
        - a later ``alloc=...`` call on a default-bound registration
          raises (the silent manifest bypass the review flagged);
        - a repeated ``alloc=...`` call is idempotent (returns the
          existing manifest-backed tables, no re-allocation).

        ``alloc(name, shape, dtype)`` returns the named device buffer (a
        manifest ``nest_*`` scratch slot).  Passing ``name`` makes binding
        independent of dictionary/allocation order.
        """
        if self._device:
            if alloc is not None and self._binding == "default":
                raise RuntimeError(
                    "NestRegistration geometry tables were already "
                    "uploaded into default (non-manifest) allocations by "
                    "an earlier driver call; manifest binding must happen "
                    "BEFORE first use (F4 bind-before-first-use contract)")
            return self._device
        import cupy as cp

        def _up(name, host):
            if alloc is None:
                return cp.asarray(host)
            buf = alloc(name, host.shape, host.dtype)
            buf[...] = cp.asarray(host)
            return buf

        self._device = {name: _up(name, getattr(self, name))
                        for name in ("ci", "ip", "cj", "jp",
                                     "xig", "xjg")}
        self._binding = "default" if alloc is None else "manifest"
        return self._device


def _donor_maps(n_child, ratio, pos, ioff, n_parent, label):
    """0-based donor/sub-cell maps for one direction (interp_fcn.F:975-985)."""
    n1 = np.arange(1, n_child + 1, dtype=np.int64) + ioff
    ci = pos + (n1 - 1) // ratio          # 1-based donor cell
    ip = (n1 - 1) % ratio
    lo, hi = int(ci.min()) - 1, int(ci.max()) - 1     # 0-based extremes
    if lo < 2 or hi > n_parent - 3:
        raise ValueError(
            f"child {label} extent needs parent donors {lo}..{hi} with a "
            f"+-2 SINT stencil (sint.F ior=2), outside the parent extent "
            f"{n_parent}; move the nest inward or enlarge the parent")
    return (ci - 1).astype(np.int32), ip.astype(np.int32)


def register_nest(*, nri, nrj, i_parent_start, j_parent_start,
                  child_nx, child_ny, parent_nx, parent_ny,
                  stagger="", wrapper="interp") -> NestRegistration:
    """Build the static SINT geometry for one field stagger.

    Stagger dispatch per the ratified section-D contract (the
    interp_fcn_sint call sites): ``"x"`` for u, ``"y"`` for v, ``""`` for
    mass scalars/mu and for w/ph full levels (the horizontal stencil is
    unchanged by z-staggering).  ``child_nx``/``parent_nx`` are the MASS
    counts; staggered extents (+1) are derived here.  WRF's sint assumes a
    square refinement ratio (``rr = nint(sqrt(float(nf)))``, sint.F:46);
    non-square ratios are rejected loudly.
    """
    if int(nri) != int(nrj):
        raise ValueError("SINT assumes a square refinement ratio "
                         "(rr = nint(sqrt(float(nf))), sint.F:46)")
    if stagger not in ("", "x", "y"):
        raise ValueError('stagger must be "", "x" or "y" '
                         "(interp_fcn.F:860: 'there are only two "
                         "staggerings I accept')")
    nri, nrj = int(nri), int(nrj)
    xstag, ystag = stagger == "x", stagger == "y"
    nxc = int(child_nx) + (1 if xstag else 0)
    nyc = int(child_ny) + (1 if ystag else 0)
    nxp = int(parent_nx) + (1 if xstag else 0)
    nyp = int(parent_ny) + (1 if ystag else 0)
    ioff = _wrapper_offset(wrapper, xstag, nri)
    joff = _wrapper_offset(wrapper, ystag, nrj)
    ci, ip = _donor_maps(nxc, nri, int(i_parent_start), ioff, nxp, "x")
    cj, jp = _donor_maps(nyc, nrj, int(j_parent_start), joff, nyp, "y")
    return NestRegistration(
        nri=nri, nrj=nrj, i_parent_start=int(i_parent_start),
        j_parent_start=int(j_parent_start), nxc=nxc, nyc=nyc,
        nxp=nxp, nyp=nyp, xstag=xstag, ystag=ystag, wrapper=wrapper,
        ioff=ioff, joff=joff, ci=ci, ip=ip, cj=cj, jp=jp,
        xig=sint_offsets(nri, xstag), xjg=sint_offsets(nrj, ystag))


@lru_cache(maxsize=None)
def _nest_module(preparation=False):
    """Compile nest.cu under the forecast or prepared-input arithmetic contract.

    A dedicated loader (rather than ``kernels.load_module``) because this
    module alone carries the ``-fmad=false`` requirement: the SINT/TR4
    summation paths must round like the FP64 mirrors for the pinned
    fp64_mirror_floor comparator (nest_gates.FP64_MIRROR_MAX_ULPS).
    Fast forecast numerics relaxes that requirement; preparation keeps it
    to match the host FP32 operator regardless of the forecast mode.
    """
    import cupy as cp
    from pathlib import Path
    src = _preamble() + (Path(__file__).parent / "kernels"
                         / "nest.cu").read_text()
    options = ("-std=c++17", "-fmad=false")
    if preparation:
        # The host SINT rounds each FP32 operation. Contracting its CUDA
        # counterpart changes the child's terrain before the column physics.
        # This module is separate from the forecast's fast nest operators.
        options += ("-DGPUWM_PREPARATION_IDENTITY=1",)
    mod = cp.RawModule(code=src,
                       options=options,
                       name_expressions=None)
    mod.compile()
    from woof.certify.kernel_manifest import record_module
    name = "nest_preparation" if preparation else "nest"
    record_module(f"woof.core.nest_interp:{name}", source=src,
                  options=options, module=mod)
    return mod


def _kernel(name):
    return _nest_module().get_function(name)


def _launch(kernel, count, args):
    kernel(((count + _THREADS - 1) // _THREADS,), (_THREADS,), args)


def _as3d(arr):
    return arr[None] if arr.ndim == 2 else arr


def _check_table(buf, shape, what):
    """Fail loud on a mis-shaped/mis-typed/non-contiguous device buffer.

    The kernels index linearly over ``prod(shape)`` threads; a wrong
    preallocated buffer (a manifest slot with a drifted shape formula)
    must raise here instead of writing out of bounds on device.
    """
    if tuple(buf.shape) != tuple(shape):
        raise ValueError(f"{what} has shape {tuple(buf.shape)}, "
                         f"expected {tuple(shape)}")
    if buf.dtype != np.float32:
        raise ValueError(f"{what} must be float32, got {buf.dtype}")
    flags = getattr(buf, "flags", None)
    if flags is not None and not flags.c_contiguous:
        raise ValueError(f"{what} must be C-contiguous")
    return buf


_SINT_EP = float(np.float32(1.0e-10))


def _numpy_sint_donor(y1, y2, a):
    """WRF ``DONOR`` statement used by the host SINT implementation."""
    sign = np.copysign(1.0, a)
    return (y1 * np.maximum(0.0, sign)
            - y2 * np.minimum(0.0, sign)) * a


def _numpy_sint_tr4(ym1, y0, yp1, yp2, a):
    """WRF ``TR4`` statement used by the host SINT implementation."""
    one12 = 1.0 / 12.0
    one24 = 1.0 / 24.0
    return (a * one12 * (7.0 * (yp1 + y0) - (yp2 + ym1))
            - a * a * one24 * (15.0 * (yp1 - y0) - (yp2 - ym1))
            - a * a * a * one12 * ((yp1 + y0) - (yp2 + ym1))
            + a * a * a * a * one24
            * (3.0 * (yp1 - y0) - (yp2 - ym1)))


def _numpy_sint_pass(ym2, ym1, y0, yp1, yp2, a):
    """One WRF SINTB residual-advection pass, evaluated in FP32."""
    fl0 = _numpy_sint_donor(ym1, y0, a)
    fl1 = _numpy_sint_donor(y0, yp1, a)
    w = y0 - (fl1 - fl0)
    maximum = np.maximum(np.maximum(ym1, y0), np.maximum(yp1, w))
    minimum = np.minimum(np.minimum(ym1, y0), np.minimum(yp1, w))
    f0 = _numpy_sint_tr4(ym2, ym1, y0, yp1, a) - fl0
    f1 = _numpy_sint_tr4(ym1, y0, yp1, yp2, a) - fl1
    pp0, pn0 = np.maximum(0.0, f0), np.minimum(0.0, f0)
    pp1, pn1 = np.maximum(0.0, f1), np.minimum(0.0, f1)
    overshoot = (maximum - w) / (-pn1 + pp0 + _SINT_EP)
    undershoot = (w - minimum) / (pp1 - pn0 + _SINT_EP)
    c0 = (pp0 * np.minimum(1.0, overshoot)
          + pn0 * np.minimum(1.0, undershoot))
    c1 = (pp1 * np.minimum(1.0, undershoot)
          + pn1 * np.minimum(1.0, overshoot))
    return w - (c1 - c0)


def _numpy_sint(cfld, reg):
    """Production host mirror of the FP32 ``nest_sint`` CUDA kernel.

    This implementation deliberately lives in the production core rather
    than ``woof.verify``: parallel CPU hierarchy finalization is a public
    RW-WPS runtime path and must not depend on developer-only verification
    modules omitted from the standalone distribution.
    """
    parent = np.asarray(cfld, dtype=np.float32)
    squeeze = parent.ndim == 2
    if squeeze:
        parent = parent[None]
    if parent.ndim != 3:
        raise ValueError("parent field must be 2-D or 3-D")
    _, nyp, nxp = parent.shape
    ci = np.asarray(reg.ci, dtype=np.int64)
    cj = np.asarray(reg.cj, dtype=np.int64)
    if (ci.min() < 2 or ci.max() > nxp - 3
            or cj.min() < 2 or cj.max() > nyp - 3):
        raise ValueError("parent field too small for the +-2 SINT stencil "
                         "around the registered donor cells")
    ax = np.asarray(reg.xig, dtype=np.float32)[
        np.asarray(reg.ip, dtype=np.int64)][None, None, :]
    ay = np.asarray(reg.xjg, dtype=np.float32)[
        np.asarray(reg.jp, dtype=np.int64)][None, :, None]
    ii = ci[None, :]
    jj = cj[:, None]
    rows = []
    for joff in range(-2, 3):
        row = [parent[:, jj + joff, ii + ioff]
               for ioff in range(-2, 3)]
        rows.append(_numpy_sint_pass(*row, ax))
    result = _numpy_sint_pass(*rows, ay)
    return result[0] if squeeze else result


def window_registration(reg: NestRegistration, window):
    """Exact SINT geometry and minimal donor rectangle for a child window.

    ``window`` is two bounded unit-step slices in this registration's field
    extent (including its staggered endpoint). Subcell phases are sliced from
    the original tables, never recomputed from a fictitious slab placement.
    Donor indices are translated by the crop origin; the two-cell SINT halo
    remains present on every side. Device tables belong to the new geometry.
    """
    if not isinstance(window, tuple) or len(window) != 2:
        raise ValueError("SINT window must be a pair of bounded slices")
    normalized = []
    for axis, extent in zip(window, (reg.nyc, reg.nxc)):
        if (not isinstance(axis, slice) or axis.step not in (None, 1)
                or axis.start is None or axis.stop is None
                or not isinstance(axis.start, (int, np.integer))
                or not isinstance(axis.stop, (int, np.integer))
                or not 0 <= axis.start < axis.stop <= extent):
            raise ValueError("SINT window slices must be nonempty, bounded and unit-step")
        normalized.append(slice(int(axis.start), int(axis.stop)))
    sy, sx = normalized
    ci, cj = reg.ci[sx], reg.cj[sy]
    x0, x1 = int(ci.min()) - 2, int(ci.max()) + 3
    y0, y1 = int(cj.min()) - 2, int(cj.max()) + 3
    if x0 < 0 or y0 < 0 or x1 > reg.nxp or y1 > reg.nyp:
        raise ValueError("SINT window's donor halo is outside the parent")
    cropped = replace(
        reg, nxc=sx.stop-sx.start, nyc=sy.stop-sy.start,
        nxp=x1-x0, nyp=y1-y0,
        i_parent_start=reg.i_parent_start-x0,
        j_parent_start=reg.j_parent_start-y0,
        ci=np.array(ci-x0, dtype=np.int32),
        cj=np.array(cj-y0, dtype=np.int32),
        ip=reg.ip[sx].copy(), jp=reg.jp[sy].copy(),
        xig=reg.xig.copy(), xjg=reg.xjg.copy(), _device={}, _binding="")
    return cropped, (slice(y0, y1), slice(x0, x1))


def sint(cfld, reg: NestRegistration, *, out=None, alloc=None, window=None,
         preparation=False):
    """SINT-interpolate a parent device field onto the child grid.

    ``cfld`` is ``(ny_p, nx_p)`` or ``(nz, ny_p, nx_p)`` FP32 (already
    coupled by the caller where WRF couples -- this operator is the bare
    horizontal interpolation the wrappers share).  Returns the child-extent
    field with the same leading shape.

    ``window`` requests only a child rectangle. Only its exact donor rectangle
    plus the SINT halo is copied; no full child intermediate is constructed.
    ``preparation`` preserves the host's separate FP32 multiply/add rounding
    for prepared child inputs, independently of the forecast numerics mode.
    """
    if window is not None:
        if tuple(cfld.shape[-2:]) != (reg.nyp, reg.nxp):
            raise ValueError("parent field does not match the registration extent")
        cropped, donor = window_registration(reg, window)
        if isinstance(cfld, np.ndarray):
            payload = np.ascontiguousarray(cfld[(...,) + donor])
        else:
            import cupy as cp
            payload = cp.ascontiguousarray(cfld[(...,) + donor])
        return sint(payload, cropped, out=out, alloc=alloc,
                    preparation=preparation)
    if isinstance(cfld, np.ndarray):
        if alloc is not None:
            raise ValueError("NumPy SINT does not accept a device allocator")
        result = np.ascontiguousarray(
            _numpy_sint(cfld, reg), dtype=np.float32)
        if out is None:
            return result
        _check_table(_as3d(out), _as3d(result).shape, "sint out= buffer")
        out[...] = result
        return out

    c3 = _as3d(cfld)
    nz, nyp, nxp = c3.shape
    if (nyp, nxp) != (reg.nyp, reg.nxp):
        raise ValueError(f"parent field {c3.shape[1:]} does not match the "
                         f"registration extent {(reg.nyp, reg.nxp)}")
    if out is not None:
        _check_table(_as3d(out), (nz, reg.nyc, reg.nxc), "sint out= buffer")
        out = _as3d(out)
    else:
        import cupy as cp
        out = (alloc((nz, reg.nyc, reg.nxc), np.float32) if alloc is not None
               else cp.empty((nz, reg.nyc, reg.nxc), dtype=cp.float32))
    dev = reg.device_tables(alloc=None)
    kernel = (_nest_module(preparation=True).get_function("nest_sint")
              if preparation else _kernel("nest_sint"))
    _launch(kernel, nz * reg.nyc * reg.nxc, (
        c3, out, dev["ci"], dev["ip"], dev["cj"], dev["jp"],
        dev["xig"], dev["xjg"],
        np.int32(nz), np.int32(reg.nyc), np.int32(reg.nxc),
        np.int32(nyp), np.int32(nxp)))
    return out if cfld.ndim == 3 else out[0]


_SIDES = ("west", "east", "south", "north")


def bdy_interp1(cfld, nfld, reg: NestRegistration, *,
                parent_dt_fp32, parent_interval_ticks=None,
                spec_zone=1, relax_zone=4, spec_bdy_width=5,
                out=None, alloc=None, sides=None):
    """Build the child's four-side boundary VALUE/TENDENCY device tables.

    Transliteration of ``bdy_interp1`` (interp_fcn.F:2423-2626): VALUE is
    the child's current coupled state (:2584); TENDENCY is
    ``rdt * (SINT(coupled parent @ t+dtp) - child)`` (:2583) with REAL*8
    ``rdt = 1.D0/cdt`` (:2480/:2500).

    TENDENCY-INTERVAL CONTRACT (F7 amendment): ``cdt`` IS THE
    PARENT/COARSE-GRID STEP -- interp_fcn.F:2320 declares ``cdt, ndt`` as
    "Time step size for CG and FG" (dummy declarations :2345, :2472).  The
    API therefore carries ``parent_dt_fp32``/``parent_interval_ticks``
    explicitly; the child's own dt has no business here (it governs only
    the Davies application).  ``reg`` must be built with
    ``wrapper="bdy"`` (the MAX((nri-1)/2,1) stagger offset, :2504-2510).

    Returns ``{"west"|"east"|"south"|"north": (value, tendency)}`` FP32
    device tables in the Phase-4 lateral_bc.py orientation: west/east
    ``(nz, ny_c, sz)``, south/north ``(nz, sz, nx_c)``, width index 0 at
    the domain edge.  ``out`` may pass preallocated tables of exactly that
    layout (the L6 rolling nest_* slots); ``alloc(shape, dtype)`` places
    fresh ones.
    """
    selected = _SIDES if sides is None else tuple(sides)
    if not selected or len(set(selected)) != len(selected) or any(
            side not in _SIDES for side in selected):
        raise ValueError("boundary sides must be a nonempty unique selection")
    if reg.wrapper != "bdy":
        raise ValueError("bdy_interp1 needs a wrapper='bdy' registration "
                         "(stagger ioff = MAX((nri-1)/2,1), "
                         "interp_fcn.F:2504-2510)")
    if parent_interval_ticks is not None and int(parent_interval_ticks) <= 0:
        raise ValueError("parent_interval_ticks must be positive")
    cdt = np.float32(parent_dt_fp32)
    if not np.isfinite(cdt) or cdt <= 0.0:
        raise ValueError("parent_dt_fp32 must be a positive finite parent "
                         "step (interp_fcn.F:2320 cdt semantics)")
    c3, n3 = _as3d(cfld), _as3d(nfld)
    nz, nyp, nxp = c3.shape
    if (nyp, nxp) != (reg.nyp, reg.nxp):
        raise ValueError("parent field does not match the registration")
    if n3.shape != (nz, reg.nyc, reg.nxc):
        raise ValueError("child field does not match the registration")
    sz = bdy_width(spec_zone, relax_zone, spec_bdy_width)     # :2517
    shapes = {"west": (nz, reg.nyc, sz), "east": (nz, reg.nyc, sz),
              "south": (nz, sz, reg.nxc), "north": (nz, sz, reg.nxc)}
    if out is None:
        import cupy as cp

        def _new(shape):
            return (alloc(shape, np.float32) if alloc is not None
                    else cp.empty(shape, dtype=cp.float32))
        out = {side: (_new(shapes[side]), _new(shapes[side]))
               for side in selected}
    for side in selected:            # fail loud BEFORE any device work
        value, tendency = out[side]
        _check_table(value, shapes[side], f"{side} value table")
        _check_table(tendency, shapes[side], f"{side} tendency table")
    dev = reg.device_tables(alloc=None)
    if selected == _SIDES:
        tables = tuple(array for side in _SIDES for array in out[side])
        count = 2 * nz * sz * (reg.nyc + reg.nxc)
        _launch(_kernel("nest_bdy_interp1_all_sides"), count, (
            c3, n3, *tables,
            dev["ci"], dev["ip"], dev["cj"], dev["jp"],
            dev["xig"], dev["xjg"], cdt, np.int32(sz),
            np.int32(nz), np.int32(reg.nyc), np.int32(reg.nxc),
            np.int32(nyp), np.int32(nxp)))
        return out
    for index, side in enumerate(_SIDES):
        if side not in selected:
            continue
        value, tendency = out[side]
        count = nz * sz * (reg.nyc if index < 2 else reg.nxc)
        _launch(_kernel("nest_bdy_interp1"), count, (
            c3, n3, value, tendency,
            dev["ci"], dev["ip"], dev["cj"], dev["jp"],
            dev["xig"], dev["xjg"], cdt,
            np.int32(index), np.int32(sz),
            np.int32(nz), np.int32(reg.nyc), np.int32(reg.nxc),
            np.int32(nyp), np.int32(nxp)))
    return out


def blend_terrain(ter_interpolated, ter_input, *,
                  spec_bdy_width=5, blend_width=5,
                  domain_shape=None, origin=(0, 0)):
    """Blend parent-interpolated and fine terrain in place on ``ter_input``.

    ``blend_terrain`` (dyn_em/nest_init_utils.F:712-785): rows <=
    spec_bdy_width take the parent value (:766-769); the next blend_width
    frames blend with weights ``blend_cell/(blend_width+1)`` (:759-765);
    the interior stays fine.  Applied to ht, mub AND phb at child init
    (mediation_integrate.F:733-741; blending all three is the ratified
    adjudication).  2-D ``(ny, nx)`` or 3-D ``(nk, ny, nx)`` FP32 device
    arrays; k is inert.

    A reconstruction window supplies its mass-grid ``domain_shape`` and
    zero-based ``origin``. Blend weights still refer to the true domain
    edges; an internal slab edge must never become a new boundary frame.
    """
    ny, nx = ter_input.shape[-2:]
    domain_ny, domain_nx = (ny, nx) if domain_shape is None else domain_shape
    j0, i0 = origin
    if (any(not isinstance(v, (int, np.integer)) for v in
            (domain_ny, domain_nx, j0, i0))
            or j0 < 0 or i0 < 0 or ny < 1 or nx < 1
            or j0 + ny > domain_ny or i0 + nx > domain_nx):
        raise ValueError("terrain window lies outside its domain extent")
    if isinstance(ter_input, np.ndarray):
        if not isinstance(ter_interpolated, np.ndarray):
            raise TypeError("NumPy terrain blending requires NumPy operands")
        coarse = np.asarray(ter_interpolated, dtype=np.float64)
        fine = np.asarray(ter_input, dtype=np.float64)
        if coarse.shape != fine.shape:
            raise ValueError("blend operands must have identical shapes")
        squeeze = fine.ndim == 2
        work_f = fine[None] if squeeze else fine
        work_c = coarse[None] if squeeze else coarse
        _, ny, nx = work_f.shape
        sbw, width = int(spec_bdy_width), int(blend_width)
        ide, jde = domain_nx + 1, domain_ny + 1
        i1 = np.arange(i0 + 1, i0 + nx + 1)[None, None, :]
        j1 = np.arange(j0 + 1, j0 + ny + 1)[None, :, None]
        reciprocal = 1.0 / (width + 1)
        blended_output = work_f.copy()
        for blend_cell in range(width, 0, -1):
            hit = (
                (i1 == sbw + blend_cell)
                | (j1 == sbw + blend_cell)
                | (i1 == ide - sbw - blend_cell)
                | (j1 == jde - sbw - blend_cell)
            )
            blended = (
                blend_cell * work_f
                + (width + 1 - blend_cell) * work_c
            ) * reciprocal
            blended_output = np.where(hit, blended, blended_output)
        specified = (
            (i1 <= sbw) | (j1 <= sbw)
            | (i1 >= ide - sbw) | (j1 >= jde - sbw)
        )
        blended_output = np.where(specified, work_c, blended_output)
        # Stored in the target's own dtype: an FP32 state array rounds
        # exactly as before, and a float64 target keeps the float64 blend
        # (the child base the EOS correction is built from, see
        # woof.ingest.nest_init._blend_terrain_triple).
        ter_input[...] = blended_output[0] if squeeze else blended_output
        return ter_input

    ci3, ti3 = _as3d(ter_interpolated), _as3d(ter_input)
    if ci3.shape != ti3.shape:
        raise ValueError("blend operands must have identical shapes")
    nk, ny, nx = ti3.shape
    _launch(_kernel("nest_blend_terrain"), nk * ny * nx, (
        ci3, ti3, np.int32(spec_bdy_width), np.int32(blend_width),
        np.int32(nk), np.int32(ny), np.int32(nx),
        np.int32(domain_ny), np.int32(domain_nx),
        np.int32(j0), np.int32(i0)))
    return ter_input


def adjust_tempqv(mub, save_mub, c3, c4, p_top, th, pp, qv, *,
                  use_theta_m=0):
    """Correct theta/qv in place for the blended-MUB pressure change.

    ``adjust_tempqv`` (dyn_em/nest_init_utils.F:812-890), called at child
    init right after blend_terrain (mediation_integrate.F:749): RH is
    conserved across the ``save_mub -> mub`` column-mass change; ``th``
    and ``qv`` are updated in place, ``pp`` is read only (the Fortran
    INTENT(INOUT) never writes it; its ``znw`` dummy is unused and
    dropped).  ``c3``/``c4`` are the half-level hybrid coefficients
    (state ``c3h``/``c4h``); ``R_v/R_d`` is passed from the constants
    module.  Evaluated in FP64, stored FP32 (documented deviation -- see
    the kernel header).
    """
    if isinstance(th, np.ndarray):
        if not all(isinstance(value, np.ndarray)
                   for value in (mub, save_mub, c3, c4, pp, qv)):
            raise TypeError("NumPy adjust_tempqv requires NumPy operands")
        mub64 = np.asarray(mub, dtype=np.float64)
        saved64 = np.asarray(save_mub, dtype=np.float64)
        c3_64 = np.asarray(c3, dtype=np.float64)[:, None, None]
        c4_64 = np.asarray(c4, dtype=np.float64)[:, None, None]
        th64 = np.asarray(th, dtype=np.float64)
        pp64 = np.asarray(pp, dtype=np.float64)
        qv64 = np.asarray(qv, dtype=np.float64)
        rvord = c.RVOVRD
        p_old = c4_64 + c3_64 * saved64[None] + float(p_top) + pp64
        if use_theta_m == 1:
            tc = ((th64 + 300.0) * pm.power(p_old / 1.0e5, 2.0 / 7.0)
                  / (1.0 + rvord * qv64) - 273.15)
        else:
            tc = ((th64 + 300.0) * pm.power(p_old / 1.0e5, 2.0 / 7.0)
                  - 273.15)
        es = 610.78 * pm.exp(17.0809 * tc / (234.175 + tc))
        rh = (qv64 * p_old / (0.622 + qv64)) / es
        p_new = c4_64 + c3_64 * mub64[None] + float(p_top) + pp64
        if use_theta_m == 1:
            thloc = (th64 + 300.0) / (1.0 + rvord * qv64)
        else:
            thloc = th64 + 300.0
        dth1 = (-191.86e-3 * thloc / (p_new + p_old)
                * (p_new - p_old))
        dth = (-191.86e-3 * (thloc + 0.5 * dth1) / (p_new + p_old)
               * (p_new - p_old))
        if use_theta_m == 1:
            adjusted_th = ((thloc + dth) * (1.0 + rvord * qv64)
                           - 300.0)
        else:
            adjusted_th = thloc + dth - 300.0
        tc = ((thloc + dth) * pm.power(p_new / 1.0e5, 2.0 / 7.0)
              - 273.15)
        es = 610.78 * pm.exp(17.0809 * tc / (234.175 + tc))
        vapor_pressure = rh * es
        adjusted_qv = 0.622 * vapor_pressure / (p_new - vapor_pressure)
        th[...] = np.asarray(adjusted_th, dtype=np.float32)
        qv[...] = np.asarray(adjusted_qv, dtype=np.float32)
        return th, qv

    th3, pp3, qv3 = th, pp, qv
    nz, ny, nx = th3.shape
    if pp3.shape != th3.shape or qv3.shape != th3.shape:
        raise ValueError("th/pp/qv shapes differ")
    if mub.shape != (ny, nx) or save_mub.shape != (ny, nx):
        raise ValueError("mub/save_mub must be (ny, nx)")
    if c3.shape != (nz,) or c4.shape != (nz,):
        raise ValueError("c3/c4 must be half-level (nz,) columns")
    _launch(_kernel("nest_adjust_tempqv"), nz * ny * nx, (
        mub, save_mub, c3, c4, np.float64(p_top), th3, pp3, qv3,
        np.float64(c.RVOVRD), np.int32(use_theta_m),
        np.int32(nz), np.int32(ny), np.int32(nx)))
    return th, qv


def feedback_parent_bounds(reg: NestRegistration, *, spec_zone=1
                           ) -> tuple[int, int, int, int]:
    """Return the 0-based inclusive parent rectangle written by feedback.

    These are ``copy_fcn``'s WRF v4.6.1 loop bounds.  The child specified
    zone is excluded in parent-cell units; staggered directions retain the
    coincident high-side face.
    """
    spec_zone = int(spec_zone)
    if spec_zone < 0:
        raise ValueError("feedback spec_zone must be non-negative")
    istag = 0 if reg.xstag else 1
    jstag = 0 if reg.ystag else 1
    nide_span = reg.nxc - (1 if reg.xstag else 0)
    njde_span = reg.nyc - (1 if reg.ystag else 0)
    ci_lo = reg.i_parent_start + spec_zone
    ci_hi = (reg.i_parent_start + nide_span // reg.nri
             - istag - spec_zone)
    cj_lo = reg.j_parent_start + spec_zone
    cj_hi = (reg.j_parent_start + njde_span // reg.nrj
             - jstag - spec_zone)
    # Fortran indices are 1-based; Python/CUDA array indices are 0-based.
    return ci_lo - 1, ci_hi - 1, cj_lo - 1, cj_hi - 1


def _copy_common(cfld, nfld, reg: NestRegistration, spec_zone):
    c3, n3 = _as3d(cfld), _as3d(nfld)
    nz, nyp, nxp = c3.shape
    if (nyp, nxp) != (reg.nyp, reg.nxp) or n3.shape[1:] != (reg.nyc,
                                                            reg.nxc):
        raise ValueError("fields do not match the registration extents")
    if n3.shape[0] != nz:
        raise ValueError("parent/child level counts differ")
    args = (np.int32(reg.i_parent_start), np.int32(reg.j_parent_start),
            np.int32(reg.nri), np.int32(reg.nrj), np.int32(spec_zone),
            np.int32(reg.xstag), np.int32(reg.ystag),
            np.int32(nz), np.int32(nyp), np.int32(nxp),
            np.int32(reg.nyc), np.int32(reg.nxc))
    ci_lo, ci_hi, cj_lo, cj_hi = feedback_parent_bounds(
        reg, spec_zone=spec_zone)
    ni_cells = ci_hi - ci_lo + 1
    nj_cells = cj_hi - cj_lo + 1
    count = nz * max(ni_cells, 0) * max(nj_cells, 0)
    return c3, n3, args, count


def copy_fcn(cfld, nfld, reg: NestRegistration, *, spec_zone=1,
             parent_window=None, child_window=None):
    """Feedback cell-average of the child onto the parent, in place.

    ``copy_fcn`` (interp_fcn.F:1397-1742), BOTH parity branches: odd
    (:1463-1562) and even (:1567-1737) ratios cell-average all nri*nrj
    child points at weight 1/(nri*nrj) for mass fields and 1/nri along
    the face for u/v.  DORMANT Phase-5b machinery (design D4): the
    kernels land and are oracled at N1; the
    ``feedback_prepare``/``feedback_commit``/``feedback_finalize``
    transaction is a no-op at feedback=0 and nothing calls this in Phase 5
    runs.
    """
    if parent_window is None and child_window is None:
        c3, n3, args, count = _copy_common(cfld, nfld, reg, spec_zone)
        ilo, ihi, jlo, jhi = feedback_parent_bounds(reg, spec_zone=spec_zone)
        origins = (0, 0, 0, 0, ilo, jlo, max(0, ihi-ilo+1), max(0, jhi-jlo+1))
    elif parent_window is None or child_window is None:
        raise ValueError("windowed feedback requires both operand windows")
    else:
        donor = feedback_child_window(reg, parent_window, spec_zone=spec_zone)
        if child_window != donor:
            raise ValueError("feedback child window does not match exact donors")
        c3, n3 = _as3d(cfld), _as3d(nfld)
        py, px = parent_window
        cy, cx = child_window
        if c3.shape != (n3.shape[0], py.stop-py.start, px.stop-px.start):
            raise ValueError("parent feedback operand does not match its window")
        if n3.shape[1:] != (cy.stop-cy.start, cx.stop-cx.start):
            raise ValueError("child feedback operand does not match its window")
        nz, nyp, nxp = c3.shape
        args = tuple(np.int32(v) for v in (
            reg.i_parent_start, reg.j_parent_start, reg.nri, reg.nrj,
            spec_zone, reg.xstag, reg.ystag, nz, nyp, nxp,
            n3.shape[1], n3.shape[2]))
        origins = (px.start, py.start, cx.start, cy.start,
                   px.start, py.start, nxp, nyp)
        count = c3.size
    if count > 0:
        _launch(_kernel("nest_copy_fcn"), count,
                (c3, n3, *args, *(np.int32(v) for v in origins)))
    return cfld


def feedback_child_window(reg, parent_window, *, spec_zone=1):
    """Exact child donor rectangle for a bounded raw restriction launch."""
    if len(parent_window) != 2:
        raise ValueError("feedback window must have two bounded slices")
    y, x = parent_window
    ilo, ihi, jlo, jhi = feedback_parent_bounds(reg, spec_zone=spec_zone)
    for axis, lo, hi in ((x, ilo, ihi), (y, jlo, jhi)):
        if (not isinstance(axis, slice) or axis.step not in (None, 1)
                or axis.start is None or axis.stop is None
                or axis.start < lo or axis.stop > hi + 1
                or axis.start >= axis.stop):
            raise ValueError("feedback window lies outside the restriction rectangle")
    ri, rj = reg.nri, reg.nrj
    odd = rj % 2 != 0
    if odd:
        if reg.xstag:
            ij0, ij1, stride = (ri+1)//2, (ri+1)//2 + ri*(ri-1), ri
            oi, oj = 0, rj//2
        elif reg.ystag:
            ij0 = (rj*rj+1)//2 - rj//2
            ij1, stride, oi, oj = ij0+rj-1, 1, ri//2, 0
        else:
            ij0, ij1, stride, oi, oj = 1, ri*rj, 1, ri//2, rj//2
    else:
        ij0, ij1, stride, oi, oj = 1, ri*rj, 1, 0, 0
        if reg.xstag:
            stride = ri
        elif reg.ystag:
            ij1 = ri
    points = range(ij0, ij1+1, stride)
    dx = [(ij-1) % ri - (ri//2 if odd else 0) + oi for ij in points]
    dy = [(ij-1) // ri - (rj//2 if odd else 0) + oj for ij in points]
    i0 = (x.start+1-reg.i_parent_start)*ri + min(dx)
    i1 = (x.stop-reg.i_parent_start)*ri + max(dx) + 1
    j0 = (y.start+1-reg.j_parent_start)*rj + min(dy)
    j1 = (y.stop-reg.j_parent_start)*rj + max(dy) + 1
    if i0 < 0 or j0 < 0 or i1 > reg.nxc or j1 > reg.nyc:
        raise ValueError("feedback donors lie outside the child extent")
    return slice(j0, j1), slice(i0, i1)


def copy_fcnm(cfld, nfld, reg: NestRegistration, *, spec_zone=1):
    """Feedback 1-pt pick for masked REAL fields, in place.

    ``copy_fcnm`` (interp_fcn.F:1747-1824): center child on odd ratios,
    SW-corner nearest neighbor on even ratios (:1806).  WOOF's live
    feedback inventory currently has no masked fields; this exact WRF
    behavior is retained as the certification-matching default if that
    inventory is extended.
    """
    c3, n3, args, count = _copy_common(cfld, nfld, reg, spec_zone)
    if count > 0:
        _launch(_kernel("nest_copy_fcnm"), count, (c3, n3, *args))
    return cfld


def copy_fcni(cfld, nfld, reg: NestRegistration, *, spec_zone=1):
    """Feedback 1-pt pick for INTEGER fields, in place.

    ``copy_fcni`` (interp_fcn.F:1829-1906), the int32 twin of copy_fcnm.
    No integer field is present in WOOF's live feedback inventory.
    """
    c3, n3, args, count = _copy_common(cfld, nfld, reg, spec_zone)
    if c3.dtype != np.int32 or n3.dtype != np.int32:
        raise TypeError("copy_fcni operates on INTEGER (int32) fields")
    if count > 0:
        _launch(_kernel("nest_copy_fcni"), count, (c3, n3, *args))
    return cfld


#: ``smdsm``'s smoother/de-smoother pair, ``xnu = (/ 0.50 , -0.52 /)``
#: (interp_fcn.F:3976).  The odd pass smooths, the even pass de-smooths.
SMDSM_XNU = (np.float32(0.50), np.float32(-0.52))


def smoother_parent_window(reg: NestRegistration
                           ) -> tuple[int, int, int, int]:
    """The 0-based parent window WRF's feedback smoothers filter.

    ``[pos+2, pos+span-2-stag]`` per axis in 1-based domain cells
    (interp_fcn.F:3918-3921 / :3993-3994), ``span = (n?de-n?ds)/nr?`` --
    the nest's mass extent in parent cells, the same number for every
    stagger -- and ``stag`` 0 on the field's staggered axis, 1 otherwise.
    Returned as ``(i0, j0, niw, njw)``: 0-based low corner plus counts,
    the launch shape rather than the Fortran bounds.  A count can be
    ``<= 0`` on a tiny nest, which is WRF's zero-trip DO loop: nothing
    is smoothed.
    """
    span_i = (reg.nxc - (1 if reg.xstag else 0)) // reg.nri
    span_j = (reg.nyc - (1 if reg.ystag else 0)) // reg.nrj
    istag = 0 if reg.xstag else 1
    jstag = 0 if reg.ystag else 1
    return (reg.i_parent_start + 1, reg.j_parent_start + 1,
            span_i - 3 - istag, span_j - 3 - jstag)


def smoother(cfld, reg: NestRegistration, *, smooth_option: int, scratch):
    """WRF's post-feedback parent smoother, in place (interp_fcn.F:3794).

    Dispatch exactly as ``SUBROUTINE smoother``: 0 is a no-op, 1 is
    ``sm121`` (:3864-3935, one 1-2-1 pass), 2 is ``smdsm`` (:3937-4014,
    a +0.50 smoothing pass then a -0.52 de-smoothing pass).  The caller
    owns the ``feedback == 0 -> RETURN`` guard (:3823) because WOOF's
    coupler, not a namelist read, knows the feedback setting.

    Each pass is two launches per the dataflow both Fortran routines
    share: a J-direction filter into ``scratch`` (whose one-cell ring
    around the window keeps the unfiltered field, standing in for
    sm121's ``cfldnew`` init copy and for smdsm's never-written cells),
    then an I-direction filter from ``scratch`` back into the field.

    ``scratch`` is any float32 device buffer with capacity for
    ``cfld``'s elements; the audited ``nest_parent_field`` arena slot is
    the intended one -- the restriction is done with it by the time the
    smoother runs, so the smoother introduces no allocation.
    """
    smooth_option = int(smooth_option)
    if smooth_option == 0:
        return cfld
    if smooth_option not in (1, 2):
        raise ValueError(
            f"smooth_option must be 0, 1 or 2, got {smooth_option}")
    c3 = _as3d(cfld)
    nz, nyp, nxp = c3.shape
    if (nyp, nxp) != (reg.nyp, reg.nxp):
        raise ValueError("field does not match the registration extents")
    i0, j0, niw, njw = smoother_parent_window(reg)
    if niw <= 0 or njw <= 0:
        return cfld
    count = math.prod(c3.shape)
    s3 = scratch.reshape(-1)[:count].reshape(c3.shape)
    if s3.dtype != np.float32:
        raise ValueError("smoother scratch must be float32")
    mode = np.int32(0 if smooth_option == 1 else 1)
    passes = (SMDSM_XNU if smooth_option == 2
              else (np.float32(0.0),))
    dims = (np.int32(i0), np.int32(j0), np.int32(niw), np.int32(njw),
            np.int32(nz), np.int32(nyp), np.int32(nxp))
    for xnu in passes:
        _launch(_kernel("nest_smooth_j"),
                nz * (njw + 2) * (niw + 2), (c3, s3, mode, xnu, *dims))
        _launch(_kernel("nest_smooth_i"),
                nz * njw * niw, (c3, s3, mode, xnu, *dims))
    return cfld


# ---------------------------------------------------------------------------
# The MASKED parent->child interpolator (surface/soil fields)
# ---------------------------------------------------------------------------

#: Branch labels of :func:`interp_mask_field`, in WRF's own decision order.
#: Every child column lands in exactly one, and the caller receipts the
#: tally -- ``opposite_class_bilinear`` is the land/water CONFLICT count
#: (WRF's one-cell island/lake compromise), never a silent path.
MASK_INTERP_BRANCHES = ("same_class_bilinear", "class_matched_average",
                        "opposite_class_bilinear")


def mask_donor_index(n_child: int, ratio: int, pos: int):
    """Lower-left bilinear donor + fractional offset, 0-based.

    ``interp_fcn.F:4140-4155`` (and the identical arithmetic at :3230-3245
    in ``interp_mask_land_field``).  For WRF 1-based child index ``n`` and
    1-based ``ipos``/``jpos``::

        odd  ratio: ci = ( n + (nri-1)/2 ) / nri + ipos - 1
        even ratio: ci = ( n + (nri/2)-1 ) / nri + ipos - 1
        odd  ratio: dx =   MOD( n + (nri-1)/2 , nri )         / nri
        even ratio: dx = ( MOD( n + (nri-1)/2 , nri ) + 0.5 ) / nri

    This is NOT the SINT donor map (:func:`_donor_maps`): SINT picks the
    coarse cell CONTAINING the child point for its +-2 flux stencil, while
    the masked interpolator picks the coarse cell at or below/left of it,
    the SW corner of a 4-point bilinear cell.  The two differ by design and
    both are transliterated rather than shared.

    Returns ``(donor0, frac)``, both ``(n_child,)``.
    """
    ratio = int(ratio)
    if ratio < 1:
        raise ValueError("parent_grid_ratio must be >= 1")
    n = np.arange(1, int(n_child) + 1, dtype=np.int64)
    half = (ratio - 1) // 2               # Fortran integer divide
    if ratio % 2 == 0:
        donor = (n + ratio // 2 - 1) // ratio + int(pos) - 1
        frac = (np.mod(n + half, ratio) + 0.5) / ratio
    else:
        donor = (n + half) // ratio + int(pos) - 1
        frac = np.mod(n + half, ratio) / ratio
    return (donor - 1).astype(np.int64), frac.astype(np.float64)


def interp_mask_field(cfld, *, nri, nrj, i_parent_start, j_parent_start,
                      child_landuse, parent_landuse, flag_category,
                      child_shape=None):
    """WRF's masked surface/soil nest interpolator, host side.

    ``interp_mask_field`` (interp_fcn.F:4075-4275) -- the interpolator the
    Registry names for every land-surface state field that crosses a nest
    boundary::

        state real TSLB  ilj ... i02rhd=(interp_mask_field:lu_index,iswater)
        state real SMOIS ilj ... i02rhd=(interp_mask_field:lu_index,iswater)
        state real XICE  ij  ... i0124rhd=(interp_mask_field:lu_index,isice)

    (Registry/Registry.EM_COMMON:790, :839-842, :868-872, :1417.)  The mask
    is the LAND-USE CATEGORY compared against one flag category, not the
    binary landmask: ``flag_category`` is ``ISWATER`` for the soil/snow/skin
    family and ``ISICE`` for the sea-ice family, exactly as the Registry
    spells it per field.

    WRF's decision, per child column (:4224-4266):

    1. all four coarse corners in the child's own class, or all four in the
       other class -> plain 4-point bilinear.  The second half of that is
       the one-cell island/lake case, where WRF's own comment says it has
       "no better way to come up with the value";
    2. corners mixed -> average of ONLY the corners matching the child's
       class.

    Returns ``(nfld, counts)``.  ``counts`` splits WRF's first branch back
    into its two meanings (:data:`MASK_INTERP_BRANCHES`) so the land/water
    conflicts are counted rather than folded into the clean path -- the
    ``donor_fill_plan`` receipt discipline, applied to the interpolator.

    ``cfld`` is ``(..., ny_parent, nx_parent)``; leading dimensions (soil
    layers) ride through untouched, and the mask decision is per column,
    identical on every layer, as WRF's ``nk`` loop makes it.
    """
    cfld = np.asarray(cfld)
    parent_lu = np.asarray(parent_landuse)
    child_lu = np.asarray(child_landuse)
    if parent_lu.ndim != 2 or child_lu.ndim != 2:
        raise ValueError("interp_mask_field masks are 2-D (ny, nx) "
                         f"land-use fields; got {parent_lu.shape} and "
                         f"{child_lu.shape}")
    if cfld.shape[-2:] != parent_lu.shape:
        raise ValueError(
            f"coarse field trailing shape {cfld.shape[-2:]} does not match "
            f"the parent land-use mask {parent_lu.shape}")
    ny_child, nx_child = (child_lu.shape if child_shape is None
                          else tuple(int(n) for n in child_shape))
    if (ny_child, nx_child) != child_lu.shape:
        raise ValueError(
            f"child shape {(ny_child, nx_child)} disagrees with the child "
            f"land-use mask {child_lu.shape}")

    ci, dx = mask_donor_index(nx_child, nri, i_parent_start)
    cj, dy = mask_donor_index(ny_child, nrj, j_parent_start)
    ny_par, nx_par = parent_lu.shape
    if ci.min() < 0 or cj.min() < 0 or ci.max() + 1 > nx_par - 1 \
            or cj.max() + 1 > ny_par - 1:
        raise ValueError(
            "the masked interpolator's 4-point cell reaches parent indices "
            f"i {ci.min()}..{ci.max() + 1}, j {cj.min()}..{cj.max() + 1}, "
            f"outside the parent extent ({ny_par}, {nx_par}); the placement "
            "is off-grid")

    # WRF evaluates dx/dy and the whole 4-point form in REAL (FP32).  The
    # offsets are exact either way (small integers over a small integer,
    # one correctly-rounded divide), but the products are NOT, so a
    # float32 field is interpolated in float32, in WRF's operand order.
    work = (cfld.dtype if cfld.dtype.kind == "f" and cfld.dtype.itemsize >= 4
            else np.dtype(np.float64))
    one = work.type(1.0)
    jj = cj[:, None]
    ii = ci[None, :]
    wx = dx.astype(work)[None, :]
    wy = dy.astype(work)[:, None]

    def corners(field):
        return (field[..., jj, ii], field[..., jj, ii + 1],
                field[..., jj + 1, ii], field[..., jj + 1, ii + 1])

    f00, f10, f01, f11 = (part.astype(work, copy=False)
                          for part in corners(cfld))
    bilinear = ((one - wx) * ((one - wy) * f00 + wy * f01)
                + wx * ((one - wy) * f10 + wy * f11))

    flag = int(flag_category)
    m00, m10, m01, m11 = (np.rint(part).astype(np.int64) == flag
                          for part in corners(parent_lu))
    child_is_flag = np.rint(child_lu).astype(np.int64) == flag

    in_flag = m00.astype(np.int64) + m10 + m01 + m11
    all_flag = in_flag == 4
    all_other = in_flag == 0
    uniform = all_flag | all_other
    # "All four corners are the child's own class" collapses to comparing
    # the child's class against the (single) coarse class of a uniform cell.
    same_class = uniform & (child_is_flag == all_flag)
    opposite_class = uniform & ~same_class
    mixed = ~uniform

    # Branch 2: mean over the corners whose class matches the child's.  A
    # mixed cell always has at least one of each class, so the divisor is
    # never zero -- the WRF loop relies on the same fact.
    keep = [np.where(child_is_flag, part, ~part)
            for part in (m00, m10, m01, m11)]
    tally = (keep[0].astype(np.int64) + keep[1] + keep[2] + keep[3]
             ).astype(work)
    zero = work.type(0.0)
    total = np.zeros(bilinear.shape, dtype=work)
    for part, corner in zip(keep, (f00, f10, f01, f11)):
        total = total + np.where(part, corner, zero)
    averaged = total / np.where(tally > zero, tally, work.type(1.0))

    nfld = np.where(mixed, averaged, bilinear).astype(cfld.dtype, copy=False)
    counts = {
        "cells": int(child_lu.size),
        "child_flag_class_cells": int(np.count_nonzero(child_is_flag)),
        "same_class_bilinear": int(np.count_nonzero(same_class)),
        "class_matched_average": int(np.count_nonzero(mixed)),
        "opposite_class_bilinear": int(np.count_nonzero(opposite_class)),
    }
    return nfld, counts
