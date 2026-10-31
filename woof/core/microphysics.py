"""Microphysics: scheme registry and the Kessler warm-rain launcher.

:func:`apply` is the single public surface, called by ``dycore.step`` after
the RK3 loop (WRF solve_em: the non-timesplit microphysics runs once per
model step on the updated fields) and dispatching on ``cfg.mp_physics``:
0 is a no-op (the state is untouched -- the whole dry/moist-without-mp
suite is bitwise unaffected), 1 is Kessler
(kernels/kessler.cu, transcribed from the bundle's WRF v4.6.1
``phys/module_mp_kessler.F``; float64 mirror
``woof.verify.npref.np_kessler_column``), 6 is WSM6, 8 is the classic
Thompson CUDA path (packaged byte-validated WRF v4.6.1 tables, env-var
table-root override honored), 10 is Morrison two-moment
(``woof.core.morrison`` / ``kernels/morrison.cu``), 18 is NSSL
two-moment through its persistent production binding, and 28 is
aerosol-aware Thompson (``woof.core.microphysics_aerosol`` and the eight
``kernels/thompson_aerosol_*.cu`` translation units; WRF
Registry/Registry.EM_COMMON:3036).  mp=28 is a SIBLING of the mp=8 path,
never a branch inside it: ``kernels/thompson.cu`` and
``woof.core.thompson`` are byte-frozen and gain nothing from this port,
which is what makes the mp=8 non-regression a statement about bytes.

The scheme inputs follow WRF ``moist_physics_prep_em``
(dyn_em/module_big_step_utilities_em.F) with ``use_theta_m = 0`` (woof's
dry-theta prognostic): full potential temperature ``thb + thp``, DRY
density ``rho = 1/alt``, full-pressure Exner ``pii = (p/p0)^(Rd/cp)``, and
half-level heights / layer depths from the full geopotential -- so
``apply`` must run with up-to-date EOS diagnostics (``dycore.step`` calls
it right after its epilogue ``update_diagnostics``).  The scheme updates
theta (latent heating) and the active water mass/number moments in place;
surface precipitation accumulates in persistent ``mp_*`` scratch slots
(mm, matching WRF's RAINNC/SNOWNC/GRAUPELNC and per-call ``*NCV`` fields).

h_diabatic (audit finding R7): each scheme adapter brackets its column
kernel with WRF's prep/finish pair.  :func:`save_pre_mp_theta` parks the
pre-microphysics full theta in ``state.h_diabatic`` (moist_physics_prep_em,
module_big_step_utilities_em.F:5503/:5523-5526) and
:func:`moist_physics_finish` applies the clamped theta increment ONCE to
the prognostic theta and stores the heating rate ``mpten/dt`` back in
``state.h_diabatic`` (moist_physics_finish_em, :5682-5746) for the NEXT
step's RK tendencies (dycore.add_h_diabatic_tendency).  Float64 mirror:
``woof.verify.npref.np_moist_physics_finish``.

Specified-zone ring exclusion (Wave-1 rank-1 seam fix): on every
specified or nested WRF domain the microphysics tiles are clipped by
``sz = spec_zone`` (solve_em.F:3618-3622; ``specified .or. nested`` per
:3692) -- moist_physics_prep (:3631-3639), the scheme driver
(module_microphysics_driver.F:802-806/:870-879), and
moist_physics_finish including the h_diabatic capture (:4040-4048) all
run ``its = max(i_start, ids+sz) .. ite = min(i_end, ide-1-sz)``,
``jts = max(j_start, jds+sz) .. jte = min(j_end, jde-1-sz)``, so the
outermost ``sz`` mass ring is NEVER touched: ring RAINNC is exactly 0.0
at every lead, ring theta/moisture keep their boundary-installed values,
and ring h_diabatic stays 0 (allocator zero init,
frame/module_domain.F:770-777 / tools/gen_allocs.c:411-415;
set_physical_bc3d writes halo indices only, module_bc.F:867-885).
Scope qualifications: WRF's optional ``mp_zero_out`` path floors the
outermost ring at zero over WHOLE-field mass bounds before the clipped
finish (solve_em.F:4002-4038); :func:`microphysics_zero_out` runs it
after the ring restore when ``mp_zero_out > 0``, so the never-touched
statement holds for ring values only while ``mp_zero_out = 0`` (the
default); and a specified+periodic_x channel clips j only
(module_microphysics_driver.F:871-873) -- woof has no channel mode and
always excludes the four-sided ring.  An earlier
in-tree justification argued the whole-field call was inert because the
spec-zone theta TENDENCY is replaced and spec-zone theta values are
overwritten each stage; that argument was overbroad -- the schemes'
direct in-place ring theta/moisture updates and the finish's direct
``thp += mpten`` land AFTER dycore.step's end-of-step
``apply_state_boundary_values``, and ring RAINNC accumulated where WRF's
is exactly zero, leaking into interior rows through the next step's
stencils (measured: O(0.1 mm/h) ring RAINNC and a sustained rows-1..4
theta/qv envelope).  :func:`apply` therefore excludes the ring for
specified/nested configs: every scheme here is a per-column kernel with
no horizontal coupling inside the call, so capturing the ring before
dispatch and bit-restoring it afterwards is bitwise identical -- ring
AND interior -- to WRF's clipped tiles (see
:func:`spec_zone_ring_slices`).  Periodic/open configs keep the exact
whole-field behavior (WRF: sz = 0), so every frozen idealized case is
bitwise unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import weakref

import cupy as cp
import numpy as np

from woof.microphysics_schemes import scheme as _named_scheme
from woof.config import RunConfig
from woof.core import constants as c
from woof.core.kernels import get_kernel
from woof.core.state import DTYPE, DomainState
from woof.core.wdm6_constants import WDM6_NUMBER_SPECIES
from woof.physics_compat import thompson_table_root as _thompson_table_root

_TPB = 64          # threads per block (one thread per column)
_VALIDATION_TPB = 256
_KMAX = 256        # matches KESS_KMAX in kernels/kessler.cu
KESSLER_VERTICAL_LEVEL_BOUNDS = (None, _KMAX)

# ---------------------------------------------------------------------------
# WRF specified-zone ring exclusion (solve_em.F:3618-3639)
# ---------------------------------------------------------------------------

#: State attributes a scheme adapter (or the prep/finish bracket) updates in
#: place.  Attributes the configured scheme does not allocate are skipped.
#: h_diabatic is deliberately absent: its ring value is pinned to WRF's
#: exact 0 in :func:`_restore_spec_zone_ring` instead of being restored.
#: ``nwfa``/``nifa`` join the list for mp=28: the aerosol adapter's terminal
#: apply (module_mp_thompson.F:3972-4021) and its surface emission
#: (mp_gt_driver:1310-1327) both write them in place, so an unrestored ring
#: would accumulate aerosol WRF's clipped tiles never touch.  Presence
#: guards mean every other scheme is unaffected -- no state but mp=28's
#: allocates them (woof/core/moist.py, ``nwfa`` is the mp=28 discriminator).
#: ``nwfa2d``/``nifa2d`` are deliberately absent: they are INTENT(IN) in WRF
#: and the microphysics never writes them.
#: ``qh``/``nh`` join for mp=9 (Milbrandt-Yau): its hail mass and hail
#: number are prognostic and the adapter writes them in place, so an
#: unrestored ring would carry hail WRF's clipped tiles never touch.
#: Presence guards keep every other scheme unaffected -- mp=18 allocates
#: ``qh`` too and has always been captured through it, and no scheme but
#: mp=9 allocates ``nh``.
#: ``qir``/``qib`` (P3's rime mass and rime volume) and ``th_old``/``qv_old``
#: join for mp=50 under the same presence guard: p3_main updates all four in
#: place, and ``th_old``/``qv_old`` in particular are written whole-array at
#: the end of every call (module_mp_p3.F:5018-5021), so an unrestored ring
#: would feed the NEXT step's supersaturation tendency (:3171) from columns
#: WRF's clipped tiles never advanced.  No other scheme allocates them.
#:
#: DERIVED: the union, over every implemented scheme, of the registry's
#: ``consumers.ring_guard.state_fields`` row -- the same row
#: woof.core.physics_inventory prices from -- so the captured family and
#: the priced family are one family.  Presence guards at the capture site
#: keep every other scheme unaffected by another scheme's names; the union
#: is what lets NSSL's registry-native moment names (qndrop, qnr, ..., qvolh)
#: be captured at all, which the typed tuple this replaces never listed.
def _ring_state_fields() -> tuple[str, ...]:
    # The derivation lives beside the per-scheme row it unions, in the
    # device-free module the preflight prices from, so a card-free install
    # can read the captured family too.
    from woof.core.physics_inventory import ring_guard_state_fields

    return ring_guard_state_fields()


_RING_STATE_FIELDS = _ring_state_fields()

#: Persistent (ny, nx) surface slots the adapters write: the accumulators
#: RAINNC/SNOWNC/GRAUPELNC plus the per-call *NCV/SR diagnostics (the
#: canonical driver mapping is physics.microphysics_scratch_slots; this is
#: its union across schemes).  WRF never writes any of them in the ring:
#: the accumulators/NCVs by the clipped tiles (zero init), SR by
#: ``grid%sr = 0.`` whole-field each call (solve_em.F:3691) followed by
#: clipped tile writes.
_RING_SURFACE_SLOTS = (
    "mp_rainnc", "mp_rainncv", "mp_snownc", "mp_snowncv",
    "mp_graupelnc", "mp_graupelncv", "mp_hailnc", "mp_hailncv",
    "mp_sr", "mp_kessler_sr",
)

#: Persistent 3-D scratch slots that survive the call (the REFL_10CM
#: output stash -- WRF's clipped tiles leave ring refl at its allocated 0).
_RING_VOLUME_SLOTS = ("refl_10cm",)


from woof.core.physics_inventory import (  # noqa: F401,E402
    spec_zone_ring_save_slots,  # one home; re-exported here
    spec_zone_ring_slices,  # one home (cupy-free); re-exported here
)


def _ring_guard_slices(state: DomainState, cfg: RunConfig):
    """WRF solve_em.F:3618-3622: ``sz = spec_zone`` iff ``specified .or.
    nested``, else 0 -- so periodic/open configs return None and keep the
    frozen whole-field call bitwise.  woof has no specified+periodic_x
    channel mode, so WRF's channel i-clip exemption
    (module_microphysics_driver.F:871-873) is deliberately not wired.
    """
    if not (getattr(cfg, "specified", False)
            or getattr(cfg, "nested", False)):
        return None
    sz = int(cfg.spec_zone)
    if sz <= 0:
        return None
    _, ny, nx = state.p.shape
    return spec_zone_ring_slices(ny, nx, sz)


#: One launch moves every ring section (kernels/microphysics_validation.cu,
#: ``mp_ring_copy``).
_RING_TPB = 256
_RING_MAX_X_BLOCKS = 64
#: Device descriptor tables by the state whose arrays they address, then by
#: their exact contents.  A table is a pure function of its key (addresses
#: and extents), so a hit is always the right table; persistent state and
#: scratch keep their addresses from call to call, so a run uploads each of
#: its two tables once.  A table lives exactly as long as its state: a CUDA
#: graph that captured a ring launch (the tiled runner's --graph path)
#: replays with the table's address baked in, and with the state's own
#: array addresses baked in too, so no graph that can still replay outlives
#: the tables it holds.  Tables used to be kept for the whole process, and
#: every forecast run in one process (each member of an ensemble) left its
#: tables behind for the next (tests/test_ensemble_member_release_gpu.py).
#: A table is 72 bytes per ring section, a few kilobytes per domain.
_RING_TABLES: "weakref.WeakKeyDictionary[DomainState, dict[tuple, cp.ndarray]]" = \
    weakref.WeakKeyDictionary()


def _ring_row(arr, slc, buf):
    """The ``mp_ring_copy`` descriptor of one ring section, or ``None``
    where the section keeps the plain slice copy (not float32, not
    C-contiguous, or not a 2-D/3-D mass-grid array).  ``buf=None`` asks for
    a zero fill."""
    if (arr.dtype != DTYPE or not arr.flags.c_contiguous
            or arr.ndim not in (2, 3) or slc[0] is not Ellipsis):
        return None
    ny, nx = arr.shape[-2:]
    j0, j1, js = slc[1].indices(ny)
    i0, i1, is_ = slc[2].indices(nx)
    nj, ni = j1 - j0, i1 - i0
    if js != 1 or is_ != 1 or nj <= 0 or ni <= 0:
        return None
    nlev = arr.size // (ny * nx)
    if buf is not None and (buf.dtype != DTYPE or not buf.flags.c_contiguous
                            or buf.size != nlev * nj * ni):
        return None
    return (int(arr.data.ptr), 0 if buf is None else int(buf.data.ptr),
            nlev, j0, nj, i0, ni, nx, ny * nx)


def _launch_ring_rows(rows, *, direction: int, owner) -> bool:
    """Gather (0) or scatter/zero (1) every described section in one launch.

    ``owner`` is the state whose arrays the rows address; its table is kept
    while that state lives (see ``_RING_TABLES``).

    Returns False, launching nothing, when the table is not resident yet and
    the current stream is capturing a CUDA graph: uploading it would be a
    host transfer inside the capture, which fails it.  The caller then does
    those sections with slice copies, exactly as before this kernel existed.
    """
    if not rows:
        return True
    key = tuple(rows)
    tables = _RING_TABLES.get(owner)
    table = None if tables is None else tables.get(key)
    if table is None:
        if cp.cuda.get_current_stream().is_capturing():
            return False
        table = cp.asarray(np.asarray(rows, dtype=np.int64).reshape(-1))
        _RING_TABLES.setdefault(owner, {})[key] = table
    count = max(r[2] * r[4] * r[6] for r in rows)
    blocks_x = min((count + _RING_TPB - 1) // _RING_TPB, _RING_MAX_X_BLOCKS)
    get_kernel("microphysics_validation", "mp_ring_copy")(
        (blocks_x, len(rows)), (_RING_TPB,),
        (table, np.int32(len(rows)), np.int32(direction)))
    return True


def _capture_spec_zone_ring(state: DomainState, slices):
    """Snapshot every ring section the microphysics call could touch.

    Every scheme here is a per-column kernel (and prep/finish are
    elementwise), so the call has no horizontal coupling: restoring the
    ring afterwards is bitwise identical -- ring AND interior -- to never
    running the scheme there, which is woof's equivalent of WRF's
    clipped tiles for whole-field kernel launches
    (tests/test_mp_spec_zone_ring.py pins both halves).
    """
    saved = []
    captured_slots = set()
    fused = []
    deferred = []

    def snap(arr, key):
        for index, slc in enumerate(slices):
            part = arr[slc]
            if part.size == 0:
                continue
            buf = state.scratch(part.shape, f"mp_ring_save_{key}_{index}")
            row = _ring_row(arr, slc, buf)
            if row is None:
                buf[...] = part
            else:
                fused.append(row)
                deferred.append((buf, part))
            saved.append((arr, slc, buf))

    for name in _RING_STATE_FIELDS:
        arr = getattr(state, name, None)
        if arr is not None:
            snap(arr, name)
    for slot in _RING_SURFACE_SLOTS + _RING_VOLUME_SLOTS:
        arr = state.existing_scratch(slot)
        if arr is not None:
            snap(arr, slot)
            captured_slots.add(slot)
    if not _launch_ring_rows(fused, direction=0, owner=state):
        for buf, part in deferred:
            buf[...] = part
    return saved, captured_slots


def _restore_spec_zone_ring(state: DomainState, slices, saved,
                            captured_slots) -> None:
    """Bit-restore the captured ring, zero the ring of any persistent slot
    the call just created (its pre-call value WAS the fresh zero fill),
    and pin ring h_diabatic to WRF's exact value.

    h_diabatic is pinned to 0 rather than restored: WRF's ring h_diabatic
    is identically zero in every specified/nested run (allocator zero
    init, frame/module_domain.F:770-777 / tools/gen_allocs.c:411-415;
    the clipped finish tiles never write it; set_physical_bc3d writes
    halo indices only, module_bc.F:867-885), and the pinned zero -- not a
    restored stale rate -- is what the next step's rk_addtend_dry slot
    must see.
    """
    # Every write below targets a ring section; the three families (the
    # captured sections, the ring of a slot created during the call, and
    # h_diabatic) are disjoint arrays, so one launch doing all of them
    # writes exactly what the sequence of slice assignments wrote.
    fused = []
    deferred = []

    def plain(arr, slc, buf):
        if buf is None:
            arr[slc] = 0
        else:
            arr[slc] = buf

    def put(arr, slc, buf):
        row = _ring_row(arr, slc, buf)
        if row is not None:
            fused.append(row)
            deferred.append((arr, slc, buf))
        else:
            plain(arr, slc, buf)

    for arr, slc, buf in saved:
        put(arr, slc, buf)
    for slot in _RING_SURFACE_SLOTS + _RING_VOLUME_SLOTS:
        if slot in captured_slots:
            continue
        arr = state.existing_scratch(slot)
        if arr is None:
            continue
        for slc in slices:
            put(arr, slc, None)
    if state.h_diabatic is not None:
        for slc in slices:
            put(state.h_diabatic, slc, None)
    if not _launch_ring_rows(fused, direction=1, owner=state):
        for arr, slc, buf in deferred:
            plain(arr, slc, buf)


@dataclass(frozen=True)
class MicrophysicsDiagnostics:
    """Named post-RK microphysics-to-physics-driver contract.

    Accumulators/increments are millimetres (kg m-2), and ``sr`` is the
    scheme-native frozen fraction used by Noah when ``frpcpn`` is active.  This
    replaces knowledge of private ``DomainState._scratch`` names in the
    pre-RK surface driver.
    """

    rainnc: cp.ndarray
    rainncv: cp.ndarray
    sr: cp.ndarray
    snownc: cp.ndarray | None = None
    snowncv: cp.ndarray | None = None
    graupelnc: cp.ndarray | None = None
    graupelncv: cp.ndarray | None = None
    hailnc: cp.ndarray | None = None
    hailncv: cp.ndarray | None = None


def validate_surface_diagnostics(
        values: tuple[cp.ndarray, ...], active: int, sr_upper: np.float32,
        status: cp.ndarray, *, describe=None) -> int:
    """Return compact finite/range flags for canonical surface outputs.

    ``describe`` opts this site into :mod:`woof.core.health_ledger`: with a
    ledger active the flags are accumulated on the device and ``0`` is
    returned, and ``describe(flags)`` raises at the drain instead.  Without
    it -- the default, and every historical caller -- the word is read here
    and nothing about the site changes.
    """
    from woof.core import health_ledger

    status.fill(cp.uint32(0))
    sr = values[2]
    count = sr.size
    blocks = (count + _VALIDATION_TPB - 1) // _VALIDATION_TPB
    kernel = get_kernel(
        "microphysics_validation", "microphysics_validate_outputs")
    kernel(
        (blocks,), (_VALIDATION_TPB,),
        values + (np.uint32(active), DTYPE(sr_upper), status,
                  np.int64(count)))
    return health_ledger.read_status(
        status, site="microphysics", describe=describe)


def prepare_microphysics_exner(pressure, out):
    """Rounded moist_physics_prep_em Exner, shared with the physics caller.

    The array power moved the step's PII by one ULP on identical pressure
    words. That error enters latent heating even when the column scheme
    itself agrees exactly with its Fortran oracle.
    """
    from woof.core.physics import _physics_exner
    return _physics_exner(pressure, out=out)


#: WRF v4.6.1 ``mp_gt_driver``'s negative-vapour fill
#: (phys/module_mp_thompson.F:1436-1445), run once per column after the
#: scheme, in the driver's own k loop (:1350), which writes qv(k) = qv1d(k)
#: and then, where qv1d(k) < 0,
#:     qv(k) = MAX(1.E-7, 0.5*(qv(k-1) + qv(k+1)))   for kts+1 < k < kte-2
#:     qv(k) = 1.E-7                                 otherwise.
#: qv(k-1) is the level the same loop just wrote (filled if it was
#: negative); qv(k+1) is not written yet, so it is the driver's INPUT vapour.
#: mp_thompson floors every level of a column it works on at 1.E-10 (:3974),
#: so a negative qv1d is only ever found in a column it left at the
#: no-microphysics exit (:2020), where qv1d is the input vapour unchanged:
#: the post-scheme array therefore holds exactly the words the WRF loop reads
#: and the fill needs no copy of the input.  A NaN is not < 0 and -0.0 is not
#: < 0, so neither is filled.  The breakage this prevents: without it, a
#: dry level the advection took a few 1e-13 below zero stayed negative in
#: ArWen while wrf.exe set it to ~1e-3 (the mean of its neighbours), and the
#: strict UAE 9 km pair parted at step 2 (9 words, then every field by 1 h).
_THOMPSON_VAPOUR_FILL_SOURCE = r"""
extern "C" __global__ void thompson_negative_vapour_fill(
    float* __restrict__ qv, const int nz, const long long ncol)
{
    const long long col = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncol) return;
    for (int k = 0; k < nz; ++k) {
        const unsigned int u = __float_as_uint(qv[(long long)k * ncol + col]);
        // qv1d(k) .lt. 0.0: negative, not -0.0, not NaN (a bit test, so a
        // flushing compiler cannot hide a subnormal negative).
        if (u > 0x80000000u && u <= 0xff800000u) {
            // 1-based k+1: kts+1 < k+1 < kte-2  <=>  1 < k < nz-3.
            if (k > 1 && k < nz - 3) {
                const float mean = __fmul_rn(0.5f, __fadd_rn(
                    qv[(long long)(k - 1) * ncol + col],
                    qv[(long long)(k + 1) * ncol + col]));
                qv[(long long)k * ncol + col] = fmaxf(1.0e-7f, mean);
            } else {
                qv[(long long)k * ncol + col] = 1.0e-7f;
            }
        }
    }
}
"""
_thompson_vapour_fill_kernel = None


def thompson_negative_vapour_fill(qv) -> None:
    """WRF ``mp_gt_driver``'s negative-vapour fill on ``qv`` ``(nz, ny, nx)``,
    in place (see ``_THOMPSON_VAPOUR_FILL_SOURCE``).  Call it after the
    scheme's reflectivity and effective radii, which WRF computes from the
    unfilled qv1d (:1456, :1466), and before moist_physics_finish, which
    reads the filled vapour (use_theta_m = 1)."""
    global _thompson_vapour_fill_kernel
    import cupy
    if not isinstance(qv, cupy.ndarray):
        _thompson_negative_vapour_fill_host(qv)
        return
    if _thompson_vapour_fill_kernel is None:
        # No compiler options: every operation is an explicit round-to-
        # nearest intrinsic (nothing to contract), the negative test reads
        # the bits, and the result is MAX(1.E-7, mean), so whether the
        # compiler flushes subnormals cannot change a word: a subnormal mean
        # loses to 1.E-7 either way, and beside a neighbour large enough to
        # lift the mean above 1.E-7 a subnormal addend is far below its ULP.
        from woof.certify.kernel_manifest import record_module
        from woof.core.kernels import _compile_observed
        module = cp.RawModule(code=_THOMPSON_VAPOUR_FILL_SOURCE, options=())
        key = "woof.core.microphysics:thompson_negative_vapour_fill"
        _compile_observed(module, key)
        record_module(key, source=_THOMPSON_VAPOUR_FILL_SOURCE, options=(), module=module)
        _thompson_vapour_fill_kernel = module.get_function("thompson_negative_vapour_fill")
    if qv.dtype != np.float32 or not qv.flags.c_contiguous or qv.ndim != 3:
        raise ValueError("the vapour fill takes a C-contiguous float32 "
                         "(nz, ny, nx) qv")
    nz = int(qv.shape[0])
    ncol = int(qv.shape[1]) * int(qv.shape[2])
    threads = 128
    _thompson_vapour_fill_kernel(
        ((ncol + threads - 1) // threads,), (threads,),
        (qv, np.int32(nz), np.int64(ncol)))


def _thompson_negative_vapour_fill_host(qv) -> None:
    """The same loop on a host array (the launcher-recording gates run the
    adapters on NumPy states): level by level, every column at once."""
    if qv.dtype != np.float32 or qv.ndim != 3:
        raise ValueError("the vapour fill takes a float32 (nz, ny, nx) qv")
    nz = qv.shape[0]
    q = qv.reshape(nz, -1)
    for k in range(nz):
        bits = q[k].view(np.uint32)
        negative = (bits > np.uint32(0x80000000)) & (bits <= np.uint32(0xFF800000))
        if not negative.any():
            continue
        if 1 < k < nz - 3:
            mean = np.float32(0.5) * (q[k - 1][negative] + q[k + 1][negative])
            q[k][negative] = np.maximum(np.float32(1.0e-7), mean)
        else:
            q[k][negative] = np.float32(1.0e-7)


def save_pre_mp_theta(state: DomainState) -> None:
    """WRF ``moist_physics_prep_em``: park the pre-microphysics FULL theta
    in the h_diabatic array (module_big_step_utilities_em.F:5503 "use
    h_diabatic to temporarily save pre-microphysics full theta",
    :5523-5526; ``use_theta_m = 0`` conversion ``th_phy = t_new + t0`` at
    :5513-5521).  :func:`moist_physics_finish` consumes and overwrites it.
    """
    thb = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
    cp.add(thb, state.thp, out=state.h_diabatic)


def moist_physics_finish(state: DomainState, cfg: RunConfig, th_phy,
                         dt: float) -> None:
    """WRF ``moist_physics_finish_em`` (module_big_step_utilities_em.F:
    5593-5784), ``use_theta_m = 0`` branch; float64 mirror
    ``woof.verify.npref.np_moist_physics_finish``.

    ``th_phy`` is the scheme's post-microphysics full theta;
    ``state.h_diabatic`` holds the :func:`save_pre_mp_theta` full theta.
    With ``cfg.no_mp_heating == 0`` (:5682; Registry default) the theta
    increment ``mpten = th_phy - saved`` (:5688) is clamped to
    ``+/- DTYPE(cfg.mp_tend_lim) * DTYPE(dt)``, the REAL product WRF forms
    (:5706-5707), added ONCE to the prognostic
    perturbation theta (:5743), and retained as the heating rate
    ``h_diabatic = mpten/dt`` (:5745) that the NEXT step's RK loop feeds
    to the theta tendency.  With ``no_mp_heating = 1`` theta is left
    untouched (:5775) and ``h_diabatic = 0`` (:5776) -- the scheme's
    moisture updates stand either way.
    """
    if cfg.no_mp_heating == 0:
        # WRF multiplies the REAL mp_tend_lim by the REAL dt (:5706-5707),
        # one float32 rounding of two float32 operands, as classic
        # Thompson's fused finish does (thompson.py launch_adapter_finish).
        # Rounding the double product instead moves a clamped increment by
        # one float32 unit for some pairs (HRRR's 0.07 K/s at dt = 12 s:
        # 0x3f570a3d, not WRF's 0x3f570a3e), and with it thp and h_diabatic.
        lim = DTYPE(DTYPE(cfg.mp_tend_lim) * DTYPE(dt))
        mpten = state.h_diabatic
        cp.subtract(th_phy, mpten, out=mpten)        # :5688
        cp.minimum(lim, mpten, out=mpten)            # :5706
        cp.maximum(-lim, mpten, out=mpten)           # :5707
        cp.add(state.thp, mpten, out=state.thp)      # :5743
        cp.divide(mpten, DTYPE(dt), out=mpten)       # :5745
    else:
        state.h_diabatic[...] = 0.0                  # :5775-5776


def launch_kessler(t, qv, qc, qr, rho, pii, z, dz8w, rainnc, rainncv,
                   dt: float) -> None:
    """Run the ``kessler_column`` kernel on device arrays (in place).

    ``t`` (full theta), ``qv``/``qc``/``qr`` (kg/kg) are updated; ``rho``
    (dry density), ``pii`` (Exner), ``z``/``dz8w`` (m) are read-only, all
    ``(nz, ny, nx)`` float32; ``rainnc``/``rainncv`` are ``(ny, nx)``
    accumulated / per-call surface rain in mm.
    """
    nz, ny, nx = t.shape
    if nz > KESSLER_VERTICAL_LEVEL_BOUNDS[1]:
        raise ValueError(f"nz={nz} exceeds the Kessler kernel's per-thread "
                         "column storage KESS_KMAX="
                         f"{KESSLER_VERTICAL_LEVEL_BOUNDS[1]}")
    kern = get_kernel("kessler", "kessler_column")
    blocks = (ny * nx + _TPB - 1) // _TPB
    kern((blocks,), (_TPB,),
         (t, qv, qc, qr, rho, pii, z, dz8w, rainnc, rainncv,
          DTYPE(dt), np.int32(nz), np.int32(ny), np.int32(nx)))


def _apply_kessler(state: DomainState, cfg: RunConfig, dt: float, *,
                    refl_10cm_due: bool = False) -> MicrophysicsDiagnostics:
    """Build the WRF prep fields from the state and run the column kernel."""
    nz, ny, nx = state.p.shape
    thb = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
    phb = state.phb if state.phb.ndim == 3 else state.phb[:, None, None]

    th = state.scratch((nz, ny, nx), "mp_th")
    rho = state.scratch((nz, ny, nx), "mp_rho")
    pii = state.scratch((nz, ny, nx), "mp_pii")
    zh = state.scratch((nz, ny, nx), "mp_z")
    dz8w = state.scratch((nz, ny, nx), "mp_dz8w")
    z8w = state.scratch((nz + 1, ny, nx), "mp_z8w")

    th[...] = thb + state.thp
    rho[...] = 1.0 / state.alt                   # dry density 1/(al+alb)
    prepare_microphysics_exner(state.p, pii)
    z8w[...] = (phb + state.php) / DTYPE(c.G)
    zh[...] = 0.5 * (z8w[:nz] + z8w[1:])
    dz8w[...] = z8w[1:] - z8w[:nz]

    rainnc = state.scratch((ny, nx), "mp_rainnc")
    rainncv = state.scratch((ny, nx), "mp_rainncv")
    save_pre_mp_theta(state)                     # WRF moist_physics_prep_em
    launch_kessler(th, state.qv, state.qc, state.qr, rho, pii, zh, dz8w,
                   rainnc, rainncv, dt)
    if refl_10cm_due:
        # Kessler has no native refl10cm routine; keep woof's documented
        # fallback on the same prepared-p/post-scheme-T timing as Morrison.
        from woof.core.refl import compute_and_stash_refl_10cm
        refl_t = state.scratch((nz, ny, nx), "refl_t")
        refl_t[...] = th * pii
        compute_and_stash_refl_10cm(state, cfg, refl_t, state.p)
    moist_physics_finish(state, cfg, th, dt)     # theta' += mpten; h_diabatic
    sr = state.scratch((ny, nx), "mp_kessler_sr")
    sr[...] = 0.0
    return MicrophysicsDiagnostics(rainnc=rainnc, rainncv=rainncv, sr=sr)


def _apply_thompson(state: DomainState, cfg: RunConfig, dt: float, *,
                    refl_10cm_due: bool = False) -> MicrophysicsDiagnostics:
    """Keep the qualified default adapter; use the scalar port in strict mode."""
    from woof.wrf_exact import ENABLED
    adapter = _apply_thompson_strict if ENABLED else _apply_thompson_default
    return adapter(state, cfg, dt, refl_10cm_due=refl_10cm_due)


def _apply_thompson_default(
        state: DomainState, cfg: RunConfig, dt: float, *,
        refl_10cm_due: bool = False) -> MicrophysicsDiagnostics:
    """Forecast adapter for the WRF-ordered classic Thompson path.

    The source, adjustment, same-call melt/fallout, and final phase-cleanup
    order mirrors ``mp_thompson``.  Formerly reachable only behind the
    WOOF_EXPERIMENTAL_THOMPSON_MP8 process guard; promoted to a first-class
    scheme when the canonical classic tables became package data (product
    decision, product/v1 packaging lane 2026-07-28).  The table root
    resolves to the packaged directory unless WOOF_THOMPSON_TABLE_ROOT
    overrides it, and ``load_classic_device_tables`` still byte-validates
    every asset before GPU setup -- a wrong or missing root fails closed.
    """
    from woof.thompson_generations import CLASSIC_GENERATIONS
    generation = CLASSIC_GENERATIONS[getattr(cfg,"thompson_version","wrf_461")]
    if generation.backend == "column":
        from woof.core.thompson_column import apply_generated
        return apply_generated(state,cfg,dt,refl_10cm_due=refl_10cm_due)
    table_root = _thompson_table_root()
    required = ("qc", "qi", "ni", "qs", "qg", "qr", "nr")
    missing = [name for name in required
               if getattr(state, name, None) is None]
    if missing:
        raise ValueError(
            "Thompson mp=8 state lacks " + ", ".join(missing))

    from woof.core.thompson import (
        launch_adapter_entry,
        launch_adapter_finish,
        launch_adapter_masks,
        launch_adapter_prepare,
        launch_cloud_sedimentation,
        launch_cloud_saturation_adjust,
        launch_classic_graupel_number_finalize,
        launch_effective_radius,
        launch_final_phase_cleanup,
        launch_frozen_vapor_network_from_owner,
        launch_graupel_sedimentation,
        launch_hydrometeor_column_mask,
        launch_ice_sedimentation,
        launch_rain_evaporation,
        launch_rain_sedimentation,
        launch_snow_sedimentation,
        launch_warm_frozen_source_network_from_owner,
    )
    from woof.core.thompson_runtime import load_classic_device_tables

    nz, ny, nx = state.p.shape
    th = state.scratch((nz, ny, nx), "mp_th")
    pii = state.scratch((nz, ny, nx), "mp_pii")
    temperature = state.scratch((nz, ny, nx), "mp_thompson_temperature")
    dz = state.scratch((nz, ny, nx), "mp_dz8w")
    frozen_reference_density = state.scratch(
        (nz, ny, nx), "mp_thompson_frozen_reference_density")
    frozen_reference_temperature = state.scratch(
        (nz, ny, nx), "mp_thompson_frozen_reference_temperature")
    rain_reference_density = state.scratch(
        (nz, ny, nx), "mp_thompson_rain_reference_density")
    snow_melt_marker = state.scratch(
        (nz, ny, nx), "mp_thompson_snow_melt_marker")
    graupel_melt_marker = state.scratch(
        (nz, ny, nx), "mp_thompson_graupel_melt_marker")
    snow_velocity_boost = state.scratch(
        (nz, ny, nx), "mp_thompson_snow_velocity_boost")
    # mp_z8w stays drawn so the mp=8 scratch arena keeps its frozen layout
    # (tests/test_mp8_frozen.py); the layer depths below no longer pass
    # through it and nothing reads it after this adapter.
    state.scratch((nz + 1, ny, nx), "mp_z8w")

    surface_shape = (ny, nx)
    rainnc = state.scratch(surface_shape, "mp_rainnc")
    rainncv = state.scratch(surface_shape, "mp_rainncv")
    snownc = state.scratch(surface_shape, "mp_snownc")
    snowncv = state.scratch(surface_shape, "mp_snowncv")
    graupelnc = state.scratch(surface_shape, "mp_graupelnc")
    graupelncv = state.scratch(surface_shape, "mp_graupelncv")
    sr = state.scratch(surface_shape, "mp_sr")
    table_owner = load_classic_device_tables(table_root)
    # Classic Thompson's ng1d is transient but it is part of every call's
    # source/melt/fallout trajectory.  Output cadence may decide whether it is
    # consumed by reflectivity, never whether it exists or evolves.
    graupel_number_shadow = state.scratch(
        (nz, ny, nx), "mp_thompson_graupel_number_shadow")

    # The Exner function stays CuPy's divide and power: the fused launches
    # below reproduce every other operation of this adapter bit for bit on
    # every card, but the same p / P0 and powf compiled into the fused
    # kernel came out one ULP apart in 9 percent of cells on an RTX 5090.
    pii[...] = cp.power(state.p / DTYPE(c.P0), DTYPE(c.RCP))
    # One launch before any process (launch_adapter_prepare): theta,
    # temperature and the layer depths; the pre-microphysics full theta
    # parked in h_diabatic (save_pre_mp_theta); GRAUPELNCV reset (a
    # current-call diagnostic with no earlier species kernel to reset it
    # before the graupel slice); and two entry markers.  The zero/one entry
    # graupel marker is lifetime-aliased with the held-temperature buffer:
    # the column mask consumes it before cloud adjustment overwrites every
    # element with the reference temperature needed by snow fallout.  The
    # cold source writes its latent heating in-place, so WRF's
    # entry-temperature branch decision (T >= 273.15 K) is kept in
    # graupel_melt_marker and a cold cell heated across 0 C cannot execute
    # the warm source path again in this same call; the warm source consumes
    # that mask and overwrites the buffer with its held prr_gml > 0 marker
    # (it writes prr_sml > 0 separately: neither marker may alias the later
    # RHOF output).
    # The same launch makes WRF's entry rewrite (:1844-1845, :1871-1872,
    # :1900-1901, :1911, :1941-1942): cloud, ice, rain, snow and graupel
    # whose entry mixing
    # ratio is at or below R1 are ZEROED, mass and number, before any
    # process runs, and in every column, because mp_gt_driver copies the
    # rewritten 1-D arrays back whether or not the column had microphysics.
    # Classic Thompson reads these the way mp=28 does: a cloud residue joins
    # whatever the fallout and the melt bring to its level (:3943-3966,
    # :3975), an orphan number (q <= R1, n > 0, which advection leaves at
    # cloud edges and an analysis increment anywhere) is read by the
    # nucleation, the fallout, the terminal numbers and the reflectivity, and
    # the private graupel number starts from the graupel mass
    # (launch_classic_graupel_number_init).  Without it the final rain
    # differed from WRF v4.6.1's own Fortran beyond 1e-2, unexplained by
    # rounding, at 3,154 levels of two saved real-data analysis states and
    # the echo by up to 43.9 dB in 3,394 cells of one
    # (tools/thompson_real_column_parity --mp 8).  Each cell's presence is
    # read before its mass or number is written, and the zero is a +0.0,
    # as WRF writes.
    micro_columns = state.scratch(surface_shape, "mp_thompson_micro_columns")
    launch_adapter_prepare(
        state.thb, state.thp, state.phb, state.php,
        th, pii, temperature, dz, state.h_diabatic,
        state.qc, state.qi, state.ni, state.qr, state.nr, state.qs,
        state.qg, frozen_reference_temperature, graupel_melt_marker,
        graupelncv, micro_columns)
    # On the rewritten entry state (launch_adapter_entry): the private
    # graupel number, and WRF's column exit (:1646, :1827-1990, :2020): a
    # column whose entry condensate is all at or below R1 and which is
    # nowhere supersaturated over ice leaves mp_thompson before the source
    # loop, and its vapour is not floored at 1.E-10 by the terminal apply
    # (:3974).  The flag is read by the phase cleanup, which carries the
    # floor.
    launch_adapter_entry(
        state.qc, state.qi, state.qr, state.qs, state.qg,
        temperature, state.p, state.qv, graupel_number_shadow,
        micro_columns)
    launch_frozen_vapor_network_from_owner(
        state.qi, state.ni, state.qs, state.qg, state.qr, state.nr,
        temperature, state.p, state.qv, table_owner, dt, qc=state.qc,
        graupel_number_shadow=graupel_number_shadow,
        snow_velocity_boost=snow_velocity_boost)
    launch_warm_frozen_source_network_from_owner(
        state.qc, state.qr, state.nr, state.qs, state.qg,
        graupel_number_shadow, graupel_melt_marker, snow_melt_marker,
        temperature, state.p, state.qv,
        table_owner, dt)
    # WRF's rain velocity pass refreshes RHOF for every level only when the
    # post-source column contains rain.  RAINNCV is not populated until the
    # later ice fallout launch, so it safely carries this held column mask.
    # SR is refreshed only after all fallout, so its 2-D buffer safely carries
    # the graupel fallout's zero/one column guard until the graupel launch
    # consumes it.  Both masks come from one launch.
    launch_adapter_masks(
        state.qr, state.qg, frozen_reference_temperature, rainncv, sr)
    # The adjustment writes WRF's L_qc(k) into ``cloud_presence``: set from
    # the post-source cloud (:3215-3223) and cleared where the adjustment
    # leaves rc(k) at R1 (:3485), never set by it.  The buffer is the rain
    # evaporation's density output, which that kernel writes at every
    # element before anything reads it; the column mask below is the
    # presence's only reader.
    cloud_presence = rain_reference_density
    launch_cloud_saturation_adjust(
        temperature, state.p, state.qv, state.qc,
        reference_density=frozen_reference_density,
        reference_temperature=frozen_reference_temperature,
        # Full theta was saved before the source call. This scratch is not
        # read again until the final temperature-to-theta conversion.
        condensation_marker=th,
        cloud_presence=cloud_presence)
    # :3645 ``if (ANY(L_qc .eqv. .true.))`` reads L_qc as the sources set it
    # and the adjustment cleared it.  A column whose only cloud condensed
    # this step keeps that cloud where it formed, and a column whose cloud
    # the adjustment emptied does not sediment; the post-source mask the
    # adapter took before did both, which moved cloud water that WRF v4.6.1
    # leaves in place on saved real-data columns
    # (tools/thompson_real_column_parity --mp 8).  SNOWNCV is overwritten by
    # ice fallout before it becomes a public current-call diagnostic.
    launch_hydrometeor_column_mask(cloud_presence, snowncv)
    # The rain evaporation writes WRF's L_qr (:3236, zero where it failed)
    # and the :3568 rewrite (negative) into the rain fallout's density, which
    # the rain and snow fallout read below as mp=28's do: the mixing-ratio
    # stand-in for L_qr they read before fired where WRF's rr(k) sits at or
    # below R1 and missed levels whose rain the evaporation had just taken.
    launch_rain_evaporation(
        state.qr, state.nr, temperature, state.p, state.qv, dt,
        reference_density=rain_reference_density,
        graupel_melt_marker=graupel_melt_marker,
        source_density=frozen_reference_density, condensation_marker=th,
        density_carries_rain_presence=True)
    # WRF solve_em passes the physical, full-level grid%w_2 field unchanged
    # through microphysics_driver; Thompson copies w(i,k,j) directly into
    # w1d(k).  woof's matching kts:kte view is the lower full-level slice,
    # not a mass-level average (the upper extra interface is outside kte).
    # Correct mode, finding WB1 (RunConfig.thompson_cloud_surface_booking,
    # woof/core/thompson_cloud_booking.py): the cloud water this fallout
    # takes out of the column is booked in CLOUDNC instead of vanishing.
    # Exact mode (the key off) issues exactly the launcher sequence above
    # and below, nothing more.
    book_cloud = getattr(cfg, "thompson_cloud_surface_booking", False) is True
    if book_cloud:
        from woof.core.thompson_cloud_booking import (book_settled_cloud,
                                                       cloud_column_water)
        cloud_before = cloud_column_water(
            state.qc, temperature, state.p, state.qv, dz)
    launch_cloud_sedimentation(
        state.qc, temperature, state.p, state.qv, state.w[:-1], dz, dt,
        reference_density=frozen_reference_density,
        rain_active_columns=rainncv, cloud_active_columns=snowncv)
    if book_cloud:
        book_settled_cloud(state, cfg, cloud_before, state.qc, temperature,
                           state.p, state.qv, dz)
    launch_ice_sedimentation(
        state.qi, state.ni, temperature, state.p, state.qv, dz,
        rainnc, rainncv, snownc, snowncv, dt,
        reference_density=frozen_reference_density)
    # Melting snow falls at its speed blended with the rain pass's own fall
    # speed vtrk(k) (:3612-3634, :3722-3724), which a level without rain
    # inherits from above: the blend reads the rain fallout's density.
    launch_snow_sedimentation(
        state.qs, temperature, state.p, state.qv, dz,
        rainnc, rainncv, snownc, snowncv, dt,
        reference_density=frozen_reference_density,
        reference_temperature=frozen_reference_temperature,
        snow_melt_marker=snow_melt_marker,
        melt_rain_qr=state.qr,
        melt_rain_nr=state.nr,
        velocity_boost=snow_velocity_boost,
        melt_rain_density=rain_reference_density,
        melt_rain_density_carries_presence=True,
        accumulate_surface=True)
    launch_graupel_sedimentation(
        state.qg, temperature, state.p, state.qv, dz,
        rainnc, rainncv, graupelnc, graupelncv, dt,
        reference_density=frozen_reference_density,
        active_columns=sr,
        graupel_number_shadow=graupel_number_shadow,
        accumulate_surface=True)
    launch_rain_sedimentation(
        state.qr, state.nr, temperature, state.p, state.qv, dz,
        rainnc, rainncv, dt, reference_density=rain_reference_density,
        accumulate_surface=True, density_carries_rain_presence=True)
    launch_final_phase_cleanup(
        state.qc, state.qi, state.ni, temperature, state.p, state.qv,
        micro_columns=micro_columns)
    launch_classic_graupel_number_finalize(
        state.qg, temperature, state.p, state.qv,
        graupel_number_shadow)
    if refl_10cm_due:
        from woof.core.refl import compute_and_stash_refl_10cm
        compute_and_stash_refl_10cm(
            state, cfg, temperature, state.p,
            thompson_graupel_number=graupel_number_shadow)
    # The effective-radius kernel writer already applies WRF's driver-side
    # metre->micron conversion after its clamps (kernels/thompson.cu), so
    # state.effc/effi/effs receive the radiation-facing MICRON contract
    # directly -- no further adapter-side scaling is permitted here.
    launch_effective_radius(
        temperature, state.p, state.qv, state.qc,
        state.qi, state.ni, state.qs,
        state.effc, state.effi, state.effs)
    # mp_gt_driver's negative-vapour fill (:1436-1445), after the
    # reflectivity and effective radii, which read the unfilled qv1d.
    thompson_negative_vapour_fill(state.qv)
    # Theta from temperature, moist_physics_finish and SR in one launch.
    launch_adapter_finish(
        temperature, pii, th, state.thp, state.h_diabatic,
        rainncv, snowncv, graupelncv, sr, cfg, dt)
    return MicrophysicsDiagnostics(
        rainnc=rainnc, rainncv=rainncv, sr=sr,
        snownc=snownc, snowncv=snowncv,
        graupelnc=graupelnc, graupelncv=graupelncv)



def _apply_thompson_strict(
        state: DomainState, cfg: RunConfig, dt: float, *,
        refl_10cm_due: bool = False) -> MicrophysicsDiagnostics:
    """Forecast adapter for classic Thompson (mp_physics=8).

    Runs ``woof.core.thompson_classic``, the literal transcription of WRF
    v4.6.1 ``mp_gt_driver``/``mp_thompson`` for the classic scheme: WRF's
    tendency structure (sources, condensation, evaporation and
    sedimentation accumulate float tendencies; the terminal apply writes
    the state once), every expression in gfortran's order and precision,
    glibc's libm, no fused multiply-add, and the constants WRF's own
    thompson_init computes.  Word for word against WRF's Fortran on the
    committed real-column fixture (tools/thompson_real_column_parity/
    mp8_wrf_ulp.py, tests/test_thompson_classic_wrf461.py).

    The table root resolves to the packaged directory unless
    WOOF_THOMPSON_TABLE_ROOT overrides it, and
    ``load_classic_device_tables`` byte-validates every asset before GPU
    setup -- a wrong or missing root fails closed.
    """
    from woof.thompson_generations import CLASSIC_GENERATIONS
    generation = CLASSIC_GENERATIONS[getattr(cfg,"thompson_version","wrf_461")]
    if generation.backend == "column":
        from woof.core.thompson_column import apply_generated
        return apply_generated(state,cfg,dt,refl_10cm_due=refl_10cm_due)
    table_root = _thompson_table_root()
    required = ("qc", "qi", "ni", "qs", "qg", "qr", "nr")
    missing = [name for name in required
               if getattr(state, name, None) is None]
    if missing:
        raise ValueError(
            "Thompson mp=8 state lacks " + ", ".join(missing))

    from woof.core.thompson_classic import apply_classic
    from woof.core.thompson_runtime import load_classic_device_tables

    table_owner = load_classic_device_tables(table_root)
    (rainnc, rainncv, snownc, snowncv, graupelnc, graupelncv,
     sr) = apply_classic(state, cfg, dt, table_owner,
                         refl_10cm_due=refl_10cm_due)
    return MicrophysicsDiagnostics(
        rainnc=rainnc, rainncv=rainncv, sr=sr,
        snownc=snownc, snowncv=snowncv,
        graupelnc=graupelnc, graupelncv=graupelncv)


def normalize_spec_zone_ring_after_restore(state: DomainState,
                                           cfg: RunConfig,
                                           *, relocated: bool = False
                                           ) -> None:
    """Restart migration normalization: zero the specified-zone ring of
    every microphysics-owned accumulator/diagnostic slot and of
    h_diabatic after a checkpoint restore.

    Checkpoints written before the ring exclusion landed carry the SAME
    restart format version, and whole-field microphysics could leave them
    with nonzero ring RAINNC/*NCV/SR/refl and a stale ring h_diabatic.
    No WRF-valid trajectory can contain such values: the fields are
    allocator-zero-initialized (frame/module_domain.F:770-777,
    tools/gen_allocs.c:411-415) and every MP tile write is clipped
    (solve_em.F:3631-3639/:4040-4048).  Without this step the guard's
    capture/restore would preserve the stale ring FOREVER, and ring
    h_diabatic would feed every RK stage (dycore.add_h_diabatic_tendency)
    before post-RK microphysics re-pins it.  Ring prognostics (theta,
    moisture) are deliberately NOT touched -- the boundary machinery owns
    and overwrites them every step.  Value-level and idempotent (post-fix
    checkpoints already carry zero rings), so the restart format version
    is unchanged: this function IS the documented migration step for
    pre-fix checkpoints.  Periodic/open domains are untouched.

    ``relocated`` NAMES THE ONE TRAJECTORY THAT BREAKS THE IDEMPOTENCE
    CLAIM ABOVE, and it is not a pre-fix file.  A relocation shifts the
    carried accumulators in INDEX space
    (:func:`woof.core.physics_continuation.shift_continuation`), so a
    column that spent the run accumulating as interior ``i = 1`` becomes
    ring ``i = 0`` the moment the nest steps one cell east -- still the
    same ground, still that ground's real rain total, and still able to
    return to the interior on the next westward step.  A post-fix
    checkpoint of a MOVING nest therefore carries a nonzero ring
    legitimately, this normalization is not idempotent on it, and firing
    it destroys accumulated precipitation that no later step recomputes
    (MEASURED: a 1 km nest 6 h into a 24 h run resumed with its whole
    west column of RAINNC zeroed, up to 3.77 mm, frozen for the rest of
    the forecast).  WRF's own moving nest shifts RAINNC wholesale for the
    same reason.

    The exemption is exactly the RELOCATION-CARRIED set, read from the
    registry that decides what a move shifts rather than restated here,
    so the two cannot drift.  ``refl_10cm`` and ring h_diabatic are NOT
    in it and are normalized either way: nothing shifts them, the ring
    guard re-pins h_diabatic on every microphysics call, and WRF's ring
    value for both is an unconditional zero.
    """
    if cfg.mp_physics == 0:
        return
    slices = _ring_guard_slices(state, cfg)
    if slices is None:
        return
    carried: frozenset[str] = frozenset()
    if relocated:
        from woof.core.physics_continuation import continuation_slots

        carried = frozenset(continuation_slots())
    for slot in _RING_SURFACE_SLOTS + _RING_VOLUME_SLOTS:
        if slot in carried:
            continue
        arr = state.existing_scratch(slot)
        if arr is None:
            continue
        for slc in slices:
            arr[slc] = 0
    if state.h_diabatic is not None:
        for slc in slices:
            state.h_diabatic[slc] = 0


# ---------------------------------------------------------------------------
# Single-moment column schemes admitted by table row
# ---------------------------------------------------------------------------

from woof.core.column_schemes import COLUMN_SCHEMES, ColumnSchemeRow

_COLUMN_WORDS: dict = {}


def _column_words(row: ColumnSchemeRow, which: str):
    """A row's init or radar words, uploaded once per device."""
    key = (row.mp_physics, which, cp.cuda.runtime.getDevice())
    words = _COLUMN_WORDS.get(key)
    if words is None:
        if which == "init":
            words = cp.asarray(row.init_words())
        else:
            f32, f64 = row.radar_words()
            words = (cp.asarray(f32), cp.asarray(f64))
        _COLUMN_WORDS[key] = words
    return words


def column_scheme_kernel(row: ColumnSchemeRow, func: str, nz: int,
                         conservative_surface: bool = False):
    """``func`` from the row's unit, compiled at its level tier for ``nz``.

    ``conservative_surface`` is correct-mode finding WB5 (the row's
    ``conservative_surface_key``): the unit compiled with the row's
    ``conservative_surface_define``.  False compiles exactly the unit this
    function compiled before the key existed (the matched WRF arithmetic).
    """
    from woof.core.kernels import get_kernel_unflushed
    tier = row.level_tier(nz)
    defines = () if tier == row.shallow_tier else ((row.tier_define, tier),)
    if conservative_surface:
        if not row.conservative_surface_define:
            raise ValueError(f"{row.label} has no conservative surface arm")
        defines = defines + ((row.conservative_surface_define, 1),)
    return get_kernel_unflushed(row.unit, func, defines)


def _check_column_fields(label, fields: dict, shape) -> None:
    for name, value in fields.items():
        if value.shape != shape:
            raise ValueError(f"{label} {name} must have shape {shape}, "
                             f"got {value.shape}")
        if value.dtype != DTYPE:
            raise TypeError(f"{label} {name} must be float32, got {value.dtype}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{label} {name} must be C-contiguous")


def launch_column_scheme(row: ColumnSchemeRow, theta, species, den, p, pii,
                         dz, surface, effc, effi, effs, dt: float, *,
                         conservative_surface: bool = False) -> None:
    """One call of a table-row scheme over contiguous FP32 ``(nz,ny,nx)``.

    ``species`` and ``surface`` are sequences in the row's order.
    ``conservative_surface``: correct-mode finding WB5 (see
    :func:`column_scheme_kernel`).
    """
    if not isinstance(conservative_surface, bool):
        raise TypeError("conservative_surface must be a bool, got "
                        f"{conservative_surface!r}")
    shape = theta.shape
    if len(shape) != 3:
        raise ValueError(f"{row.label} fields must be 3-D, got {shape}")
    nz, ny, nx = shape
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    if len(species) != len(row.species) or len(surface) != len(row.surface):
        raise ValueError(f"{row.label} takes species {row.species} and "
                         f"surface {row.surface}")
    _check_column_fields(row.label, {
        "theta": theta, **dict(zip(row.species, species)), "den": den,
        "p": p, "pii": pii, "dz": dz, "effc": effc, "effi": effi,
        "effs": effs}, shape)
    _check_column_fields(row.label, dict(zip(row.surface, surface)), (ny, nx))
    ncol = ny * nx
    kernel = column_scheme_kernel(row, row.column_entry, nz,
                                  conservative_surface)
    kernel(((ncol + _TPB - 1) // _TPB,), (_TPB,),
           (theta, *species, den, p, pii, dz, *surface, effc, effi, effs,
            _column_words(row, "init"), DTYPE(dt),
            np.int32(nz), np.int32(ny), np.int32(nx)))


def launch_column_refl(row: ColumnSchemeRow, refl_species, t, p, refl) -> None:
    """The row's own REFL_10CM (dBZ, floored at -35 as WRF's wrapper does)."""
    shape = t.shape
    nz, ny, nx = shape
    _check_column_fields(row.label, {
        **dict(zip(row.refl_species, refl_species)), "t": t, "p": p,
        "refl": refl}, shape)
    f32, f64 = _column_words(row, "radar")
    ncol = ny * nx
    kernel = column_scheme_kernel(row, row.refl_entry, nz)
    kernel(((ncol + _TPB - 1) // _TPB,), (_TPB,),
           (*refl_species, t, p, f32, f64, refl,
            np.int32(nz), np.int32(ny), np.int32(nx)))


def _apply_column_scheme(state: DomainState, cfg: RunConfig, dt: float,
                         row: ColumnSchemeRow, *, refl_10cm_due: bool
                         ) -> MicrophysicsDiagnostics:
    """Prepare WRF's driver fields, run the row's scheme, finish heating."""
    nz, ny, nx = state.p.shape
    required = tuple(row.species) + ("effc", "effi", "effs")
    missing = [n for n in required if getattr(state, n, None) is None]
    if missing:
        raise ValueError(f"mp_physics={row.mp_physics} state lacks "
                         f"{row.label} fields: " + ", ".join(missing))
    thb = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
    phb = state.phb if state.phb.ndim == 3 else state.phb[:, None, None]
    theta = state.scratch((nz, ny, nx), "mpcol_theta")
    rho = state.scratch((nz, ny, nx), "mpcol_rho")
    pii = state.scratch((nz, ny, nx), "mpcol_pii")
    dz = state.scratch((nz, ny, nx), "mpcol_dz")
    z8w = state.scratch((nz + 1, ny, nx), "mpcol_z8w")
    theta[...] = thb + state.thp
    rho[...] = 1.0 / state.alt
    pii[...] = cp.power(state.p / DTYPE(c.P0), DTYPE(c.RCP))
    z8w[...] = (phb + state.php) / DTYPE(c.G)
    dz[...] = z8w[1:] - z8w[:-1]
    surface = [state.scratch((ny, nx), name) for name in row.surface]
    save_pre_mp_theta(state)
    launch_column_scheme(
        row, theta, [getattr(state, n) for n in row.species], rho, state.p,
        pii, dz, surface, state.effc, state.effi, state.effs, dt,
        conservative_surface=bool(row.conservative_surface_key) and bool(
            getattr(cfg, row.conservative_surface_key, False)))
    if refl_10cm_due:
        from woof.core.refl import stash_refl_10cm
        refl_t = state.scratch((nz, ny, nx), "refl_t")
        refl_t[...] = theta * pii
        refl = state.scratch((nz, ny, nx), "refl_10cm")
        launch_column_refl(row, [getattr(state, n) for n in row.refl_species],
                           refl_t, state.p, refl)
        stash_refl_10cm(state, refl)
    moist_physics_finish(state, cfg, theta, dt)
    rainnc, rainncv, snownc, snowncv, sr = surface
    return MicrophysicsDiagnostics(rainnc=rainnc, rainncv=rainncv, sr=sr,
                                   snownc=snownc, snowncv=snowncv)
def _wsm_family_selectors() -> frozenset[int]:
    from woof.core.wsm_family import family_selectors
    return family_selectors()


def _dispatch_scheme(state: DomainState, cfg: RunConfig, dt: float, *,
                     refl_10cm_due: bool) -> MicrophysicsDiagnostics:
    """Scheme dispatch exactly as before the ring guard existed."""
    row = COLUMN_SCHEMES.get(int(cfg.mp_physics))
    if row is not None:
        return _apply_column_scheme(state, cfg, dt, row,
                                    refl_10cm_due=refl_10cm_due)
    if cfg.mp_physics == 1:
        return _apply_kessler(
            state, cfg, dt, refl_10cm_due=refl_10cm_due)
    elif cfg.mp_physics == 2:
        from woof.core import lin
        return lin.apply(state, cfg, dt, refl_10cm_due=refl_10cm_due)
    elif cfg.mp_physics == 5:
        # Ferrier-Aligo.  Lazy import for the Morrison reason: mp=5 compiles
        # its own CUDA translation unit and reads its own lookup table,
        # which no other scheme's run may pay for.
        from woof.core.fer_hires import apply as apply_fer_hires
        return apply_fer_hires(
            state, cfg, dt, refl_10cm_due=refl_10cm_due)
    elif cfg.mp_physics == 3:
        # WSM3.  Lazy import for the Morrison reason: mp=3 compiles its own
        # CUDA translation unit, which no other scheme's run may build.
        from woof.core.wsm3 import apply as apply_wsm3
        return apply_wsm3(
            state, cfg, dt, refl_10cm_due=refl_10cm_due)
    elif cfg.mp_physics in (6, 106):
        # The WSM6 family module: 106 (WSM6R) is its second row
        # (woof.core.wsm6.WSM6_FAMILY), with its own kernels/wsm6r.cu.
        from woof.core.wsm6 import apply as apply_wsm6
        return apply_wsm6(
            state, cfg, dt, refl_10cm_due=refl_10cm_due)
    elif cfg.mp_physics == 8:
        return _apply_thompson(
            state, cfg, dt, refl_10cm_due=refl_10cm_due)
    elif cfg.mp_physics == 9:
        # Milbrandt-Yau two-moment.  Lazy import for the same reason the
        # Morrison and aerosol arms use one: an mp=8 or Kessler run must
        # never compile kernels/milbrandt2.cu.
        from woof.core.milbrandt2 import apply as apply_milbrandt2
        return apply_milbrandt2(
            state, cfg, dt, refl_10cm_due=refl_10cm_due)
    elif cfg.mp_physics == 16:
        # Lazy import for the Morrison reason: mp=16 compiles its own CUDA
        # translation unit, which a WSM6 or Kessler run must never build.
        from woof.core.wdm6 import apply as apply_wdm6
        return apply_wdm6(
            state, cfg, dt, refl_10cm_due=refl_10cm_due)
    elif cfg.mp_physics == 10:
        # Lazy import leaves the frozen Kessler module/import path intact.
        from woof.core.morrison import apply as apply_morrison
        return apply_morrison(
            state, cfg, dt, refl_10cm_due=refl_10cm_due)
    elif cfg.mp_physics == 28:
        # Thompson aerosol-aware.  A SIBLING adapter, not a branch inside
        # _apply_thompson: port-era isolation is recorded in
        # tests/test_mp8_frozen.py. Those historical receipts are not a
        # current mp=8 byte-freeze claim; its kernels changed later.
        # Lazy import for the same reason the Morrison arm
        # uses one -- the aerosol module pulls in eight new CUDA translation
        # units that an mp=8 or Kessler run must never compile.
        from woof.core.microphysics_aerosol import _apply_thompson_aerosol
        return _apply_thompson_aerosol(
            state, cfg, dt, refl_10cm_due=refl_10cm_due)
    elif cfg.mp_physics == 30:
        from woof.core.fast_sbm_runtime import apply as apply_fast_sbm
        return apply_fast_sbm(state,cfg,dt,refl_10cm_due=refl_10cm_due)
    elif _named_scheme(cfg.mp_physics) is not None:
        # A NAMED scheme (woof.microphysics_schemes): its adapter module is
        # woof.core.<name>, imported lazily so no other run compiles it.
        import importlib
        adapter = importlib.import_module(
            "woof.core." + _named_scheme(cfg.mp_physics).name)
        return adapter.apply(state, cfg, dt, refl_10cm_due=refl_10cm_due)
    elif cfg.mp_physics in _wsm_family_selectors():
        # A WSM-family row (woof/data/wsm_family/schemes.json), run by
        # the one table-driven runner; lazy for the Morrison reason.
        from woof.core.wsm_family import apply as apply_wsm_family
        return apply_wsm_family(
            state, cfg, dt, refl_10cm_due=refl_10cm_due)
    elif cfg.mp_physics == 50:
        # P3 one-category.  Lazy import for the reason the Morrison and
        # aerosol arms use one, plus a P3-specific one: importing this
        # module does not read the 1.6 MiB lookup table, but the adapter's
        # first call does, and a Kessler or WSM6 run must never pay that.
        from woof.core.p3 import apply as apply_p3
        return apply_p3(state, cfg, dt, refl_10cm_due=refl_10cm_due)
    else:
        # Branch-local imports avoid the microphysics -> nssl2_runtime ->
        # microphysics partial-initialization cycle.
        from woof.core.nssl2_default_hooks import NSSL2ProductionBinding
        from woof.core.nssl2_production_coordinator import (
            NSSL2ProductionConfigurationError,
        )
        from woof.core.nssl2_runtime import apply_nssl2_production

        driver = getattr(state, "physics", None)
        if driver is None or getattr(driver, "state", None) is not state:
            raise NSSL2ProductionConfigurationError(
                "mp_physics=18 requires the DomainState PhysicsDriver")
        binding = getattr(driver, "nssl2_binding", None)
        if not isinstance(binding, NSSL2ProductionBinding):
            raise NSSL2ProductionConfigurationError(
                "mp_physics=18 requires its persistent production binding")
        # The binding was built on the step the domain started with.  An
        # adaptive clock moves dt every root step, and the binding's
        # validate() refuses any step but its own, so the selector moves
        # the binding to this call's step first.  Same buffers, new step.
        stepped = binding.with_step(dt)
        if stepped is not binding:
            driver.nssl2_binding = stepped
            binding = stepped
        return apply_nssl2_production(
            state,
            cfg,
            dt,
            binding.hooks,
            output_due=refl_10cm_due,
            radiation_due=True,
            validate_values=False,
            binding=binding,
        )


#: WRF's start-of-domain CCN fill, one row per scheme that carries WRF's
#: ``scalar:qnn`` (Registry.EM_COMMON:3031 WDM6, :3054 nssl_ccn_opt):
#: (state field, rule).  The fill value is :func:`_ccn_start_value`.
#:
#: * ``"if_unset"`` is ``dyn_em/start_em.F:1750-1772`` (v4.6.1).  Every start
#:   of a domain whose scalars carry QNN takes ``grid%ccn_conc`` everywhere
#:   when the field's interior maximum is below 1.0 ("initialization of ccn
#:   not already done").  For NSSL, ``ccn_conc = nssl_cccn/1.225`` with the
#:   Registry defaults ``nssl_cccn = 0.5e9`` and ``nssl_ccn_is_ccna = 0``
#:   (Registry.EM_COMMON:2408, :2422): the FP32 word is
#:   ``NSSL2_BACKGROUND_CCN_PER_KG``.
#: * ``"always"`` is ``phys/module_mp_wdm6.F:219-227``: WDM6 overwrites its
#:   whole CCN window with ``ccn0`` (the namelist ``ccn_conc``) at
#:   ``itimestep == 1``, whatever the start held.
#:
#: Allocation already seeds both values (``woof/core/state.py``), but the
#: native WRF-file door then loads the file's QNCCN over them, and real.exe
#: writes that field as zeros.  Before this row existed an NSSL forecast
#: started from a real.exe file ran with no CCN at all: no droplet
#: activation, no cloud water where WRF condenses (the coverage-suite
#: matched run of 2026-10-07: QNCCN 0 against WRF's 4.08e8 on every word at
#: step 0, and 0.95 K of missing latent heating after one 60 s step).
_CCN_START_FILL = {
    16: ("nn", "always"),
    18: ("qnn", "if_unset"),
}


def _ccn_start_value(cfg: RunConfig) -> float:
    if int(cfg.mp_physics) == 16:
        return float(cfg.wdm6_ccn_conc)
    from woof.core.microphysics_transition import NSSL2_BACKGROUND_CCN_PER_KG
    return float(NSSL2_BACKGROUND_CCN_PER_KG)


def ccn_start_fill(state: DomainState, cfg: RunConfig) -> dict[str, object]:
    """WRF's start-of-domain CCN fill (see ``_CCN_START_FILL``).

    Returns ``{}`` when the scheme carries no CCN scalar or when the fill
    changes no word; otherwise a receipt naming the field, the rule, the
    FP32 value written and the maximum the field held before.
    """
    row = _CCN_START_FILL.get(int(cfg.mp_physics))
    if row is None:
        return {}
    name, rule = row
    field = getattr(state, name, None)
    if field is None:
        return {}
    value = DTYPE(_ccn_start_value(cfg))
    prior_max = float(cp.max(field))
    if rule == "if_unset" and prior_max >= 1.0:
        return {}
    if rule == "always" and bool(cp.all(field == value)):
        return {}
    field[...] = value
    return {"field": name, "rule": rule, "value": float(value),
            "prior_max": prior_max}


def microphysics_init(state: DomainState, cfg: RunConfig) -> dict[str, object]:
    """WRF ``microphysics_init``'s ONE-TIME, per-domain scheme setup.

    The analogue of ``phys/module_physics_init.F::microphysics_init``, which
    WRF calls once per domain at construction and never again.  It is
    separate from :func:`apply` for the reason WRF keeps it separate: what
    it does is not a tendency, and doing it twice is not idempotent in
    general.  Returns a RECEIPT -- a mapping naming what actually ran --
    rather than ``None``, so a caller can print it and a test can assert on
    it.  ``{}`` means "this scheme needs no domain-construction step", which
    is true of every scheme woof shipped before mp=28.

    ``mp_physics = 28`` is the first scheme with real work here.  WRF's
    ``thompson_init`` fills a SYNTHETIC CCN/IN profile whenever the
    water/ice-friendly aerosol fields arrive unset
    (module_mp_thompson.F:482-559; the CCN decision at :493 and the IN
    decision at :530 are two independent ``MAXVAL`` tests), and derives the
    surface emission ``nwfa2d`` from it at :509-510.  Nothing inside
    ``mp_gt_driver`` ever refills that profile.  Without this call an mp=28
    domain integrates with ``nwfa``/``nifa`` at whatever the state
    allocation left -- exactly zero on a cold start -- which is not an
    error anywhere: the terminal apply's clamps
    (:3972-4021) simply hold the aerosol at its floors, the run stays
    finite, and the aerosol-aware physics is silently inert.  That is the
    single most dangerous failure mode this scheme has, which is why the
    hook is a named, receipt-returning function rather than a line inside
    an adapter.

    THE CALLER.  Domain construction, once, before the first step, beside
    the other one-time physics initializations --
    ``woof/core/physics.py::initialize_physics``, which is woof's
    ``module_physics_init.F`` and which every mp-bearing configuration
    already reaches (``physics_driver_required`` is true whenever
    ``cfg.mp_physics`` is nonzero, physics.py:241-250).  WP-11a does not own
    that file; until the call lands there, an mp=28 forecast started from a
    zero-aerosol state runs the inert-aerosol configuration described above.
    Calling this per step would be worse than not calling it: it would
    overwrite an advected, activated and scavenged aerosol field with the
    synthetic profile every step while leaving every bound intact.
    """
    ccn = ccn_start_fill(state, cfg)
    receipt = {"ccn_start_fill": ccn} if ccn else {}
    return {**receipt, **_scheme_init(state, cfg)}


def _scheme_init(state: DomainState, cfg: RunConfig) -> dict[str, object]:
    """The scheme-specific half of :func:`microphysics_init`."""
    named = _named_scheme(cfg.mp_physics)
    if named is not None and named.aerosol_aware:
        # A named aerosol-aware scheme takes the same once-per-domain
        # synthetic CCN/IN profile mp=28 takes when the analysis carries no
        # aerosol (the named first-call init is inert when the host passes
        # aerosol arrays, so the host must seed them).
        from woof.core.microphysics_aerosol import aerosol_profile_fill
        return {"aerosol_profile": aerosol_profile_fill(state)}
    if cfg.mp_physics != 28:
        # Kessler/WSM6/Thompson/Morrison/NSSL have no domain-construction
        # step in woof: their tables are loaded lazily at first launch and
        # their state is initialized by woof/core/state.py.  Returning an
        # empty receipt (rather than raising) is what lets the caller be an
        # unconditional line in the init path.
        return {}
    from woof.core.microphysics_aerosol import thompson_aerosol_init_fill
    return {"thompson_aerosol_profile": thompson_aerosol_init_fill(state, cfg)}


def apply(state: DomainState, cfg: RunConfig, dt: float, *,
          refl_10cm_due: bool = False) -> MicrophysicsDiagnostics | None:
    """Apply the configured microphysics to ``state`` over ``dt`` seconds.

    ``cfg.mp_physics = 0`` returns immediately (bitwise no-op); ``1`` runs
    Kessler, ``2`` runs Purdue Lin, ``3`` runs WSM3, ``4`` runs WSM5 (column row),
    ``5`` runs Ferrier-Aligo, ``6`` runs WSM6, ``8`` runs classic Thompson, ``10`` runs
    Morrison two-moment, ``18`` runs NSSL two-moment, and ``28`` runs
    aerosol-aware Thompson (all require moisture); anything else raises.
    ``refl_10cm_due`` is woof's history-step
    ``diagflag`` equivalent.  The active adapter computes from its unchanged
    prepared pressure and post-scheme temperature before
    ``moist_physics_finish``.

    On specified/nested domains the call excludes the outermost
    ``spec_zone`` mass ring exactly as WRF's clipped tiles do
    (solve_em.F:3618-3639/:4040-4048; module docstring): the ring is
    captured before dispatch and bit-restored afterwards, ring
    h_diabatic is pinned to WRF's 0, and ring RAINNC/SNOWNC/GRAUPELNC,
    the per-call *NCV/SR diagnostics, and the REFL_10CM stash keep their
    never-written values (exactly 0 from a fresh init).  Periodic/open
    configs (WRF: sz = 0) keep the whole-field call bitwise.

    A due diagnostic with ``mp_physics = 0`` is an invalid schedule and
    raises instead of silently producing an output frame without radar data.

    Radar latent heating (WRF ``mp_tend_radar = 1``,
    module_big_step_utilities_em.F:5913-5938/:5991-6005) is one optional
    read: a forcing attached to the state as ``radar_tten_forcing``
    (:mod:`woof.da.radar_tten`).  With none attached the call is
    :func:`_apply_scheme` and nothing else.  With one, theta as it stood
    before the scheme is kept, the scheme runs unchanged, and each covered
    interior point below the top level then takes ``tendency * dt`` in
    place of the microphysics increment; ``h_diabatic`` keeps the
    microphysics rate everywhere (:6014).  The replacement is made here
    rather than inside :func:`moist_physics_finish` because classic
    Thompson finishes in its own fused kernel and never calls that
    function, and a forcing that one scheme silently ignored would be a
    refusal nobody raised.  The scheme runs under the forcing's
    ``mp_tend_lim`` (HRRR pairs 0.07 K/s with ``mp_tend_radar = 1``,
    parm/conus/hrrr_wrfpre.nl:108-109), which WRF applies to the
    microphysics increment before the radar select (:5968-5969).
    """
    forcing = getattr(state, "radar_tten_forcing", None)
    if forcing is None:
        return _apply_scheme(state, cfg, dt, refl_10cm_due=refl_10cm_due)
    forcing.before_microphysics(state, cfg, dt)
    scheme_cfg = forcing.scheme_config(cfg)
    result = _apply_scheme(state, scheme_cfg, dt,
                           refl_10cm_due=refl_10cm_due)
    forcing.after_microphysics(state, scheme_cfg, dt,
                               ring_width=_ring_guard_width(cfg))
    return result


def _ring_guard_width(cfg: RunConfig) -> int:
    """The width :func:`_ring_guard_slices` excludes, 0 for periodic/open
    configs (WRF solve_em.F:3618-3622)."""
    if not (getattr(cfg, "specified", False)
            or getattr(cfg, "nested", False)):
        return 0
    return max(int(cfg.spec_zone), 0)


#: Two-moment rain schemes whose rain number a cumulus scheme's feedback
#: does not supply: classic Thompson (8) and aerosol-aware Thompson (28).
#: Morrison (10) seeds its own inside the scheme (woof.core.morrison's
#: kernel, WRF module_mp_morr_two_moment.F:1332-1343), so it is not listed.
CUMULUS_RAIN_NUMBER_SCHEMES = frozenset({8, 28})

#: Cells per slab of :func:`seed_cumulus_rain_number`: its few float32
#: temporaries stay near 8 MB each on any grid.
_SEED_SLAB_CELLS = 2_000_000


def seed_cumulus_rain_number(state: DomainState, cfg: RunConfig,
                             dt: float) -> bool:
    """Give the rain a cumulus scheme feeds two-moment Thompson its number.

    THE DEFECT.  Kain-Fritsch feeds precipitation back to the grid as rain
    MASS, ``RQRCUTEN`` (all of a shallow cloud's: ``fbfrc = 1``,
    module_cu_kfeta.F), and nothing for the rain number.  Classic and
    aerosol-aware Thompson carry a prognostic rain number, and given a mass
    with no number ``mp_thompson`` clamps the mean volume diameter at
    2.5 mm, so a few mg/kg of shallow-cumulus drizzle became 2.5 mm drops
    and 20-30 dBZ of composite reflectivity.  WOOF Mexico City, 29 Oct 12Z
    (GFS; 6.75 km parent with Kain-Fritsch, 2.25 km nest without):
    13 percent of the parent read >= 20 dBZ from layers 0-1.2 km deep
    holding 4-10 mg/kg of rain and no ice; the relaxation zone's shallow
    convection drew straight 20-30 dBZ strips along the parent's edges; and
    the nest took the parent's drizzle at its boundary every parent step,
    a frame of 20-25 dBZ strips and trace rain around an interior that had
    none.  WRF v4.6.1 does the same with Thompson.  Its Morrison does not,
    because it seeds a number for the cumulus rain it is handed.

    THE RULE.  Before the scheme runs, the step's fed rain ``RQRCUTEN*dt``
    gets the number Thompson's own cold start gives rain that has a mass
    and no number, ``make_RainNumber`` (dyn_em/module_initialize_real.F
    :9144-9175, :func:`woof.core.thompson_entry.make_rain_number`: an
    exponential at the Marshall-Palmer intercept, 8e6 m-4, rising to 8e8
    between 0 and -2 C), per step as Morrison seeds per step, wherever the
    rate is at least 1e-10 kg/kg/s (Morrison's threshold).  Its closed
    form is ``N0**0.75 * (q / (pi rho_w rho**3))**0.25`` per kg, Morrison's
    own expression with Thompson's intercept.  Mass, heat and every other
    field are untouched.  Off under GPUWM_WRF_EXACT, so proof runs keep
    WRF's own behaviour.
    """
    if (int(getattr(cfg, "mp_physics", 0)) not in CUMULUS_RAIN_NUMBER_SCHEMES
            or not getattr(cfg, "cu_physics", 0)):
        return False
    from woof.wrf_exact import ENABLED
    if ENABLED:
        return False
    rates = getattr(getattr(state, "physics", None), "cu_rates", None)
    rqr = None if rates is None else rates.get("rqrcuten")
    nr = getattr(state, "nr", None)
    if rqr is None or nr is None:
        return False
    if rqr.shape != nr.shape:
        raise ValueError(
            f"the cumulus rain rate {rqr.shape} does not cover the rain "
            f"number {nr.shape}; seeding it would number the wrong cells")
    nz, ny, nx = nr.shape
    slab = max(1, _SEED_SLAB_CELLS // max(1, ny * nx))
    # (rqr dt / (pi rho_w))**0.25 * 8**0.75 * 10**(0.75 clip(279.15-T, 6, 8))
    # * rho**-0.75, with rho = p / (R_d T): make_RainNumber(q rho, T) / rho.
    mass_scale = DTYPE(float(dt) / (np.pi * 1000.0))
    log_ten = DTYPE(0.75 * np.log(10.0))
    eight = DTYPE(8.0 ** 0.75)
    for k0 in range(0, nz, slab):
        k = slice(k0, min(nz, k0 + slab))
        thb = (state.thb[k] if state.thb.ndim == 3
               else state.thb[k, None, None])
        p = state.p[k]
        temp = cp.power(p * DTYPE(1.0 / c.P0), DTYPE(c.RCP))
        temp *= state.thp[k] + thb
        number = DTYPE(279.15) - temp
        cp.clip(number, DTYPE(6.0), DTYPE(8.0), out=number)
        number *= log_ten
        cp.exp(number, out=number)
        rho = p / temp
        rho *= DTYPE(1.0 / c.RD)
        cp.power(rho, DTYPE(-0.75), out=rho)
        number *= rho
        fed = rqr[k] * mass_scale
        cp.maximum(fed, DTYPE(0.0), out=fed)
        cp.power(fed, DTYPE(0.25), out=fed)
        number *= fed
        number *= eight
        number *= rqr[k] >= DTYPE(1.0e-10)
        nr[k] += number
    return True


def _apply_scheme(state: DomainState, cfg: RunConfig, dt: float, *,
                  refl_10cm_due: bool = False
                  ) -> MicrophysicsDiagnostics | None:
    """:func:`apply` without the optional radar forcing: the validation,
    the specified-zone ring guard and the scheme dispatch."""
    if cfg.mp_physics == 0:
        if refl_10cm_due:
            raise ValueError("REFL_10CM is due without active microphysics")
        return None
    if (cfg.mp_physics not in (1, 2, 3, 5, 6, 8, 9, 10, 16, 18, 24, 28, 30, 50, 106)
            and _named_scheme(cfg.mp_physics) is None
            and int(cfg.mp_physics) not in COLUMN_SCHEMES):
        raise ValueError(f"unknown mp_physics={cfg.mp_physics} "
                         "(0 = none, 1 = Kessler, 2 = Purdue Lin, 3 = WSM3, 4 = WSM5, "
                         "5 = Ferrier-Aligo, 6 = WSM6, "
                         "8 = Thompson, 9 = Milbrandt-Yau, "
                         "10 = Morrison, 16 = WDM6, "
                         "18 = NSSL, 24 = WSM7, 28 = Thompson aerosol-aware, "
                         "50 = P3 one-category, 106 = WSM6R)")
    if state.qv is None:
        raise ValueError(f"mp_physics={cfg.mp_physics} requires "
                         "cfg.moist=True: the state "
                         "carries no qv/qc/qr arrays")
    if cfg.mp_physics == 18:
        # An NSSL variant's absent categories are pinned to exact zero
        # HERE, ahead of the spec-zone ring snapshot, and not inside the
        # scheme: the ring is bit-restored after the call, so a pin applied
        # further in would leave a nested or specified domain's ring
        # carrying a hydrometeor the resolved mode has no physics for.  The
        # default lane pins nothing and this costs it one dataclass.
        # Branch-local import for the same partial-initialization reason as
        # the dispatch arm below.
        from woof.core.nssl2_contract import resolve_nssl2_mode_for_config
        from woof.core.nssl2_runtime import pin_absent_nssl2_fields
        pin_absent_nssl2_fields(state, resolve_nssl2_mode_for_config(cfg))
    slices = _ring_guard_slices(state, cfg)
    if slices is None:
        seed_cumulus_rain_number(state, cfg, dt)
        result = _dispatch_scheme(state, cfg, dt, refl_10cm_due=refl_10cm_due)
    else:
        saved, captured_slots = _capture_spec_zone_ring(state, slices)
        try:
            # The restore must run on EVERY exit path: a post-mutation
            # raise (e.g. the reflectivity handoff refusing a missing
            # driver or an unconsumed stash, refl.py stash_refl_10cm) would
            # otherwise leave ring columns mutated that WRF never
            # dispatches at all.  The seeded rain number is inside the
            # guard, so the ring keeps the number WRF's clipped tiles keep.
            seed_cumulus_rain_number(state, cfg, dt)
            result = _dispatch_scheme(state, cfg, dt,
                                      refl_10cm_due=refl_10cm_due)
        finally:
            _restore_spec_zone_ring(state, slices, saved, captured_slots)
    if getattr(cfg, "mp_zero_out", 0):
        microphysics_zero_out(state, cfg)
    return result


#: The species WRF's Registry places first in each scheme's ``scalar``
#: array (v4.6.1 Registry.EM_COMMON packages; the fork's thompson and
#: thompsonaero packages put qni first too, :2791/:2804), in woof's
#: spelling.  module_microphysics_zero_out tests ``n .NE. P_QV`` on the
#: array INDEX, and P_QV is PARAM_FIRST_SCALAR, so the first scalar takes
#: vapour's rule (floored at zero under 2, untouched under 1) and every
#: other scalar takes the threshold.  A scheme with no scalar array has no
#: entry.
WRF_FIRST_SCALAR_SPECIES: dict[int, str] = {
    30: "ff1i01",
    8: "ni", 10: "ni", 28: "ni", 50: "ni",   # qni
    9: "nc",                                 # qnc (milbrandt2mom)
    16: "nn",                                # qnn (wdm6scheme)
    18: "qndrop",                            # qndrop (v4.6.1 nssl2mconc)
}

_MP_ZERO_OUT_KERNELS: dict[str, object] = {}


def _mp_zero_out_kernel(name: str):
    """The two mp_zero_out kernels, built on first use, not at import:
    tools/health_field_census.py imports this module under a host-only
    array backend that refuses kernel construction, and an import-time
    ElementwiseKernel failed all 14 census checks."""

    kernel = _MP_ZERO_OUT_KERNELS.get(name)
    if kernel is None:
        if name == "floor_zero":
            kernel = cp.ElementwiseKernel(
                "", "float32 x", "if (x < 0.0f) x = 0.0f;",
                "gpuwm_mp_floor_zero")
        else:
            kernel = cp.ElementwiseKernel(
                "float32 t", "float32 x", "if (x < t) x = 0.0f;",
                "gpuwm_mp_zero_below")
        _MP_ZERO_OUT_KERNELS[name] = kernel
    return kernel


def microphysics_zero_out(state: DomainState, cfg: RunConfig) -> None:
    """WRF ``microphysics_zero_outb`` then ``_outa`` after the scheme.

    phys/module_microphysics_zero_out.F (the same file in WRF v4.6.1 and
    the NOAA-EMC WRFV3.9 fork), called from solve_em right after the
    microphysics driver (v4.6.1 :4009-4080, fork :4078-4141):

    - ``_outb`` floors every species at zero on the outermost ring
      (j = jds, jde-1 and i = ids, ide-1, whatever the boundary type);
    - ``_outa`` then, over the interior that excludes the ``spec_zone``
      ring on a specified or nested domain (the whole domain otherwise),
      sets every species but the first of its array to zero where it is
      below ``mp_zero_out_thresh``, and under ``mp_zero_out = 2`` floors
      the first (vapour in the moist array) at zero.

    v4.6.1 runs both on the moist array, and on the scalar array as well
    when ``mp_zero_out_all = 1``; the fork always runs them on every array
    (v4.6.1's 1). Active chemical rows declare their chem or tracer group
    in the species table. Each group starts at Registry index 2 after
    dummy index 1, and compares that index with P_QV = 2. Applied after the
    fused moist_physics_finish, which reads theta only, so the order
    against it does not change a prognostic value.
    """
    mode = int(cfg.mp_zero_out)
    if mode == 0 or (state.qv is None and getattr(state, "qke_adv", None) is None):
        return
    from woof.core.moist import WRF_MOIST_ARRAY_SPECIES, moist_species
    from woof.core.dycore import _boundary_forced
    names = moist_species(state)
    moist = [name for name in names if name in WRF_MOIST_ARRAY_SPECIES]
    arrays = [(moist, "qv")] if moist else []
    if int(cfg.mp_zero_out_all):
        scalars = [name for name in names
                   if name not in WRF_MOIST_ARRAY_SPECIES]
        if scalars:
            first = WRF_FIRST_SCALAR_SPECIES.get(int(cfg.mp_physics))
            if scalars == ["qke_adv"]:
                # MYNN may provide the only Registry scalar member when
                # the microphysics package has no number tracers.
                first = "qke_adv"
            if first not in scalars:
                raise ValueError(
                    f"mp_zero_out_all = 1 under mp_physics = "
                    f"{cfg.mp_physics}: no WRF Registry scalar order is "
                    "recorded for this scheme, so the species that takes "
                    "vapour's rule cannot be named (WRF_FIRST_SCALAR_"
                    "SPECIES)")
            arrays.append((scalars, first))
        chem = getattr(state, "chem", None)
        if chem is not None:
            # WRF calls the operator separately for chem and tracer. The
            # first real row of EACH array has the numeric P_QV index;
            # it is not selected by gas/aerosol family or species name.
            for group in ("chem", "tracer"):
                members = [row.state_attr for row in chem.transported
                           if row.wrf_array == group]
                if members:
                    arrays.append((members, members[0]))
    thresh = np.float32(cfg.mp_zero_out_thresh)
    sz = int(cfg.spec_zone) if _boundary_forced(cfg) else 0
    for species, first in arrays:
        for name in species:
            field = getattr(state, name, None)
            if field is None:
                continue
            _, ny, nx = field.shape
            # microphysics_zero_outb: the outermost ring.
            for ring in (field[:, 0, :], field[:, ny - 1, :],
                         field[:, :, 0], field[:, :, nx - 1]):
                _mp_zero_out_kernel("floor_zero")(ring)
            # microphysics_zero_outa: the interior inside spec_zone.
            inner = field[:, sz:ny - sz, sz:nx - sz]
            if name == first:
                if mode == 2:
                    _mp_zero_out_kernel("floor_zero")(inner)
            else:
                _mp_zero_out_kernel("zero_below")(thresh, inner)
