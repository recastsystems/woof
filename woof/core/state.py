"""Forecast/setup state container with FP32 arrays on one explicit backend.

In its default CuPy forecast mode, ``DomainState`` owns *all* persistent
device memory. Native stock-WRF setup/export may explicitly use NumPy host
arrays instead. Everything is allocated
once in ``__init__`` (shapes depend only on ``RunConfig``); base-state and
coordinate arrays are filled in place by ``load_base``.  Nothing else in the
model may allocate persistent device arrays: transient work buffers must go
through :meth:`DomainState.scratch`.

Array layout is ``(nz, ny, nx)`` with x fastest.  Staggering (WRF-ARW):
``u (nz,ny,nx+1)``, ``v (nz,ny+1,nx)``, ``w``/``phi (nz+1,ny,nx)``.
"""

from __future__ import annotations

from woof.core.column_schemes import COLUMN_SCHEMES

import math
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
import threading

import numpy as np

#: Why CuPy is unavailable, when it is.  Kept so the deferred failure
#: below can name the REAL cause instead of "not installed".
_CUPY_UNAVAILABLE: BaseException | None = None

try:  # Native stock-WRF setup/export has a genuine NumPy-only route.
    import cupy as cp
except Exception as _error:  # pragma: no cover - isolated subprocess
    # Deliberately not `except ImportError`.  An ABSENT CuPy raises
    # ImportError and was handled; an INSTALLED BUT UNLOADABLE one does
    # not.  A cupy-cuda12x wheel on a CUDA-13 box, or a missing
    # nvrtc DLL, raises RuntimeError or OSError from deep inside the
    # import -- and because woof.cli reaches this module at import time
    # (cli -> downscale -> offline_child -> here), that killed the whole
    # command line.  `woof run-plan --probe`, whose entire job is to
    # preflight an install and which reads the card through NVML without
    # ever touching CuPy, could not run on exactly the broken installs
    # it exists to diagnose.
    #
    # For every CPU-only path an unloadable CuPy is worth precisely what
    # an absent one is worth, so it is treated the same and the
    # difference is surfaced at USE time by _require_cupy below, which
    # names the original error.
    cp = None
    _CUPY_UNAVAILABLE = _error

from woof.config import (
    CUMULUS_ADVECTIVE_FORCING_SCHEMES, SASE_PBL_SCHEME, RunConfig)
from woof.core import constants as c
from woof.core.grid import BaseState, VerticalCoord, rebalance_hydrostatic
# Single source for the SASE realizability floor (the e_sgs cold-start
# fill).  From woof.core, not woof.verify: the standalone CPU
# preprocessing distribution omits the verification tree.
from woof.core.sase_limits import E_MIN as SASE_E_MIN
from woof.core.wdm6_constants import WDM6_NUMBER_SPECIES
from woof.microphysics_schemes import scheme as _named_scheme
from woof.wrf_exact import DIAGNOSTICS_ENABLED

#: Model-field dtype.  FP64 only in setup code and test references.  Keeping
#: the scalar type available without CuPy lets the Rust/NumPy native-input
#: path import on a CPU-only host; CUDA forecast allocation still fails with
#: the explicit optional-dependency error below.
DTYPE = np.float32 if cp is None else cp.float32


def _require_cupy():
    """Return the array backend without contacting a CUDA device.

    Backend selection also serves host setup, restart inspection and declared
    workspace sizing. Local-device opt-out forbids device contact, not these
    operations. Admission belongs at a CUDA allocation or forecast step.
    """
    if cp is None:
        from woof.forecast_gpu import ForecastGPURequired

        detected = "CuPy is required for CUDA forecast state"
        if _CUPY_UNAVAILABLE is not None:
            if not isinstance(_CUPY_UNAVAILABLE, ImportError):
                detected += "; CuPy IS installed here but failed to load"
            detected += f": {type(_CUPY_UNAVAILABLE).__name__}: {_CUPY_UNAVAILABLE}"
        raise ForecastGPURequired(detected) from None
    return cp


def _forecast_array_module():
    """Admit device allocations, retaining host-backed workspace inspection."""
    cuda = _require_cupy()
    if getattr(cuda, 'ndarray', None) is not np.ndarray:
        from woof.forecast_gpu import require_forecast_gpu

        require_forecast_gpu()
    return cuda


class SharedDycoreStateWorkspace:
    """Per-symbol max-domain storage for restart-rebuilt state arrays.

    The symbol inventory is supplied by preflight's view of the restart
    manifest, so this allocation mechanism owns no second hand-copied list.
    Each :class:`DomainState` receives a C-contiguous shaped prefix of the
    corresponding symbol backing.  The non-blocking ownership token guards
    the executor's one-STEP-or-FORCE-at-a-time correctness contract.
    """

    def __init__(self, symbol_shapes: Mapping[str, tuple[int, ...]]):
        cuda = _forecast_array_module()
        self._symbol_shapes: dict[str, tuple[int, ...]] = {}
        self._buffers: dict[str, cp.ndarray] = {}
        for symbol in sorted(symbol_shapes):
            shape = tuple(int(extent) for extent in symbol_shapes[symbol])
            if not shape or any(extent < 1 for extent in shape):
                raise ValueError(
                    f"invalid shared dycore-state shape for {symbol!r}: "
                    f"{shape}")
            self._symbol_shapes[symbol] = shape
            self._buffers[symbol] = cuda.zeros(shape, dtype=DTYPE)
        if not self._buffers:
            raise ValueError("shared dycore-state workspace has no symbols")
        self._owner_lock = threading.Lock()
        self._owner = None

    def view(self, symbol: str, shape, dtype=None) -> cp.ndarray:
        """Return one C-contiguous shaped prefix of ``symbol``'s backing."""
        shape = tuple(shape) if isinstance(shape, (tuple, list)) else (shape,)
        shape = tuple(int(extent) for extent in shape)
        requested_dtype = np.dtype(DTYPE if dtype is None else dtype)
        if requested_dtype != np.dtype(DTYPE):
            raise TypeError(
                f"shared dycore-state symbol {symbol!r} is float32, "
                f"requested {requested_dtype}")
        try:
            backing = self._buffers[symbol]
        except KeyError as exc:
            raise KeyError(
                f"rebuilt state symbol {symbol!r} is not in this workspace") \
                from exc
        requested = math.prod(shape)
        if requested > backing.size:
            raise ValueError(
                f"shared dycore-state symbol {symbol!r} capacity is "
                f"{backing.size} values ({self._symbol_shapes[symbol]}), "
                f"requested {requested} values ({shape})")
        return backing.reshape(-1)[:requested].reshape(shape)

    def backing(self, symbol: str) -> cp.ndarray:
        """Return a symbol backing for identity/debug inspection only."""
        return self._buffers[symbol]

    @property
    def symbols(self) -> frozenset[str]:
        return frozenset(self._buffers)

    @property
    def symbol_shapes(self) -> dict[str, tuple[int, ...]]:
        return dict(self._symbol_shapes)

    @property
    def nbytes(self) -> int:
        return sum(int(buf.nbytes) for buf in self._buffers.values())

    @property
    def owner(self):
        """Current executor ownership token, or ``None`` between turns."""
        return self._owner

    @contextmanager
    def acquire(self, owner):
        """Fail loudly if another STEP/FORCE still owns the workspace."""
        if not self._owner_lock.acquire(blocking=False):
            raise RuntimeError(
                f"shared dycore-state workspace is already owned by "
                f"{self._owner!r}; {owner!r} cannot observe it concurrently")
        self._owner = owner
        try:
            yield self
        finally:
            self._owner = None
            self._owner_lock.release()


class ScratchArena:
    """Shared zero-filled backing for proven step-local scratch slots.

    One backing allocation is retained per independent slot.  Audited slots
    with disjoint lifetimes may alias a larger backing allocation.  A domain
    receives a contiguous prefix reshaped to its requested dimensions, so
    differently sized domains can reuse the same allocation while the flat
    schedule steps exactly one domain at a time.  The lifetime classification
    and alias proof live beside the enforced registry in
    :mod:`woof.core.preflight`; this class is deliberately only an
    allocation/view mechanism.
    """

    def __init__(self, slot_shapes: Mapping[str, tuple[int, ...]], *,
                 slot_aliases: Mapping[str, str] | None = None):
        cuda = _forecast_array_module()
        self._slot_shapes: dict[str, tuple[int, ...]] = {}
        self._buffers: dict[str, cp.ndarray] = {}
        aliases = dict(slot_aliases or {})
        for slot, raw_shape in slot_shapes.items():
            shape = tuple(int(extent) for extent in raw_shape)
            if not shape or any(extent < 0 for extent in shape):
                raise ValueError(f"invalid scratch-arena shape for {slot!r}: "
                                 f"{shape}")
            self._slot_shapes[slot] = shape
            if slot not in aliases:
                # DomainState.scratch has always zero-allocated every slot.
                # The arena preserves that initialization exactly; the audit
                # admits only slots overwritten before their first read.
                self._buffers[slot] = cuda.zeros(shape, dtype=DTYPE)
        for slot, target in aliases.items():
            if slot not in self._slot_shapes:
                raise KeyError(f"scratch-arena alias {slot!r} has no shape")
            if target not in self._buffers:
                raise KeyError(
                    f"scratch-arena alias target {target!r} is unavailable")
            requested = math.prod(self._slot_shapes[slot])
            if requested > self._buffers[target].size:
                raise ValueError(
                    f"scratch-arena alias {slot!r} needs {requested} values, "
                    f"but target {target!r} has {self._buffers[target].size}")
            self._buffers[slot] = self._buffers[target]

    def has_slot(self, slot: str) -> bool:
        return slot in self._buffers

    def view(self, shape, slot: str, dtype=None) -> cp.ndarray:
        """Return a shaped prefix view of the slot's max-sized backing."""
        shape = tuple(shape) if isinstance(shape, (tuple, list)) else (shape,)
        shape = tuple(int(extent) for extent in shape)
        requested_dtype = np.dtype(DTYPE if dtype is None else dtype)
        if requested_dtype.itemsize != np.dtype(DTYPE).itemsize:
            # The registry accounts slots by element count at one width;
            # a wider or narrower element would make the prefix view a
            # different number of values than the shape it was priced at.
            raise TypeError(
                f"scratch arena slot {slot!r} holds {np.dtype(DTYPE).itemsize}"
                f"-byte elements, requested {requested_dtype} "
                f"({requested_dtype.itemsize}-byte); request a dtype of the "
                "same width, or keep the slot off the shared arena")
        try:
            backing = self._buffers[slot]
        except KeyError as exc:
            raise KeyError(f"scratch slot {slot!r} is not in this arena") \
                from exc
        requested = math.prod(shape)
        if requested > backing.size:
            raise ValueError(
                f"scratch slot {slot!r} arena capacity is {backing.size} "
                f"values ({self._slot_shapes[slot]}), requested {requested} "
                f"values ({shape})")
        view = backing.reshape(-1)[:requested]
        if requested_dtype != np.dtype(DTYPE):
            # Same-width reinterpretation of the same bytes (the mp=28
            # entry diagnosis asks for its two int32 slots this way):
            # the zero fill reads as integer zero, and every arena slot is
            # written before it is read.
            view = view.view(requested_dtype)
        return view.reshape(shape)

    @property
    def slot_shapes(self) -> dict[str, tuple[int, ...]]:
        return dict(self._slot_shapes)

    @property
    def nbytes(self) -> int:
        unique = {id(buf): buf for buf in self._buffers.values()}
        return sum(int(buf.nbytes) for buf in unique.values())

    def poison(self) -> None:
        """Fill every unique arena backing with NaNs.

        This is the Task-14 debug lever for proving the lifetime audit at
        runtime.  The model executor calls it only when explicitly requested
        between complete domain turns; production runs leave it disabled.
        Every arena-admitted slot is write-before-read by construction, so a
        surviving NaN identifies an invalid lifetime classification quickly.
        """
        unique = {id(buf): buf for buf in self._buffers.values()}
        for buf in unique.values():
            buf.fill(DTYPE(np.nan))


def build_shared_scratch_arena(domains: Iterable[object],
                               tree: Iterable[object] | None = None
                               ) -> ScratchArena:
    """Build the deterministic shared arena for a domain configuration set.

    This is the Task-14 handoff: ``build_experiment`` passes its parent-first
    ``DomainConfig`` sequence here, then injects the returned arena into every
    ``DomainState``. Shape selection and lifetime admission share the same
    registry used by preflight, and this function does not mutate the domains.

    ``tree`` is the whole configured tree when ``domains`` is only its
    resident part (a streamed root over a resident child), so a resident
    child's force slots are sized from its parent even when that parent
    holds no arena slot of its own.
    """
    from woof.core.preflight import (shared_scratch_arena_aliases,
                                      shared_scratch_arena_shapes)

    domain_tuple = tuple(domains)
    tree_tuple = None if tree is None else tuple(tree)
    return ScratchArena(
        shared_scratch_arena_shapes(domain_tuple, tree_tuple),
        slot_aliases=shared_scratch_arena_aliases(domain_tuple, tree_tuple))


def build_shared_dycore_state_workspace(
        domains: Iterable[object]) -> SharedDycoreStateWorkspace:
    """Allocate one maximum backing for each restart-REBUILT state symbol."""
    from woof.core.preflight import shared_dycore_state_workspace_shapes

    domain_tuple = tuple(domains)
    return SharedDycoreStateWorkspace(
        shared_dycore_state_workspace_shapes(domain_tuple))


def refresh_model_time(state, clock, *, kernel_launch: bool = False,
                       after_step: bool = False) -> None:
    """Refresh the legacy state mirror from the integer DomainClock.

    Calendar authority remains ``clock.ticks``.  At solve entry WRF-facing
    consumers receive the REAL ``curr_secs`` image; after solve the public
    compatibility mirror is refreshed from the exact next tick.  Keeping the
    assignment outside ``model.py`` also leaves T9's executor AST audit
    mechanically strict: the schedule walker never assigns an elapsed value.
    """
    if kernel_launch and after_step:
        raise ValueError("kernel_launch and after_step are mutually exclusive")
    if kernel_launch:
        value = float(clock.elapsed_seconds_fp32)
    else:
        ticks = clock.ticks + (clock.step_ticks if after_step else 0)
        value = ticks / clock.tick_den
    from woof.core.sfire_clock import bind_state_clock
    bind_state_clock(state,value)
    state.elapsed_seconds = value
    # The domain's own ACTIVATION EPOCH, from the same tick authority.
    # A physics driver counts WRF's ITIMESTEP, and ITIMESTEP is 1 on a
    # domain's first step -- not on the experiment's.  Deriving it from
    # absolute model time alone is right only for a domain that started
    # with the run; for one that activates later it names some arbitrary
    # step in the middle of a cadence, which is how a newborn nest
    # skipped the radiation call its land surface then needed.
    state.domain_start_offset = clock.spec.start_ticks / clock.tick_den


def _height_half_from_phb(phb: np.ndarray) -> np.ndarray:
    """Return the host half-level heights using the established FP64 tree."""
    return 0.5 * (phb[:-1] + phb[1:]) / c.G


def _copy_setup_float32(target, source, xp, native_host=False):
    """Fill a large explicit host state without a second cast array."""
    if native_host and xp is np and target.size >= 16_384:
        from woof.ingest.host_arrays import copy_float32
        if copy_float32(target, source):
            return
    target[...] = xp.asarray(source, dtype=np.float32)


def _array_module_for(value):
    """Return NumPy for host setup arrays, otherwise the CUDA array module."""
    if isinstance(value, np.ndarray):
        return np
    return _require_cupy()


def _state_array_module(state):
    """Honor only an explicitly requested host setup state.

    CPU-only unit tests historically monkeypatch this module's ``cp`` symbol
    to NumPy as a CUDA emulator.  Array-type inspection cannot distinguish
    that shim from a real host state, so the constructor records the explicit
    setup choice instead.
    """
    if getattr(state, "_host_setup_state", False):
        return np
    return _require_cupy()


def mu_at_u_faces(mu: cp.ndarray) -> cp.ndarray:
    """Column mass ``(ny, nx)`` averaged to u faces ``(ny, nx+1)``, periodic
    in x; face f lies between cells f-1 and f, face nx duplicates face 0.

    The single sanctioned face-averaging helper (Task 4 consolidated the
    copies that lived in dycore, advection, and diffusion).
    """
    xp = _array_module_for(mu)
    if (not isinstance(mu, np.ndarray) and xp is cp
            and mu.dtype == np.dtype(np.float32)
            and mu.ndim == 2 and mu.flags.c_contiguous
            and all(mu.shape)):
        return _mass_faces_device(mu, yface=False)
    mux = 0.5 * (mu + xp.roll(mu, 1, axis=1))
    return xp.concatenate([mux, mux[:, :1]], axis=1)


def mu_at_v_faces(mu: cp.ndarray) -> cp.ndarray:
    """Column mass ``(ny, nx)`` averaged to v faces ``(ny+1, nx)``, periodic
    in y; face f lies between rows f-1 and f, row ny duplicates row 0."""
    xp = _array_module_for(mu)
    if (not isinstance(mu, np.ndarray) and xp is cp
            and mu.dtype == np.dtype(np.float32)
            and mu.ndim == 2 and mu.flags.c_contiguous
            and all(mu.shape)):
        return _mass_faces_device(mu, yface=True)
    muy = 0.5 * (mu + xp.roll(mu, 1, axis=0))
    return xp.concatenate([muy, muy[:1, :]], axis=0)


def _mass_faces_device(mu, *, yface):
    """Preserve the separate FP32 sum and half-multiply on every face."""
    from woof.core.kernels import get_kernel
    ny, nx = mu.shape
    out = cp.empty((ny + int(yface), nx + int(not yface)), dtype=mu.dtype)
    kernel = get_kernel('face_mass', 'average_mass_faces')
    kernel(((out.size + 127) // 128,), (128,),
           (mu, out, np.int32(ny), np.int32(nx), np.int32(yface)))
    return out


class DomainState:
    """All prognostic, diagnostic, and reference arrays for one domain.

    ``array_module`` is an internal setup/export seam. The default remains
    CuPy; NumPy states are not valid inputs to the CUDA forecast integrator.
    """

    def __init__(self, cfg: RunConfig,
                 scratch_arena: ScratchArena | None = None,
                 dycore_state_workspace: SharedDycoreStateWorkspace | None =
                 None, *, array_module=None):
        nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
        if getattr(cfg, "umove", 0.0) or getattr(cfg, "vmove", 0.0):
            from woof.moving_frame import frame_velocity, validate_moving_frame
            validate_moving_frame(cfg)
            self._moving_frame_velocity = frame_velocity(cfg)
        xp = (_forecast_array_module() if array_module is None
              or (cp is not None and array_module is cp) else array_module)
        if xp is not np and xp is not cp:
            raise TypeError("array_module must be numpy or cupy")
        self._host_setup_state = array_module is np
        # Same definition as acoustic._spec_zone/_boundary_forced. Stored
        # geometry lets all health scans keep the interior scalar floor.
        self._health_spec_zone = (max(0, int(cfg.spec_zone))
                                  if cfg.specified or cfg.nested else 0)
        # Analysis-producer metadata only. Once coupled snapshots are made,
        # their immutable field inventory owns forcing through cache/restart.
        from woof.boundary_fields import external_scalar_fields
        self._external_scalar_boundary_fields = external_scalar_fields(cfg)
        if self._host_setup_state and (scratch_arena is not None
                                       or dycore_state_workspace is not None):
            raise ValueError(
                "NumPy setup states cannot use CUDA scratch/dycore workspaces")

        def zeros(*shape):
            return xp.zeros(shape, dtype=np.float32)

        def rebuilt(symbol, *shape):
            if dycore_state_workspace is None:
                return zeros(*shape)
            return dycore_state_workspace.view(symbol, shape, DTYPE)

        # Prognostic fields (perturbation form).
        self.u = zeros(nz, ny, nx + 1)
        self.v = zeros(nz, ny + 1, nx)
        self.w = zeros(nz + 1, ny, nx)
        self.thp = zeros(nz, ny, nx)        # theta' = theta - thb
        self.php = zeros(nz + 1, ny, nx)    # phi'
        self.mup = zeros(ny, nx)            # mu'

        # Diagnostic fields.
        self.p = zeros(nz, ny, nx)          # full pressure
        self.al = zeros(nz, ny, nx)         # alpha'_d
        self.alt = zeros(nz, ny, nx)        # total alpha_d
        # Allocate before restart restoration so its exact carrier is in
        # the expected inventory, including before the first diagnostic.
        self.p_perturbation = zeros(nz, ny, nx) if DIAGNOSTICS_ENABLED else None

        # Moisture mixing ratios (per unit dry mass) plus RK time-t copies.
        # The Phase-2 qv/qc/qr allocation is unchanged.  WSM6's ice mass
        # fields are allocated for mp=6/8; Thompson adds rain/ice number
        # moments for mp=8 and Morrison adds its four transported number
        # moments for mp=10.  Every
        # frozen dry/Kessler state retains its original storage and loops.
        if cfg.moist:
            self.qv = zeros(nz, ny, nx)
            self.qc = zeros(nz, ny, nx)
            self.qr = zeros(nz, ny, nx)
            self.qv0 = rebuilt("qv0", nz, ny, nx)
            self.qc0 = rebuilt("qc0", nz, ny, nx)
            self.qr0 = rebuilt("qr0", nz, ny, nx)
            # WRF h_diabatic (Registry.EM_COMMON:1389, "MICROPHYSICS LATENT
            # HEATING", K s-1): the previous step's microphysics theta
            # increment per second, retained by moist_physics_finish_em
            # (module_big_step_utilities_em.F:5745) and fed to every RK
            # step's theta tendency (rk_addtend_dry, module_em.F:1078-1079).
            # Zero at init exactly as WRF (start_em.F:643-644), so the
            # first step's dynamics see no heating (one-step lag).
            # RESTART ADVISORY: WRF carries h_diabatic in the restart
            # stream (the `r` in the Registry IO string `rdu`,
            # Registry.EM_COMMON:1389); a future woof restart
            # implementation must serialize this field: reconstructing it
            # is impossible and re-zeroing silently drops one step of
            # retained heating on the first resumed trajectory.
            self.h_diabatic = zeros(nz, ny, nx)
            # WRF RTHFTEN/RQVFTEN (Registry.EM_COMMON, the ``cu_physics``
            # forcing pair): the PURE ADVECTIVE theta and vapour rates the
            # dycore exports once per step for a cumulus scheme that reads
            # them.  Uncoupled, K s-1 and kg kg-1 s-1 -- the same units and
            # the same one-step lag as ``h_diabatic`` above, and written by
            # the same kind of producer (RK stage 1 of step N, consumed by
            # the physics call at the top of step N+1).
            #
            # Allocated only for the schemes that read them
            # (CUMULUS_ADVECTIVE_FORCING_SCHEMES), so a Kain-Fritsch or
            # cumulus-off run's device footprint is unchanged to the byte.
            # ``None`` elsewhere, exactly like the dry branch's h_diabatic:
            # the restart writer, the state hash and the health census all
            # skip on None, so no existing inventory moves.
            #
            # RESTART: SERIALIZED, not rebuilt.  Nothing between a resume
            # and the first post-resume cumulus call can refill them -- the
            # producer is a dycore stage that has not run yet -- so a
            # re-zeroed resume would hand the scheme one step of hard zeros
            # in the middle of a trajectory.  That is the h_diabatic
            # argument verbatim.
            if cfg.cu_physics in CUMULUS_ADVECTIVE_FORCING_SCHEMES:
                self.rthften = zeros(nz, ny, nx)
                self.rqvften = zeros(nz, ny, nx)
            else:
                self.rthften = self.rqvften = None
            if cfg.mp_physics == 5:
                # Ferrier-Aligo (Registry.EM_COMMON:3019): package
                # fer_mp_hires is moist:qv,qc,qr,qi (qi is the TOTAL ice,
                # cloud ice and precipitating ice together) + scalar:qt +
                # state:f_ice_phy,f_rain_phy,f_rimef_phy.  Like P3 it has
                # one ice mass and no qs/qg, and allocating those would hand
                # the model two frozen species the scheme never writes.
                # qt is rebuilt from qc+qr+qi on every FER_HIRES entry
                # (module_mp_fer_hires.F:388), so it is carried, never
                # transported (woof/core/fer_hires.py).  The three f_*
                # fields are WRF state arrays: not advected, carried from
                # step to step, and set by fer_hires_init on a non-restart
                # start to F_ICE = F_RAIN = 0 and F_RIMEF = 1 (:2912-2923).
                for name in ("qi", "qt", "f_ice", "f_rain", "f_rimef"):
                    setattr(self, name, zeros(nz, ny, nx))
                self.f_rimef[...] = DTYPE(1.0)
                self.qi0 = rebuilt("qi0", nz, ny, nx)
            elif cfg.mp_physics == 50:
                # P3 one-category (Registry.EM_COMMON:3038).  Deliberately
                # NOT folded into the tuple below: P3 has ONE ice category
                # and therefore no qs/qg/effs at all -- allocating them
                # would hand the rest of the model (advection, output,
                # nest transition, health) three frozen species P3 never
                # writes, which is the silent-zero-field failure mode the
                # mp=28 comment above warns about.  The registered package
                # is moist:qi + scalar:qni,qnr,qir,qib +
                # state:re_cloud,re_ice,th_old,qv_old; vmi3d/di3d/rhopo3d
                # are diagnostic-only and live in the adapter's scratch.
                for name in ("qi", "ni", "nr", "qir", "qib",
                             "effc", "effi"):
                    setattr(self, name, zeros(nz, ny, nx))
                # P3 needs the PREVIOUS step's theta and vapour to build its
                # supersaturation tendency ('A' term, module_mp_p3.F:3171).
                # p3_main writes them at the end of every call (:5018-5021),
                # so they are cross-step carriers, not RK-rebuilt copies.
                # WRF's th_old/qv_old are likewise plain state, zero at
                # allocation; the scheme's own max(t_old,1.) guard (:2329)
                # is what makes the zero first step well-defined.
                # RESTART ADVISORY: both belong in a restart stream (WRF
                # carries them); a resumed woof run that re-zeroes them
                # repeats the first-step transient once.
                self.th_old = zeros(nz, ny, nx)
                self.qv_old = zeros(nz, ny, nx)
                # module_mp_p3.F's own background radii, in woof's
                # radiation-facing micron convention: p3_main initializes
                # diag_effc = 10.e-6 m and diag_effi = 25.e-6 m every call
                # before any condensate test (:2279 and :2281; the :2280
                # between them is the commented-out diag_effr).  A PAIR, and
                # only a pair: the string "effs" does not occur anywhere in
                # module_mp_p3.F, and the driver's P3 call binds
                # diag_effc_3d=re_cloud and diag_effi_3d=re_ice with no snow
                # argument at all (module_microphysics_driver.F:1596-1597).
                self.effc[...] = DTYPE(10.0)
                self.effi[...] = DTYPE(25.0)
                for name in ("qi0", "ni0", "nr0", "qir0", "qib0"):
                    setattr(self, name, rebuilt(name, nz, ny, nx))
            elif cfg.mp_physics == 30:
                from woof.core.fast_sbm_contract import (BIN_SPECIES,EXTRA_SPECIES,
                                                         NUMBER_SPECIES)
                self.fast_sbm_bins=zeros(len(BIN_SPECIES),nz,ny,nx)
                for index,name in enumerate(BIN_SPECIES):
                    setattr(self,name,self.fast_sbm_bins[index])
                for name in ("qi","qs","qg",*NUMBER_SPECIES,"th_old","qv_old",
                             "effc","effi","effs"):
                    setattr(self,name,zeros(nz,ny,nx))
                for name in EXTRA_SPECIES:
                    setattr(self,name+"0",rebuilt(name+"0",nz,ny,nx))
                self.effc.fill(DTYPE(10.)); self.effi.fill(DTYPE(25.)); self.effs.fill(DTYPE(50.))
            elif cfg.mp_physics == 2:
                # Registry.EM_COMMON:3016 has six masses and no radius fields.
                for name in ("qi", "qs", "qg"):
                    setattr(self, name, zeros(nz, ny, nx))
                    setattr(self, name + "0", rebuilt(name + "0", nz, ny, nx))
            elif cfg.mp_physics == 3:
                # WSM3 (Registry.EM_COMMON:3017, wsm3scheme): moist:qv,qc,qr
                # and state:re_cloud,re_ice,re_snow.  Its cloud ice lives in
                # qc and its snow in qr below 0 C, so it allocates no
                # qi/qs/qg -- a zero ice field beside a live one would be
                # the silent-zero-field failure the mp=28 comment above
                # warns about -- but it does register the radius trio, and
                # module_physics_init.F:1004-1024 asks WSM3 for all three
                # under RRTMG.  Background radii are module_model_constants.F
                # RE_QC_BG/RE_QI_BG/RE_QS_BG (:62-64), the values
                # module_mp_wsm3.F:166-168 seeds before effectRad_wsm3, in
                # woof's radiation-facing micron convention.
                for name in ("effc", "effi", "effs"):
                    setattr(self, name, zeros(nz, ny, nx))
                self.effc[...] = DTYPE(2.49)
                self.effi[...] = DTYPE(4.99)
                self.effs[...] = DTYPE(9.99)
            elif cfg.mp_physics in COLUMN_SCHEMES:
                row = COLUMN_SCHEMES[cfg.mp_physics]
                for name in row.species:
                    if name not in ("qv", "qc", "qr"):
                        setattr(self, name, zeros(nz, ny, nx))
                        setattr(self, name + "0", rebuilt(name + "0", nz, ny, nx))
                for name, value in zip(("effc", "effi", "effs"), row.background_radii):
                    setattr(self, name, zeros(nz, ny, nx))
                    getattr(self, name)[...] = DTYPE(value)
            elif cfg.mp_physics in (6, 8, 9, 10, 16, 18, 24, 28, 106):
                common = ("qi", "qs", "qg", "effc", "effi", "effs")
                number_and_radius = (
                    ("nr", "ni") if cfg.mp_physics == 8 else
                    # Milbrandt-Yau (mp=9): hail mass beside graupel plus a
                    # number moment for all six hydrometeors.  The WRF
                    # driver binds qnc/qnr/qni/qns/qng/qnh
                    # (module_microphysics_driver.F:1857-1862) and every one
                    # is INOUT to mp_milbrandt2mom_driver.
                    (("qh", "nc", "nr", "ni", "ns", "ng", "nh")
                     if cfg.mp_physics == 9 else
                    (("nc", "nr", "ni", "ns", "ng", "effr")
                     if cfg.mp_physics == 10 else
                    # mp=16 (WDM6, Registry.EM_COMMON:3031) transports CCN,
                    # cloud droplet number and rain number as
                    # scalar:qnn,qnc,qnr beside WSM6's six masses.  The set
                    # is spelled in woof/core/wdm6.py so the allocator, the
                    # ring guard and the nest transition cannot drift apart.
                    (WDM6_NUMBER_SPECIES if cfg.mp_physics == 16 else
                    (("qh", "qndrop", "qnr", "qni", "qns", "qng",
                      "qnh", "qnn", "qvolg", "qvolh")
                     if cfg.mp_physics == 18 else
                    # mp=28 (Thompson aerosol-aware, Registry.EM_COMMON:3036)
                    # turns cloud droplet number into a prognostic scalar and
                    # adds the two aerosol number tracers.  The mp=8 tuple
                    # above is deliberately NOT shared: a shared tuple is the
                    # regression risk that would let a future mp=28 field
                    # silently appear on an mp=8 state.
                    (("nc", "nr", "ni", "nwfa", "nifa")
                     if cfg.mp_physics == 28 else ()))))))
                if cfg.mp_physics == 24:
                    # mp=24 (WSM7, Registry.EM_COMMON:3034
                    # moist:qv,qc,qr,qi,qs,qg,qh): WSM6's six masses plus
                    # the hail mass, and no number moment.
                    number_and_radius = ("qh",)
                for name in common + number_and_radius:
                    setattr(self, name, zeros(nz, ny, nx))
                if cfg.mp_physics == 16:
                    # module_mp_wdm6.F:220-227: WRF fills the WHOLE CCN
                    # memory window with the namelist ccn_conc on the first
                    # time step and never refills it, and ccn0 reaches
                    # nothing else in the module.  Doing it here instead is
                    # the NSSL qnn precedent and is exactly equivalent for a
                    # cold start; nc and nr correctly start at zero, which is
                    # what WRF's allocator leaves them at too.
                    self.nn[...] = DTYPE(cfg.wdm6_ccn_conc)
                if cfg.mp_physics == 18:
                    # WRF start_em.F initializes predicted NSSL CCN to
                    # nssl_cccn / 1.225 when no input field is present.
                    # Registry default nssl_cccn=0.5e9 m-3; the resulting
                    # dry-mass mixing ratio is exactly this FP32 value.
                    self.qnn[...] = DTYPE(408163264.0)
                # THE THREE ARMS BELOW ARE A PARTITION OF SIX-SPECIES STATES,
                # and mp=50 is deliberately in none of them.  What they decide
                # is not "which schemes declare radii" -- P3 declares two --
                # but "which writer's background TRIPLE (effc, effi, effs)
                # does a state that allocated all three receive before the
                # first microphysics call".  P3 is outside that question by
                # INVENTORY, decided rather than overlooked:
                #   * Registry.EM_COMMON:3038 gives mp=50 state:re_cloud,
                #     re_ice and NO re_snow, where wsm6scheme (:3021),
                #     thompson (:3024), wdm6scheme (:3031) and thompsonaero
                #     (:3036) -- this arm's four members -- each register the
                #     trio;
                #   * module_physics_init.F names the P3 family in the
                #     use_mp_re disjunction (:1017) and sets all three flags
                #     (:1021-1023), then immediately overrides has_reqs = 0
                #     for that family alone (:1027-1033).  WRF asks P3 for a
                #     cloud radius and an ice radius and never for a snow one.
                # So the mp=50 arm at the top of this block allocates effc and
                # effi and no effs.  Adding 50 to a tuple here is DEAD CODE
                # today -- P3 leaves this chain 50 lines above -- and an
                # AttributeError on ``self.effs`` the moment anyone also
                # folds P3 into the six-species elif, which the comment there
                # refuses for the same one-ice-category reason.  It would
                # also seed the state with a snow radius P3
                # never computes -- its single ice category spans rime
                # fraction instead of separating snow from graupel -- which
                # is the same invented number ``validate_p3_radiation``
                # (woof/config.py) already refuses to hand RRTMGP, decided
                # there with its WRF authority.  P3's own background pair is
                # seeded above from module_mp_p3.F, not from
                # module_model_constants.F.
                # Gate: tests/test_p3_port.py::
                # test_the_background_radius_rows_are_a_six_species_partition
                if cfg.mp_physics in (6, 8, 16, 24, 28, 106):
                    # module_model_constants.F WSM6/Thompson background radii,
                    # stored in woof's radiation-facing micron convention.
                    # (RE_QC_BG/RE_QI_BG/RE_QS_BG = 2.49E-6/4.99E-6/9.99E-6
                    # m, module_model_constants.F:62-64.)  mp=28 shares
                    # them: module_mp_thompson.F seeds re_qc1d/re_qi1d/
                    # re_qs1d from those same three parameters and applies
                    # the same MAX(RE_*_BG, MIN(...)) clamp for BOTH
                    # entries (:1466-1479 in mp_gt_driver, the single
                    # driver classic and aerosol-aware Thompson share), so
                    # the background a radiation call sees before the first
                    # microphysics step is identical.
                    # Public v4.1.21 module_physics_init.F:1010-1039 seeds
                    # 2.51/5.01/10.01 for mp=28 under the NOAA WRF 3.9 cloud
                    # optics.  Micron carriers, not meter inputs.  Chosen by
                    # value so this chain stays a membership ladder (gate
                    # named above).
                    noaa_wrf39 = (cfg.mp_physics == 28 and getattr(
                        cfg, "rrtmg_cloud_optics_form", "wrf_461") == "noaa_wrf39")
                    c_bg, i_bg, s_bg = ((2.51, 5.01, 10.01) if noaa_wrf39
                                        else (2.49, 4.99, 9.99))
                    self.effc[...] = DTYPE(c_bg)
                    self.effi[...] = DTYPE(i_bg)
                    self.effs[...] = DTYPE(s_bg)
                elif cfg.mp_physics in (9, 10):
                    self.effc[...] = DTYPE(2.5)
                    self.effi[...] = DTYPE(5.0)
                    self.effs[...] = DTYPE(10.0)
                    # mp=9 shares Morrison's row for the same WRF reason,
                    # not by resemblance: MILBRANDT2MOM is absent from the
                    # use_mp_re disjunction at module_physics_init.F:
                    # 1004-1023, so has_reqc/has_reqi/has_reqs are all 0 and
                    # WRF's radiation computes its own radii.  The scheme
                    # itself has the reff block COMMENTED OUT
                    # (module_mp_milbrandt2mom.F:3362/:3364/:3372/:3374), so
                    # nothing writes these after allocation; they exist for
                    # the state machinery and the spec-zone ring guard.
                    # NEITHER radiation arm reads them for mp=9:
                    # woof/core/rrtmg_legacy.py's _MP_DECLARES_RADII[9] =
                    # False keeps the legacy arm on its own WRF radii, and
                    # the RTE+RRTMGP arm's "milbrandt2" row derives the
                    # scheme's own radii from nc/ni/ns on every call and
                    # refuses these fields by name (woof/core/rrtmgp.py
                    # hydrometeor_paths).
                else:
                    # NSSL's native driver bounds are 2.51/10.01/25 um.
                    # State radii use woof's radiation-facing micron
                    # convention; the official-source CUDA diagnostic itself
                    # retains WRF's metre convention at its narrow boundary.
                    self.effc[...] = DTYPE(2.51)
                    self.effi[...] = DTYPE(10.01)
                    self.effs[...] = DTYPE(25.0)
                time_copies = ("qi0", "qs0", "qg0")
                if cfg.mp_physics == 8:
                    time_copies += ("nr0", "ni0")
                elif cfg.mp_physics == 9:
                    time_copies += (
                        "qh0", "nc0", "nr0", "ni0", "ns0", "ng0", "nh0")
                elif cfg.mp_physics == 10:
                    time_copies += ("nr0", "ni0", "ns0", "ng0")
                elif cfg.mp_physics == 24:
                    time_copies += ("qh0",)
                elif cfg.mp_physics == 16:
                    time_copies += tuple(
                        f"{name}0" for name in WDM6_NUMBER_SPECIES)
                elif cfg.mp_physics == 18:
                    time_copies += (
                        "qh0", "qndrop0", "qnr0", "qni0", "qns0",
                        "qng0", "qnh0", "qnn0", "qvolg0", "qvolh0")
                elif cfg.mp_physics == 28:
                    time_copies += ("nc0", "nr0", "ni0", "nwfa0", "nifa0")
                for name in time_copies:
                    setattr(self, name, rebuilt(name, nz, ny, nx))
                if cfg.mp_physics == 28:
                    # QNWFA2D / QNIFA2D: WRF's surface aerosol emission
                    # TENDENCIES in # kg-1 s-1 (Registry.EM_COMMON; the
                    # field was redefined from a concentration to a
                    # tendency on 13 May 2013, module_mp_thompson.F:
                    # 1313-1315).  They are INTENT(IN) to mp_gt_driver --
                    # microphysics reads them at :1310-1327 and never
                    # writes them -- so they are cross-step CONSTANTS, not
                    # RK-rebuilt copies, and must not go through
                    # ``rebuilt`` (a shared arena backing would let a
                    # sibling domain overwrite them between steps).
                    #
                    # Both start at exactly zero.  thompson_init derives
                    # nwfa2d from the synthetic CCN profile at :510, but
                    # nifa2d is not even a thompson_init dummy argument
                    # (:424-444 take nwfa2d/nbca2d only) and the whole
                    # file never assigns it.  A run with no WIF/dust
                    # ingest therefore keeps nifa2d == 0 for the entire
                    # forecast -- that is WRF's own behaviour, not an
                    # ArWen shortcut.
                    self.nwfa2d = zeros(ny, nx)
                    self.nifa2d = zeros(ny, nx)
            elif _named_scheme(cfg.mp_physics) is not None:
                # A NAMED scheme (woof.microphysics_schemes): its capability
                # row is the allocation table.  Every species gets storage
                # and an RK time copy; radii start at the Thompson-family
                # background; surface
                # emission fields are cross-step constants like mp=28's.
                named = _named_scheme(cfg.mp_physics)
                for name in (named.ice_mass_species + named.moment_species
                             + named.aerosol_species + ("effc", "effi", "effs")):
                    setattr(self, name, zeros(nz, ny, nx))
                self.effc[...] = DTYPE(2.49)
                self.effi[...] = DTYPE(4.99)
                self.effs[...] = DTYPE(9.99)
                for name in (named.ice_mass_species + named.moment_species
                             + named.aerosol_species):
                    setattr(self, name + "0", rebuilt(name + "0", nz, ny, nx))
                for name in named.surface_fields:
                    setattr(self, name, zeros(ny, nx))
        else:
            self.qv = self.qc = self.qr = None
            self.qv0 = self.qc0 = self.qr0 = None
            self.h_diabatic = None
            # A dry state runs no cumulus scheme (initialize_physics refuses
            # cu_physics without moisture), so the advective forcing pair
            # has no consumer here either.
            self.rthften = self.rqvften = None

        from woof.core.physics_inventory import PBL_ADVECTED_SCALARS
        for name, floor in PBL_ADVECTED_SCALARS.get(cfg.bl_pbl_physics, ()):
            setattr(self, name, zeros(nz, ny, nx))
            getattr(self, name).fill(DTYPE(floor))
            setattr(self, name + "0", rebuilt(name + "0", nz, ny, nx))

        # WRF two-time-level prognostic TKE (Registry.EM_COMMON:312,
        # ``state real tke ikj dyn_em 2 - r``), the km_opt=2 carrier.
        # Initial value is the allocated zero state exactly as WRF's ideal
        # path (no start_em writer; tke bootstraps from the surface terms
        # or the tke_seed).  RESTART: WRF carries tke in the restart stream
        # (the ``r`` IO flag) and so does woof -- ``tke`` is SERIALIZED and
        # ``tke0`` is REBUILT (written from tke at every dycore.step entry),
        # both classified in woof/io/restart.py.
        if cfg.bl_pbl_physics == 5 and cfg.bl_mynn_tkeadvect:
            # Registry.EM_COMMON:1118,3168: scalar qke_adv, not water mass.
            self.qke_adv = zeros(nz, ny, nx)
            self.qke_adv0 = rebuilt("qke_adv0", nz, ny, nx)
        if cfg.km_opt == 2:
            self.tke = zeros(nz, ny, nx)
            self.tke0 = rebuilt("tke0", nz, ny, nx)
        else:
            self.tke = self.tke0 = None
        # Prognostic/published subgrid turbulence energy.  The attribute
        # is ``e_sgs`` because ``self.e`` is already the WRF Coriolis
        # cosine parameter 2*Omega*cos(lat); the owning scheme's symbol
        # ``e``/``tke`` maps to this attribute everywhere (restart key
        # ``state/e_sgs``, preflight item ``e_sgs``).  Two owners, one
        # attribute: SASE (900) integrates it as its prognostic closure
        # energy, and Shin-Hong (11) publishes its own per-step TKE
        # diagnostic here -- the scheme's TKE chain IS its subgrid
        # energy, and the D1 gray-zone instrument reads state.e_sgs
        # whichever closure produced it.  Allocated ONLY when one of the
        # two is active, so every other configuration's object graph
        # stays byte-identical -- the attribute is ABSENT, not None,
        # matching the pattern the restart manifest walk expects.
        if cfg.bl_pbl_physics in (SASE_PBL_SCHEME, 11):
            self.e_sgs = zeros(nz, ny, nx)
            if cfg.bl_pbl_physics == SASE_PBL_SCHEME:
                # SASE cold start fills the realizability floor exactly
                # as the fused step clips.
                self.e_sgs.fill(DTYPE(SASE_E_MIN))
            else:
                # Shin-Hong cold start is WRF's own: shinhonginit fills
                # the TKE carrier at epsq2l/2 = 0.005 (the oracle
                # driver's documented cold-start value,
                # tools/shinhong_wrf461_oracle/run_bl_shinhong.F90:414).
                # Zero would be WRONG, not merely different: q2 = 0
                # makes mixlen's el collapse and prodq2's q2^1.5/disel
                # a 0/0, so the very first TKE-diagnostic call would
                # NaN -- WRF avoids that by never starting at zero.
                # A literal rather than an import because the authority
                # (woof/verify/shinhong_ref.py EPSQ2L) lives in the
                # verification tree this module must not depend on (the
                # sase_limits note above); the pair is gated equal in
                # tests/test_shinhong_runtime.py.
                self.e_sgs.fill(DTYPE(0.005))

        # RK stage copies (state at the start of the RK3 step).
        self.u0 = rebuilt("u0", nz, ny, nx + 1)
        self.v0 = rebuilt("v0", nz, ny + 1, nx)
        self.w0 = rebuilt("w0", nz + 1, ny, nx)
        self.thp0 = rebuilt("thp0", nz, ny, nx)
        self.php0 = rebuilt("php0", nz + 1, ny, nx)
        self.mup0 = rebuilt("mup0", ny, nx)

        # Slow-physics tendencies (coupled form).
        self.ru_t = rebuilt("ru_t", nz, ny, nx + 1)
        self.rv_t = rebuilt("rv_t", nz, ny + 1, nx)
        self.rw_t = rebuilt("rw_t", nz + 1, ny, nx)
        self.rth_t = rebuilt("rth_t", nz, ny, nx)
        self.rph_t = rebuilt("rph_t", nz + 1, ny, nx)
        self.rmu_t = rebuilt("rmu_t", ny, nx)

        # Acoustic-substep perturbation fields: deviations from the RK stage
        # reference state t* (ARW Tech Note sec. 3.1.2; Tasks 10-12).
        # u_pp/v_pp/w_pp are coupled momenta (mu*u)'' etc., th_pp is coupled
        # (mu*theta)'', ph_pp/mu_pp/p_pp are phi''/mu''/p''; ww_pp is the
        # perturbation eta mass flux Omega'' at w levels; p_pp_old keeps the
        # previous substep's p'' for divergence damping.
        self.u_pp = rebuilt("u_pp", nz, ny, nx + 1)
        self.v_pp = rebuilt("v_pp", nz, ny + 1, nx)
        self.w_pp = rebuilt("w_pp", nz + 1, ny, nx)
        self.th_pp = rebuilt("th_pp", nz, ny, nx)
        self.ph_pp = rebuilt("ph_pp", nz + 1, ny, nx)
        self.mu_pp = rebuilt("mu_pp", ny, nx)
        self.p_pp = rebuilt("p_pp", nz, ny, nx)
        self.p_pp_old = rebuilt("p_pp_old", nz, ny, nx)
        # advance_mu_th skips the forced outer column; sumflux still reads
        # its held Omega''. A child must never overwrite this domain's carrier.
        self.ww_pp = zeros(nz + 1, ny, nx)
        # Acoustic specific volume alpha'' (Task 4): diagnosed with p'' each
        # substep and consumed by advance_uv's alpha''*d(pb)/dx term, which
        # is nonzero on eta surfaces over terrain.
        self.al_pp = rebuilt("al_pp", nz, ny, nx)

        # Base-state profiles (filled by load_base).  Flat terrain
        # (cfg.terrain_opt == 0, Phase 1) keeps 1-D columns; with terrain the
        # base state is per-column, so the device fields are full 3-D.
        if cfg.terrain_opt == 0:
            self.thb = zeros(nz)
            self.pb = zeros(nz)
            self.alb = zeros(nz)
            self.phb = zeros(nz + 1)
            self.dphb_resid = zeros(nz)
        else:
            self.thb = zeros(nz, ny, nx)
            self.pb = zeros(nz, ny, nx)
            self.alb = zeros(nz, ny, nx)
            self.phb = zeros(nz + 1, ny, nx)
            self.dphb_resid = zeros(nz, ny, nx)
        self.mub = DTYPE(0.0)
        self.p_top = None

        # General-form plumbing (Phase 2 Task 3): kernels that consumed the
        # scalar mub take the (ny, nx) dry-mass field plus the hybrid
        # coefficient arrays c1h/c2h (half levels) and c1f/c2f (full levels)
        # instead: Task 3 wires the diagnostics, Task 4 the dynamics.  ht
        # is the terrain height (WRF HGT; zeros when flat).
        self.mub2d = zeros(ny, nx)
        self.ht = zeros(ny, nx)
        self.c1h = zeros(nz)
        self.c2h = zeros(nz)
        self.c1f = zeros(nz + 1)
        self.c2f = zeros(nz + 1)
        # Hybrid reference-pressure coefficients (WRF c3 = B(eta), c4 =
        # (eta - B)(p0 - pt)): consumed only by the hypsometric_opt=2 EOS
        # diagnostic (calc_p_alpha), which rebuilds the reference dry
        # pressures pfu/pfd/phm = c3*mu + c4 + p_top per column.
        self.c3h = zeros(nz)
        self.c4h = zeros(nz)
        self.c3f = zeros(nz + 1)
        self.c4f = zeros(nz + 1)
        # Full-level DROPS of the same pair, dc3f[k] = c3f[k] - c3f[k+1],
        # differenced once in float64 by load_base.  The opt-2 EOS needs
        # pfd - pfu, and adjacent c3f entries sit ~1/nz apart while each
        # carries half an ulp of 1, so redoing the subtraction on the
        # stored FP32 coefficients costs a factor 13 at nz=160 (measured
        # 6.5e-7 -> 8.3e-6 relative in p).
        self.dc3f = zeros(nz)
        self.dc4f = zeros(nz)

        # Map-scale factors at mass/u/v points and Coriolis parameters
        # f = 2*Omega*sin(lat), e = 2*Omega*cos(lat) at mass points (Phase 3
        # Task 3; WRF msftx==msfty etc. woof carries the single isotropic
        # factor per staggering, exact for Lambert/polar/Mercator).  sina /
        # cosa are the local map-rotation angle (geo_em SINALPHA/COSALPHA,
        # WRF Registry.EM_COMMON:1405-1406) consumed by the coriolis kernel's
        # rotation terms; the identity defaults (sina = 0, cosa = 1) are
        # WRF's unrotated setting (module_big_step_utilities_em.F:3703-3704).
        # The defaults (msf 1, f/e 0, identity rotation) keep every flux form
        # bitwise on the Phase 2 path; use set_map_coriolis() to change them
        # so the has_msf / rotational flags stay consistent.
        self.msft = xp.ones((ny, nx), dtype=np.float32)
        self.msfu = xp.ones((ny, nx + 1), dtype=np.float32)
        self.msfv = xp.ones((ny + 1, nx), dtype=np.float32)
        self.msfv_inv = xp.ones((ny + 1, nx), dtype=np.float32)
        self.f = zeros(ny, nx)
        self.e = zeros(ny, nx)
        self.sina = zeros(ny, nx)
        self.cosa = xp.ones((ny, nx), dtype=np.float32)
        #: any map factor != 1 (selects the msf-weighted flux forms).
        self.has_msf = False
        #: has_msf or any f/e != 0 (gates the Coriolis+curvature kernel).
        self.rotational = False

        # Phase 3 Task 8: optional setup-time lateral forcing and model
        # clock.  The object is deliberately untyped here to avoid making
        # the core state module import the ingest package.
        self.lateral_boundaries = None
        self.elapsed_seconds = 0.0

        # Phase 3 Task 12: an explicitly initialized PhysicsDriver.  Keeping
        # the default as None makes every pre-physics state allocation and
        # idealized run byte-for-byte unchanged; dycore.step only consults
        # this slot when a physics scheme is enabled in RunConfig.
        self.physics = None

        # Vertical-coordinate arrays (filled by load_base).
        self.dnw = zeros(nz)
        self.rdnw = zeros(nz)
        self.dn = zeros(nz)
        self.rdn = zeros(nz)
        self.fnp = zeros(nz)
        self.fnm = zeros(nz)
        self.znu = zeros(nz)
        self.znw = zeros(nz + 1)

        # Surface extrapolation weights for half-level fields (WRF cf1..cf3,
        # filled by load_base); the acoustic pressure gradient needs p'' at
        # the lowest full level.
        self.cf1 = DTYPE(0.0)
        self.cf2 = DTYPE(0.0)
        self.cf3 = DTYPE(0.0)
        # Model-top linear extrapolation weights for full-level fields
        # (WRF cfn/cfn1, module_initialize_real.F:3754-3755).
        self.cfn = DTYPE(0.0)
        self.cfn1 = DTYPE(0.0)

        # Scratch-buffer pool (see scratch()) and host base geopotential
        # (None until load_base runs: height_half() raises on the sentinel).
        self._scratch: dict[str, cp.ndarray] = {}
        # Keep the default object graph exactly as before: single-domain and
        # frozen the reference case paths have no _scratch_arena attribute at all. Only
        # Task-14's multi-domain builder injects this optional infrastructure.
        if scratch_arena is not None:
            self._scratch_arena = scratch_arena
        self._phb_host: np.ndarray | None = None
        self._dz_min: float | None = None

        if cfg.nwp_diagnostics == 1 or cfg.gsd_diagnostics == 1:
            # WRF UP_HELI_MAX (Registry.EM_COMMON:2083, IO "rh02"): allocate
            # the serialized running-max accumulator eagerly so restart
            # manifests and wrfout frame schemas are deterministic from the
            # first step (woof/core/uh_diag.py owns the update/reset).
            self.scratch((ny, nx), "up_heli_max")
        if cfg.nwp_diagnostics == 1:
            # The two consumer-owned tracking windows, allocated on the
            # same gate and for the same reason: deterministic manifests
            # from the first step.  They never reach a wrfout frame;
            # woof/core/uh_diag.py folds them beside UP_HELI_MAX and
            # their consumers reset them.  Their restart class is CARRIED
            # (woof/io/restart.py:CARRIED_SCRATCH_SLOTS): an ordinary
            # checkpoint writes neither, and a nest-lifecycle run opts
            # its own in per member so a resumed leg boundary reads the
            # window the run actually folded.
            self.scratch((ny, nx), "uh_follow_window")
            self.scratch((ny, nx), "uh_spawn_window")

        from woof.core.output_diagnostics import initialize as initialize_output
        initialize_output(self, cfg)

        from woof.core.hail import initialize as initialize_hail
        initialize_hail(self, cfg)

        # Chem species (woof/core/chem_state.py): the ``chem_<row>`` /
        # ``chem0_<row>`` arena views and ``self.chem``, allocated ONLY when
        # the run names chem sets.  A chem-off state gains no attribute, so
        # its object graph, restart inventory and footprint are unchanged.
        if getattr(cfg, "chem_sets", ()):
            from woof.core.chem_state import attach_chem_state
            attach_chem_state(self, cfg, xp)

    def load_base(self, coord: VerticalCoord, base: BaseState, *, native_host=False) -> None:
        """Copy the float64 setup-time coordinate/base arrays to device FP32."""
        if getattr(self, "_moving_frame_velocity", None) and base.terrain_z is not None:
            raise ValueError("umove/vmove with terrain needs ground heights translated each step")
        xp = _state_array_module(self)
        for name in ("dnw", "rdnw", "dn", "rdn", "fnp", "fnm", "znu", "znw",
                     "c1h", "c2h", "c1f", "c2f", "c3h", "c4h", "c3f", "c4f"):
            getattr(self, name)[...] = xp.asarray(getattr(coord, name),
                                                  dtype=np.float32)
        for name in ("thb", "pb", "alb"):
            dev = getattr(self, name)
            host = np.asarray(getattr(base, name), dtype=np.float64)
            if host.ndim != dev.ndim:
                raise ValueError(
                    f"base state {name} is {host.ndim}-D but the state was "
                    f"allocated for {dev.ndim}-D profiles: cfg.terrain_opt "
                    "must match the terrain_z the base state was built with")
            _copy_setup_float32(dev, host, xp, native_host)
        # Full-level coefficient DROPS, differenced once in float64 from
        # the coord's own values (see the dc3f allocation).  p_top cancels
        # out of pfd - pfu identically, so the pair needs no finalization.
        c3f64 = np.asarray(coord.c3f, dtype=np.float64)
        c4f64 = np.asarray(coord.c4f, dtype=np.float64)
        self.dc3f[...] = xp.asarray(c3f64[:-1] - c3f64[1:], dtype=np.float32)
        self.dc4f[...] = xp.asarray(c4f64[:-1] - c4f64[1:], dtype=np.float32)
        self.p_top = DTYPE(base.p_top)
        if np.ndim(base.mub) == 0:
            self.mub = DTYPE(base.mub)
            self.mub2d[...] = self.mub
        else:
            # Terrain: the (ny, nx) field is the only valid dry mass.  The
            # scalar is retired (None) so any consumer not yet wired for
            # terrain (Task 4) fails loudly instead of computing garbage.
            self.mub = None
            _copy_setup_float32(self.mub2d, base.mub, xp, native_host)
        if base.terrain_z is None:
            self.ht[...] = 0.0
        else:
            _copy_setup_float32(self.ht, base.terrain_z, xp, native_host)
        self.set_base_geopotential(base.phb, native_host=native_host)

        # WRF surface extrapolation weights (dyn_em module_initialize):
        # quadratic-in-eta extrapolation of half-level fields to znw[0].
        if coord.dnw.size >= 3:
            dn, dnw, fnp, fnm = coord.dn, coord.dnw, coord.fnp, coord.fnm
            cof1 = (2.0 * dn[1] + dn[2]) / (dn[1] + dn[2]) * dnw[0] / dn[1]
            cof2 = dn[1] / (dn[1] + dn[2]) * dnw[0] / dn[2]
            self.cf1 = DTYPE(fnp[1] + cof1)
            self.cf2 = DTYPE(fnm[1] - cof1 - cof2)
            self.cf3 = DTYPE(cof2)
        if coord.dnw.size >= 1:
            self.cfn = DTYPE(1.0 + coord.fnp[-1])
            self.cfn1 = DTYPE(-coord.fnp[-1])

    def set_base_geopotential(self, phb, *, native_host=False) -> None:
        """Install the base geopotential and everything derived from it.

        The sanctioned writer for ``phb``.  Besides the FP32 device copy
        it refreshes the two caches that are functions of it: the float64
        host snapshot behind ``height_half()``/``_dz_min``, and
        ``dphb_resid``.

        ``dphb_resid[k]`` is the float64 base layer thickness MINUS the
        float32 subtraction ``phb[k+1] - phb[k]`` the EOS kernel performs
        on the stored profile.  The kernel adds it back, recovering the
        thickness to ulp(dphb) instead of ulp(phb): over a 2400 m column
        at nz=64 that is ~368 J/kg reconstructed from two ~2.4e4 J/kg
        numbers, so FP32 storage alone costs 5e-6 relative and the cost
        grows as 1/dz (measured 2.1e-6 at nz=16, 2.9e-5 at nz=160 in p).

        The residual spelling, rather than a stored absolute thickness,
        is deliberate: it is what makes this cache SAFE to be stale.  A
        caller that assigns ``state.phb[...]`` directly leaves the
        correction describing the previous profile, and the kernel then
        adds a <=1-ulp-of-phb number to the correct FP32 difference of
        the profile it was actually handed -- the pre-fix answer, never a
        wrong one.  A stored thickness would instead diagnose the OLD
        column's alt, and so a wrong p, pressure-gradient force and
        acoustic sound speed.  ``height_half()`` has no such protection,
        which is the other reason the in-tree writers come through here.
        """
        xp = _state_array_module(self)
        host = np.asarray(phb.get() if hasattr(phb, "get") else phb,
                          dtype=np.float64)
        if host.ndim != self.phb.ndim:
            raise ValueError(
                f"base state phb is {host.ndim}-D but the state was "
                f"allocated for {self.phb.ndim}-D profiles: cfg.terrain_opt "
                "must match the terrain_z the base state was built with")
        if host.shape != self.phb.shape:
            raise ValueError(
                f"base state phb has shape {host.shape}, the state was "
                f"allocated for {tuple(self.phb.shape)}: the grid this "
                "column profile was built on is not the grid the state "
                "holds, so nz or the horizontal extent disagree.  Writing "
                "it would either overrun the allocation or silently "
                "broadcast one column's geopotential across the domain, "
                "and dphb_resid would then describe a profile no cell "
                "has")
        if native_host and xp is np and host.size >= 16_384:
            from woof.ingest.host_arrays import geopotential_cache
            if geopotential_cache(self, host, c.G):
                return
        stored = np.asarray(host, dtype=np.float32)
        self.phb[...] = xp.asarray(stored)
        # np.diff on the float32 view is the kernel's own subtraction.
        self.dphb_resid[...] = xp.asarray(
            np.diff(host, axis=0) - np.diff(stored, axis=0).astype(np.float64),
            dtype=np.float32)
        # Own the host geopotential backing the invariant spacing cache.
        # BaseState is mutable, and np.asarray would alias an FP64 base.phb;
        # a later caller mutation could then change height_half() without
        # invalidating _dz_min.  The device load already has copy semantics,
        # so retain the same snapshot on host as well.
        self._phb_host = np.array(host, dtype=np.float64, copy=True)
        z_half = _height_half_from_phb(self._phb_host)
        self._dz_min = (float(np.diff(z_half, axis=0).min())
                        if z_half.shape[0] > 1 else None)

    def set_map_coriolis(self, msft=None, msfu=None, msfv=None,
                         f=None, e=None, sina=None, cosa=None, *, msfv_inv=None) -> None:
        """Fill the map factors / Coriolis parameters (float64 host inputs).

        The sanctioned setter: it refreshes the ``has_msf`` (any msf != 1)
        and ``rotational`` (has_msf or any f/e != 0) flags the dycore keys
        its msf-weighted paths and the Coriolis+curvature kernel on.
        Identity values (all-ones msf, all-zero f/e) leave both flags off,
        preserving the bitwise Phase 2 step.  ``sina``/``cosa`` are the
        local map-rotation angle (geo_em SINALPHA/COSALPHA); they only
        scale the e-Coriolis terms inside the kernel, so they do not enter
        the flags: a rotated frame with f = e = 0 exerts no force, exactly
        as in WRF.  Direct assignment to the arrays bypasses the flags:
        don't.
        """
        if getattr(self, "_moving_frame_velocity", None):
            for name, val, identity in (("msft", msft, 1), ("msfu", msfu, 1),
                                        ("msfv", msfv, 1), ("f", f, 0), ("e", e, 0)):
                if val is not None and bool((np.asarray(val) != identity).any()):
                    raise ValueError("umove/vmove with Coriolis or map curvature would change acceleration without a shifted geostrophic balance")
        xp = _state_array_module(self)
        for name, val in (("msft", msft), ("msfu", msfu), ("msfv", msfv),
                          ("f", f), ("e", e), ("sina", sina), ("cosa", cosa)):
            if val is None:
                continue
            dev = getattr(self, name)
            host = np.asarray(val, dtype=np.float64)
            if host.shape != dev.shape:
                raise ValueError(f"{name} must have shape {dev.shape}, "
                                 f"got {host.shape}")
            dev[...] = xp.asarray(host, dtype=np.float32)
        # WRF reads MF_VX_INV from wrfinput. It is an independent stored
        # word, not necessarily fl(1 / MAPFAC_VX): an input writer may have
        # formed its inverse before rounding the map factor to float32.
        if msfv_inv is not None:
            inverse = np.asarray(msfv_inv, dtype=np.float32)
            if inverse.shape != self.msfv.shape:
                raise ValueError(f"msfv_inv must have shape {self.msfv.shape}, got {inverse.shape}")
            self.msfv_inv[...] = xp.asarray(inverse)
        elif msfv is not None:
            self.msfv_inv[...] = np.float32(1.0) / self.msfv
        self.has_msf = bool((self.msft != 1.0).any()
                            or (self.msfu != 1.0).any()
                            or (self.msfv != 1.0).any()
                            or (self.msfv_inv != 1.0).any())
        self.rotational = bool(self.has_msf or (self.f != 0.0).any()
                               or (self.e != 0.0).any())

    def scratch(self, shape, slot: str, dtype=None) -> cp.ndarray:
        """Persistent named scratch buffer; the only sanctioned extra device
        allocation.  A slot keeps the shape it was first requested with.

        Without an injected arena this is the original per-state zero-allocation
        path. With an arena, only registry-audited slots present in that arena
        draw a view; carrying/unproven slots still allocate per state.
        """
        shape = tuple(shape) if isinstance(shape, (tuple, list)) else (shape,)
        requested_dtype = np.dtype(np.float32 if dtype is None else dtype)
        buf = self._scratch.get(slot)
        if buf is None:
            arena = getattr(self, "_scratch_arena", None)
            if arena is not None and arena.has_slot(slot):
                buf = arena.view(shape, slot, requested_dtype)
            else:
                xp = _state_array_module(self)
                buf = xp.zeros(shape, dtype=requested_dtype)
            self._scratch[slot] = buf
        elif buf.shape != shape:
            raise ValueError(f"scratch slot {slot!r} has shape {buf.shape}, "
                             f"requested {shape}")
        elif buf.dtype != requested_dtype:
            raise ValueError(f"scratch slot {slot!r} has dtype {buf.dtype}, "
                             f"requested {requested_dtype}")
        return buf

    def existing_scratch(self, slot: str) -> cp.ndarray | None:
        """Return the named scratch buffer only if it already exists.

        Never allocates: lets the owner of a persistent slot family (the
        microphysics ring guard snapshotting its own ``mp_*`` accumulators)
        inspect a slot without creating unused buffers for schemes that
        never write it.
        """
        return self._scratch.get(slot)

    def total_theta(self) -> cp.ndarray:
        """Full potential temperature thb + theta' as a device array."""
        thb = self.thb
        return (thb if thb.ndim == 3 else thb[:, None, None]) + self.thp

    def total_mu(self) -> cp.ndarray:
        """Total dry column mass mub + mu' as a ``(ny, nx)`` device array
        (``mub2d`` is the scalar broadcast for flat terrain)."""
        return self.mub2d + self.mup

    def cell_area_weight(self) -> cp.ndarray:
        """Mass-point cell-area weight ``1/msft**2`` as ``(ny, nx)`` FP64.

        ARW carries the column-mass equation on the map plane: the
        acoustic mass update multiplies the layer divergence by
        ``msftx*msfty``, which for woof's isotropic factor is the
        ``m2 = msft*msft`` product formed in
        ``kernels/acoustic.cu advance_mu_th_msf``.  Dividing the column
        mass by that same product restores the physical cell area, so
        ``sum(total_mu * cell_area_weight)`` is the quantity whose
        tendency telescopes to the lateral boundary faces.  The reciprocal
        is taken in FP64 from the FP32 product the kernel itself forms, so
        the weight is the kernel's convention rather than a re-derivation;
        with identity map factors it is exactly 1.0 and the weighted sum
        is bit-identical to the unweighted one.
        """
        xp = _state_array_module(self)
        msft2 = self.msft * self.msft
        return 1.0 / msft2.astype(xp.float64)

    def height_half(self) -> np.ndarray:
        """Base-state half-level heights in metres, on host: ``(nz,)`` for a
        flat base state, per-column ``(nz, ny, nx)`` with terrain.

        Raises ``RuntimeError`` if :meth:`load_base` was never called (it
        used to silently return zeros: final-review carry-over T6).
        """
        if self._phb_host is None:
            raise RuntimeError(
                "height_half() called before load_base(): the base-state "
                "geopotential has not been loaded")
        return _height_half_from_phb(self._phb_host)

    @property
    def dz_min(self) -> float | None:
        """Cached minimum half-level spacing installed with the base state.

        ``None`` denotes a one-layer domain, whose CFL fallback remains the
        configured model-top height.  The pre-load error intentionally
        matches :meth:`height_half`, which formerly supplied this value to
        the integration health check.
        """
        if self._phb_host is None:
            raise RuntimeError(
                "height_half() called before load_base(): the base-state "
                "geopotential has not been loaded")
        return self._dz_min


def adopt_wrf_theta_carrier(state) -> None:
    """Strict mode: hold theta as WRF's ``t = theta - 300`` (``thb == 300``).

    Every strict kernel reads ``thp`` as WRF's ``t_2``: the EOS diagnosis
    (kernels/diagnostics.cu, ``theta = 300 + thp``), the specified/relaxed
    boundary coupling under GPUWM_WRF_EXACT_C_BIGSTEP (kernels/lbc_state.cu
    takes the coupled table as ``thp`` with no ``thb`` term) and the strict
    diffusion storage (kernels/smag2d.cu).  The wrfinput door supplies that
    carrier (woof/ingest/wrfinput.py sets ``thb = 300``); the prepared door
    loaded the preparation's base profile into ``thb`` (290 to 464 K on a
    12 km GFS case), so strict mode read theta up to 164 K low and the first
    steps went non-finite (``p`` after the first strict diagnosis, then
    microphysics RAINNC).  This re-expresses the loaded state in WRF's
    carrier: ``thp = (thb + thp) - 300`` in FP32, the word a default run
    writes as history ``T``, and ``thb = 300``.

    Outside strict mode, or when ``thb`` already is 300 everywhere (the
    wrfinput door), nothing changes.  The preparation's ``thb`` is kept on
    the host as ``_setup_thb`` so the setup fingerprint
    (woof/state_serialization_contract.py) still binds the prepared base
    the cache recorded, not the mode's re-expression of it.
    """
    from woof.wrf_exact import ENABLED
    if not ENABLED:
        return
    t0 = np.float32(c.T0)
    thb = state.thb
    if bool((thb == t0).all()):
        return
    state._setup_thb = np.array(thb.get() if hasattr(thb, "get") else thb,
                                dtype=np.float32, copy=True)
    full = thb if thb.ndim == 3 else thb[:, None, None]
    state.thp[...] = (full + state.thp) - t0
    thb[...] = t0


def _check_terrain_z(base: BaseState, terrain_z) -> None:
    """Cross-check an explicit terrain profile against the base state's.

    The base state (built by ``make_base_state``) is the terrain authority;
    an explicit ``terrain_z`` at init time is a call-site consistency check
    only, and any disagreement raises.
    """
    if terrain_z is None:
        return
    if base.terrain_z is None:
        raise ValueError(
            "terrain_z given but the base state is flat: build the base "
            "state with make_base_state(..., terrain_z=...) so its profiles "
            "carry the terrain")
    if not np.array_equal(np.asarray(terrain_z, dtype=np.float64),
                          np.asarray(base.terrain_z, dtype=np.float64)):
        raise ValueError(
            "terrain_z disagrees with the terrain the base state was built "
            "with (base.terrain_z)")


def init_at_rest(cfg: RunConfig, coord: VerticalCoord, base: BaseState,
                 terrain_z: np.ndarray | None = None) -> DomainState:
    """Allocate a state with zero perturbations on the given base state.

    ``terrain_z`` (optional) must match ``base.terrain_z``; ``ht`` is always
    filled from the base state, so flat call sites are unchanged.
    """
    _check_terrain_z(base, terrain_z)
    # Admitted before the constructor, so an idealized domain too big for
    # the card is refused by name rather than stopping in a CUDA
    # out-of-memory inside it.
    from woof.core.resident_admission import admit_construction

    _require_cupy()
    admit_construction("building this idealized state on the card", cfg)
    s = DomainState(cfg)
    s.load_base(coord, base)
    return s


def _rebalance_hydrostatic_terrain(th_total: np.ndarray, base: BaseState,
                                   coord: VerticalCoord) -> np.ndarray:
    """Per-column discrete hydrostatic recurrence over terrain, float64.

    The hybrid/terrain generalization of ``grid.rebalance_hydrostatic``:
    half-level dry pressure ``pd = c3h*mub + c4h + p_top`` per column,
    column-mass increments ``c1h*mub + c2h``, and the surface pinned at
    ``g*terrain_z``.  Expressions are ordered exactly as in
    ``make_base_state`` so that ``th_total == thb`` reproduces ``phb``
    bitwise and phi' stays identically zero (the base state already
    carries the terrain).
    """
    nz, ny, nx = th_total.shape
    mub = np.asarray(base.mub, dtype=np.float64)
    p = (coord.c3h[:, None, None] * mub[None]
         + coord.c4h[:, None, None] + base.p_top)
    alpha = c.RD * th_total * (p / c.P0) ** c.RCP / p

    ph = np.zeros((nz + 1, ny, nx))
    ph[0] = c.G * base.terrain_z
    for k in range(nz):
        ph[k + 1] = ph[k] - coord.dnw[k] * (coord.c1h[k] * mub
                                            + coord.c2h[k]) * alpha[k]
    return ph


def init_theta_perturbation(cfg: RunConfig, coord: VerticalCoord,
                            base: BaseState, thp_func,
                            terrain_z: np.ndarray | None = None
                            ) -> DomainState:
    """At-rest state plus a theta perturbation, hydrostatically rebalanced.

    ``thp_func(x, z) -> theta' (nz, ny, nx) numpy`` receives the domain-
    centered cell-center x coordinates ``x (nx,)`` and the base-state
    half-level heights ``z``: ``(nz,)`` for a flat base state (the Phase 1
    contract, bitwise unchanged), per-column ``(nz, ny, nx)`` with terrain.
    ``phi'`` is set so every column is discretely balanced (identically
    zero for theta' = 0); ``mu' = 0``; winds stay zero: cases set them on
    the returned state.
    """
    s = init_at_rest(cfg, coord, base, terrain_z)

    x = (np.arange(cfg.nx) + 0.5) * cfg.dx - 0.5 * cfg.nx * cfg.dx
    z = s.height_half()
    thp = np.asarray(thp_func(x, z), dtype=np.float64)

    if base.terrain_z is None:
        th_total = base.thb[:, None, None] + thp
        ph_total = rebalance_hydrostatic(th_total, base.mub, coord,
                                         cfg.p_surf)
        php = ph_total - base.phb[:, None, None]
    else:
        th_total = base.thb + thp
        ph_total = _rebalance_hydrostatic_terrain(th_total, base, coord)
        php = ph_total - base.phb

    s.thp[...] = cp.asarray(thp, dtype=DTYPE)
    s.php[...] = cp.asarray(php, dtype=DTYPE)
    return s
