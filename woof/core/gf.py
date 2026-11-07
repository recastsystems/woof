"""Grell-Freitas cumulus adapter: the ``cu_physics=3`` engine callable.

With ``GPUWM_WRF_EXACT=1``, the driver's mass-flux normalization uses
WOOF's earlier float32 operation sequence to reproduce stock WRF's words.
Default arithmetic keeps the current routine. Neither arm reads a capture.

The scheme itself is ``woof/core/kernels/gf.cu`` -- the whole of WRF
v4.6.1's GFDRV per column, held at max_ulp 0 against the byte-frozen
oracle by ``tests/test_gf_gfdrv_cuda.py`` with ``fzu`` PINNED from that
capture.  ``fzu`` is pinned because gamma is a DELIBERATE DIVERGENCE from
WRF since 2.7.0 -- WOOF's is correctly rounded and glibc's, which
gfortran binds WRF's ``gamma()`` intrinsic to, is not
(docs/gf_gamma_known_delta.md).  THIS adapter does not pin: the shipped
forecast computes the correctly rounded value, the three override slots
stay 0, and the pin exists only for column-by-column comparison against a
WRF capture.  This module is only the seam: it packs the engine's column state into the
kernel's input layout, launches ``gf_gfdrv_stage`` over every column, and
returns the driver's Task-1 :class:`~woof.core.physics.CumulusResult`
(rates held until the next due call, RAINCV consumed once per due call --
GF has no NCA persistence; WRF recomputes it every STEPCU step, and the
importer pins ``cudt = 0`` for ``cu_physics=3`` so STEPCU is 1, exactly
WRF's usual GF configuration).

The kernel runs the SHIPPED identity: corrected k22 indexing
(``k22_wrf_faithful = 0``).  The WRF-faithful off-by-one is reachable only
through the parity suites, never from here; the measured ledger for the
difference is tests/test_gf_shallow_cuda.py::
test_the_corrected_k22_ledger_entry (three fixture cases move, all three
rejected either way, zero output words differ).

Recorded deviations of THIS seam (the kernel behind it is bitwise; these
are about what the engine can hand it today):

1. **Forcing tendencies** (CLOSED, task #243).  GFDRV's forced states
   (``tn``/``qo``) sum advective (RTHFTEN/RQVFTEN), radiative (RTHRATEN)
   and boundary-layer (RTHBLTEN/RQVBLTEN) forcing, and all four auxiliary
   lanes are read off the bound driver:
   ``gf_rthdynten``/``gf_rqvdynten`` (the integrator's advective rates --
   the ARW dycore exports them at RK stage 1 of every step into
   ``state.rthften``/``state.rqvften``, bound at driver construction; the
   MPAS driver computes its own exactly as native v8.4.1's
   mpas_atm_time_integration.F:6936 + :2789 construction) and
   ``gf_rthblten``/``gf_rqvblten`` (the PBL slot's raw rates, retained by
   ``PhysicsDriver._couple_pbl_slot`` under WRF's between-calls
   persistence contract -- YSU, MYJ, MYNN, Shin-Hong and SASE all fill
   the slot on the same terms, exactly as WRF's cumulus driver reads
   RTHBLTEN/RQVBLTEN without asking which scheme wrote them).  A lane
   whose driver attribute is ``None`` feeds zeros -- what a driverless
   adapter and the parity harnesses get.

   THE ADVECTIVE LANE IS PURE ADVECTION, deliberately.
   ``module_cumulus_driver.F:867`` pre-folds ``RTHRATEN + RTHBLTEN`` into
   ``RTHFTEN`` for G3SCHEME and NTIEDTKESCHEME only; GFSCHEME is not in
   that list because the kernel sums the three lanes itself at
   ``kernels/gf.cu:4428``.  Handing this adapter a pre-folded RTHFTEN
   would integrate the boundary layer and the sky twice.
2. **Convective momentum tendencies.**  The kernel computes GF's
   dudt/dvdt; :class:`CumulusResult` carries no momentum slots, so they
   are not yet coupled.  WRF couples them; MPAS-A v8.4.1 does NOT
   (its cu_grell_freitas call carries no rucuten/rvcuten), so for the
   MPAS seam this is native parity, not a gap.
3. **Vertical velocity** uses the lower face ``w[k]`` in strict and
   default arithmetic, as WRF's GF driver does. The closure expects this
   staggered value; averaging adjacent faces changes its forcing. This
   deliberately changes the previous adapter convention.
4. **dx** is per-column when the driver carries ``gf_dx_column``
   ([ny, nx], metres); the scalar ``cfg.dx`` otherwise.  WRF's own GFDRV
   takes dx(i,j), so the per-column feed is the WRF interface, not an
   extension.
5. **Active columns.**  When the driver carries ``gf_active_column``
   ([ny, nx], bool) the scheme runs only in the columns marked True, and
   every output of a column marked False is zero: no rate, no rain, no
   cloud top.  This is how a variable-resolution mesh runs a cumulus
   scheme in its coarse cells and none in its convection-allowing region
   (HRRR's own choice at 3 km), which a scalar switch cannot say.  The
   kernel maps one thread to one column and no column reads another, so
   an active column's outputs are bitwise those of the same column in an
   unmasked call; the inactive columns are never packed or launched, so a
   mesh whose fine region holds most of its cells pays only for the rest.
   The mask is static constructor input (like ``gf_dx_column``): its
   compacted index is built once per mask array, never per step.

Default ice routing folds the 258 K split back into RQCCUTEN when there is
no prognostic ice. Strict mode retains GFDRV's split even in that case,
because stock cumulus_driver passes RQICUTEN unconditionally and the later
F_QI gate leaves that cold slot unused (module_cumulus_driver.F:1314,
module_cu_gf_wrfdrv.F:822, module_physics_addtendc.F:194).
"""

from __future__ import annotations

import numpy as np

from woof.core.kf import _model_clock_dt

DTYPE = np.float32


def cumulus_active_column(mask, ncol: int, cumulus_scheme):
    """Validate a per-column cumulus mask for the column seams, host side.

    ``None`` is "every column" (the unmasked scheme, byte-unchanged).  A
    mask is refused unless it is one boolean per column and the selected
    scheme is Grell-Freitas, the only adapter that reads it: a mask handed
    to another scheme would be recorded and never applied.
    """
    if mask is None:
        return None
    values = np.asarray(mask)
    if values.dtype != np.bool_:
        raise ValueError(
            "cumulus_active_column must be boolean (True = the cumulus "
            f"scheme runs in that column), got dtype {values.dtype}")
    values = values.reshape(-1)
    if values.shape != (ncol,):
        raise ValueError(
            "cumulus_active_column must have one value per column "
            f"({ncol},), got {values.shape}")
    if cumulus_scheme != "gf":
        raise ValueError(
            "cumulus_active_column is read by the Grell-Freitas adapter "
            f"only; cumulus_scheme={cumulus_scheme!r} would ignore it.  "
            "Follow-up: the same carrier in woof.core.ntiedtke")
    return values


def _active_index(adapter, active, ny: int, nx: int):
    """The compacted column index of a static mask, cached per array."""
    import cupy as cp

    if active.shape != (ny, nx):
        raise ValueError(
            f"driver gf_active_column must be [ny, nx]={ny, nx}, "
            f"got {active.shape}")
    if active.dtype != cp.bool_:
        raise ValueError("driver gf_active_column must be boolean")
    cached = adapter._active_cache
    if cached is not None and cached[0] is active:
        return cached[1]
    index = cp.flatnonzero(active.reshape(ny * nx)).astype(cp.int64)
    adapter._active_cache = (active, index)
    return index


def _mask_output_support(value, cfg):
    """Stock GFDRV publishes only the four-cell nonperiodic interior."""
    from woof.wrf_exact import ENABLED
    if not ENABLED:
        return value
    forced = bool(getattr(cfg, "specified", False) or getattr(cfg, "nested", False))
    if forced or getattr(cfg, "open_x", False):
        value[..., :4] = DTYPE(0)
        value[..., -4:] = DTYPE(0)
    if forced or getattr(cfg, "open_y", False):
        value[..., :4, :] = DTYPE(0)
        value[..., -4:, :] = DTYPE(0)
    return value

#: gf.cu's compiled level bound; nz above this recompiles via the
#: integer-define tier.
_GF_KMAX_DEFAULT = 40

# gf.cu input/output enum orders (gf_gfdrv_stage).  Must match the kernel;
# tools/gf_wrf461_oracle/gf_field_lists.py carries the same lists for the
# parity suites.
_IN_LEV = ("u", "v", "w", "t", "qv", "p", "pi", "rho", "dz8w", "p8w",
           "rthften", "rqvften", "rthraten", "rthblten", "rqvblten")
#: gf.cu's DINS_ order.  The last three are the per-column ``fzu`` override
#: (``<= 0`` means "compute it"), which this adapter always leaves at 0 --
#: the shipped forecast uses ArWen's own correctly rounded gamma.  They exist
#: because gamma is a DELIBERATE DIVERGENCE from WRF
#: (docs/gf_gamma_known_delta.md) and a run being graded column-by-column
#: against a WRF capture pins them from that capture; the parity suites are
#: the only callers that ever set them.
_IN_SCA = ("ht", "hfx", "qfx", "xland", "dt", "dx",
           "fzu_up", "fzu_dn", "fzu_sh")
_OUT_LEV = ("rthcuten", "rqvcuten", "rqccuten", "rqicuten", "dudt", "dvdt",
            "gdc", "gdc2",
            "outt", "outq", "outqc", "outu", "outv",
            "outts", "outqs", "outqcs")
_OUT_SCA = ("raincv", "pratec", "htop", "hbot", "xmb_shallow", "pret",
            "prets", "cuten", "cutens")
_OUT_ISCA = ("ktop_deep", "k22_shallow", "kbcon_shallow", "ktop_shallow",
             "kbcon", "ktop")


def _gf_module(nz: int, *, spp_conv: int = 0):
    if spp_conv:
        from woof.core.spp_kernel_sources import load_spp_module
        return load_spp_module("gf", gf_kernel_capacity(nz))
    if nz <= _GF_KMAX_DEFAULT:
        from woof.core.kernels import load_module

        return load_module("gf")
    from woof.core.kernels import load_module_int_defines

    return load_module_int_defines("gf", (("GF_KMAX", int(nz)),))


def gf_kernel_capacity(nz: int) -> int:
    """The ``GF_KMAX`` the module is compiled at for this ``nz``."""
    return _GF_KMAX_DEFAULT if nz <= _GF_KMAX_DEFAULT else int(nz)


# ---------------------------------------------------------------------------
# The per-thread column workspace
# ---------------------------------------------------------------------------
# gf.cu used to keep GFDRV's column arrays in the per-thread local frame, and
# CUDA prices a local frame at the card's RESIDENT-THREAD CAPACITY -- one
# per-context backing store of (frame - 1024) * SMs * maxThreadsPerSM, taken
# at first launch and never returned.  MEASURED on test host A (RTX 5070 Ti, 70
# SMs x 1,536, sm_120, NVRTC 13.3): the 22,416 B frame took 2,196.0 MiB while
# the kernel only ever had 384 threads/SM in flight.  The arrays now live in
# a global workspace this adapter allocates, sized to the threads ACTUALLY in
# flight, and the columns are launched in tiles of that size.
#
# These three must match gf.cu's GFWS_SLOT_COUNT_COL / GFWS_SLOT_COUNT_DRV /
# GFWS_SLOTS; tests/test_gf_workspace.py re-derives them from the .cu source
# and fails if either side moves alone.
GFWS_SLOT_COUNT_COL = 90
GFWS_SLOT_COUNT_DRV = 36
GFWS_SLOTS = GFWS_SLOT_COUNT_COL + GFWS_SLOT_COUNT_DRV

#: Launch block, and the tile's granularity.  gf.cu indexes the workspace by
#: the global thread id within a tile, so a tile is always a whole number of
#: blocks.
GF_BLOCK = 64

#: Blocks per SM the tile is sized for.  MEASURED, not assumed.  At 100,000
#: columns on an RTX 5070 Ti (70 SMs) the unbounded driver's wall was 31.07
#: ms at 2 blocks/SM, 24.18 at 4, 24.33 at 6, 25.14 at 8 and 26.63 at 12, so
#: 4 was the plateau while the driver held 168 registers and at most six
#: 64-thread blocks were resident.  ``gf_gfdrv_stage`` now carries
#: ``__launch_bounds__(64, 8)`` (128 registers, the same 88 B local frame),
#: so eight blocks are resident and the tile keeps them all busy.  On an
#: 82-SM RTX PRO 4500 at 50,000 columns a GF call took 10.85 ms against
#: 11.53 with the unbounded driver at 4 blocks/SM (equal at 16,384), every
#: output word unchanged; the query below only ever lowers it.
GF_TILE_BLOCKS_PER_SM = 8


_NATIVE_RESULT_TYPE = None


def _native_result_type():
    """The receipt type for fresh arrays returned by the production adapter.

    Built on first use rather than at import, because the memory pricing in
    ``woof.core.preflight`` imports this module's tile constants on hosts
    with no GPU runtime, and ``woof.core.physics`` imports CuPy.
    """
    global _NATIVE_RESULT_TYPE
    if _NATIVE_RESULT_TYPE is None:
        from woof.core.physics import CumulusResult

        class _NativeGFCumulusResult(CumulusResult):
            """Receipt for fresh arrays returned by the exact adapter."""

            def __init__(self, *, owner, **values):
                super().__init__(**values)
                self._owner = owner

        _NATIVE_RESULT_TYPE = _NativeGFCumulusResult
    return _NATIVE_RESULT_TYPE


def __getattr__(name):
    if name == "_NativeGFCumulusResult":
        return _native_result_type()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def is_native_gf_result(result, owner, state, cfg) -> bool:
    """Admit owned outputs only when provenance and layout match GF."""
    import cupy as cp

    if (int(cfg.cu_physics) != 3 or type(owner) is not GrellFreitas
            or type(result) is not _native_result_type()
            or result._owner is not owner):
        return False
    if any(getattr(result, name) is not None for name in (
            "rqrcuten", "rqscuten", "rucuten", "rvcuten",
            "nca_seconds", "pratec")):
        return False
    if ((result.rqicuten is not None)
            != (getattr(state, "qi", None) is not None)):
        return False
    shape = tuple(state.p.shape)
    history = result.history_rates or {}
    if set(history) - {"rqicuten", "rucuten", "rvcuten"}:
        return False
    for value in history.values():
        if (type(value) is not cp.ndarray or value.shape != shape
                or value.dtype != DTYPE or not value.flags.c_contiguous):
            return False
    for name in ("rthcuten", "rqvcuten", "rqccuten", "rqicuten", "rainc"):
        value = getattr(result, name)
        if name == "rqicuten" and value is None:
            continue
        expected = shape[1:] if name == "rainc" else shape
        if (type(value) is not cp.ndarray or value.shape != expected
                or value.dtype != DTYPE or not value.flags.c_contiguous):
            return False
    return True


def validate_native_gf_result(result, state) -> None:
    """Check every native output in one pass with one ordered readback."""
    import cupy as cp

    from woof.core import health_ledger
    from woof.core.kernels import get_kernel

    # The existing scanner has six volume slots and two surface slots.
    # Inactive slots use valid pointers but are never loaded.
    placeholder = result.rthcuten
    history_ice = (result.history_rates or {}).get("rqicuten", result.rqicuten)
    history_u = (result.history_rates or {}).get("rucuten")
    history_v = (result.history_rates or {}).get("rvcuten")
    values = (result.rthcuten, result.rqvcuten, result.rqccuten,
              placeholder if history_ice is None else history_ice,
              placeholder if history_u is None else history_u,
              placeholder if history_v is None else history_v, result.rainc, result.rainc)
    active = (0x47 | (0x08 if history_ice is not None else 0)
              | (0x10 if history_u is not None else 0) | (0x20 if history_v is not None else 0))
    labels = ((0, "cumulus rthcuten"), (1, "cumulus rqvcuten"),
              (2, "cumulus rqccuten"), (3, "cumulus rqicuten"),
              (4, "cumulus history rucuten"), (5, "cumulus history rvcuten"),
              (6, "cumulus RAINC increment"))

    def describe(flags):
        for bit, label in labels:
            if flags & (1 << bit):
                raise FloatingPointError(
                    f"{label} contains a non-finite value"
                    + health_ledger.deferred_note())

    status = state.scratch((1,), "physics_validation_status").view(cp.uint32)
    status.fill(cp.uint32(0))
    count = int(placeholder.size)
    get_kernel("kf_validation", "kf_validate_outputs")(
        ((count + 255) // 256,), (256,),
        values + (np.uint32(active), status,
                  np.int64(count), np.int64(result.rainc.size)))
    flags = health_ledger.read_status(
        status, site="grell-freitas", describe=describe)
    for bit, label in labels:
        if flags & (1 << bit):
            raise FloatingPointError(f"{label} contains a non-finite value")


def gf_workspace_floats(nz: int, columns: int) -> int:
    """Workspace floats for ``columns`` columns in flight at this ``nz``.

    Rounded up to whole blocks: gf.cu interleaves the workspace by LANE
    within a block, the way CUDA lays local memory out across a warp, so the
    unit of allocation is one block's region, not one column's.
    """
    blocks = (int(columns) + GF_BLOCK - 1) // GF_BLOCK
    return blocks * GFWS_SLOTS * (gf_kernel_capacity(nz) + 9) * GF_BLOCK


def gf_tile_columns(fn, ncol: int) -> int:
    """Columns to keep in flight: enough to fill the card, and no more.

    The whole point of the workspace is that it is charged per thread IN
    FLIGHT where the local-memory backing store was charged per thread the
    card could ever hold.  Over-sizing the tile hands that 4x back.
    """
    import cupy as cp

    dev = cp.cuda.Device()
    sms = dev.attributes["MultiProcessorCount"]
    per_sm = GF_TILE_BLOCKS_PER_SM
    try:
        from cupy_backends.cuda.api import driver

        from woof.backend import active_blocks_per_multiprocessor
        resident = active_blocks_per_multiprocessor(
            fn.kernel.ptr, GF_BLOCK, 0)
        # Never launch more blocks than the card can hold resident: those
        # columns would wait while their workspace slots stayed allocated.
        per_sm = min(per_sm, int(resident))
    except Exception:                              # noqa: BLE001
        pass                                       # keep the measured value
    per_sm = max(1, int(per_sm))
    tile = sms * per_sm * GF_BLOCK
    return int(min(int(ncol), tile))


class GrellFreitas:
    """Stateless ``cu_physics=3`` callable for :class:`PhysicsDriver`.

    The driver remains the sole cadence authority.  Each due call returns
    fresh rates (Task-1 contract: no ``nca_seconds``) and the RAINCV
    increment the kernel computed as ``pratec * dt``.
    """

    def __init__(self):
        self._driver = None

    _active_cache = None

    def bind_driver(self, driver) -> None:
        """Optional driver hook: gives the adapter the held radiation rates.

        Mirrors the ``update_trigger_history`` optional-hook precedent: the
        driver calls it when present; a directly attached adapter without a
        driver simply runs with zero radiative forcing, which is also what
        a radiation-off configuration feeds.
        """
        self._driver = driver

    def release(self) -> int:
        """Drop the driver back-reference so a dropped driver can be freed.

        THE SAME CYCLE New Tiedtke had (0bea9490), with a different
        consequence.  PhysicsDriver holds ``cumulus_callable`` and the
        line above holds the driver back, so ``state.physics = None`` at a
        relocation frees neither and the whole driver graph stays resident
        -- refcounting cannot collect a cycle, and CPython's cyclic
        collector triggers on Python object counts rather than on the
        device bytes hanging off them.

        Unlike New Tiedtke this adapter is STATELESS: it caches no
        pipeline and owns no workspace, so there is nothing here to free
        and the return is 0.  What the cycle was pinning is the driver's
        own state, not the scheme's.  Measured across the receipt corpus,
        Grell-Freitas grew 35.04 MiB per relocation against New Tiedtke's
        61.20 and Kain-Fritsch's 1.28.

        Returns bytes freed BY THIS ADAPTER, which is accurately zero, so
        the relocation receipt's cumulus_workspace_bytes does not claim
        credit for memory the collector reclaims elsewhere.
        """
        self._driver = None
        return 0

    def __call__(self, *, atmosphere, fields, state, cfg,
                 spp_conv: int | None = None, pattern_spp_conv=None):
        import cupy as cp
        from woof.core.spp_kernel_sources import spp_flag

        nz, ny, nx = state.p.shape
        ncol = ny * nx
        flag = getattr(cfg, "spp_conv", 0) if spp_conv is None else spp_conv
        stochastic = spp_flag(flag, "spp_conv")
        pattern = None
        if stochastic:
            pattern = pattern_spp_conv
            if pattern is None and self._driver is not None:
                pattern = getattr(self._driver, "spp_patterns", {}).get("conv")
            if (not isinstance(pattern, cp.ndarray) or pattern.shape != (4, ny, nx)
                    or pattern.dtype != DTYPE or pattern.device.id != state.p.device.id):
                raise ValueError("GF SPP requires float32 pattern_spp_conv[4,ny,nx] on the state device")
            if not bool(cp.all(cp.isfinite(pattern)).item()):
                raise ValueError("GF SPP pattern must be finite")

        active = (None if self._driver is None
                  else getattr(self._driver, "gf_active_column", None))
        index = None if active is None else _active_index(self, active, ny, nx)
        # n: the columns packed and launched (all of them without a mask).
        n = ncol if index is None else int(index.size)

        def cols(a):
            # Assignment into lvin packs this view without a temporary copy.
            view = cp.asarray(a, dtype=DTYPE).reshape(nz, ncol).T
            return view if index is None else view[index]

        def flat(a):
            # One value per column, compacted like cols().
            view = cp.asarray(a).reshape(ncol)
            return view if index is None else view[index]

        # GFDRV consumes lower-face W, including in its closure. Keep the
        # same physical convention in strict and default arithmetic.
        w_lower_face = state.w[:-1]
        if self._driver is not None:
            total = getattr(self._driver, "radiation_theta_forcing", None)
            rthraten = (total() if total is not None else
                        self._driver.rthratenlw + self._driver.rthratensw)
        else:
            rthraten = cp.zeros((nz, ny, nx), dtype=DTYPE)
        zeros_lev = cp.zeros((n, nz), dtype=DTYPE)
        forcing_mass = None
        from woof.wrf_exact import ENABLED as WRF_EXACT

        def held_lane(name):
            """Driver-held auxiliary forcing lane, zeros when absent."""
            nonlocal forcing_mass
            lane = (None if self._driver is None
                    else getattr(self._driver, name, None))
            if lane is None:
                return zeros_lev
            if lane.shape != (nz, ny, nx):
                raise ValueError(
                    f"driver {name} must be [nz, ny, nx]={nz, ny, nx}, "
                    f"got {lane.shape}")
            # Moist-theta RTHFTEN is already dry and uncoupled: the
            # RK epilogue converts it before the microphysics adjustment.
            owned = {"gf_rthdynten": "rthften", "gf_rqvdynten": "rqvften"}
            if (WRF_EXACT and name in owned
                    and lane is getattr(state, owned[name], None)
                    and cfg.cu_physics == 3
                    and not (name == "gf_rthdynten" and cfg.use_theta_m)):
                if forcing_mass is None:
                    from woof.state_serialization_contract import carried_wrf_muts
                    mut = carried_wrf_muts(state)
                    if mut is None:
                        mut = state.total_mu()
                    forcing_mass = (state.c1h[:, None, None] * mut[None]
                                    + state.c2h[:, None, None])
                lane = lane / forcing_mass
            if (name == "gf_rthdynten" and getattr(state, "use_theta_m", 0)
                    and not (cfg.cu_physics == 3
                             and lane is getattr(state, "rthften", None))):
                from woof.core.theta_m_physics import dry_cumulus_theta_tendency
                vapour = getattr(self._driver, "gf_rqvdynten", None)
                if vapour is None:
                    vapour = cp.zeros_like(lane)
                elif (WRF_EXACT and cfg.cu_physics == 3
                      and vapour is getattr(state, "rqvften", None)):
                    if forcing_mass is None:
                        from woof.state_serialization_contract import WRF_MUTS_SLOT
                        mut = state.existing_scratch(WRF_MUTS_SLOT)
                        if mut is None:
                            mut = state.total_mu()
                        forcing_mass = (state.c1h[:, None, None] * mut[None]
                                        + state.c2h[:, None, None])
                    vapour = vapour / forcing_mass
                lane = lane.copy()
                dry_cumulus_theta_tendency(state, lane, vapour)
            return cols(lane)

        lvin = cp.empty((n, len(_IN_LEV), nz), dtype=DTYPE)
        lvin[:, _IN_LEV.index("u"), :] = cols(atmosphere["u"])
        lvin[:, _IN_LEV.index("v"), :] = cols(atmosphere["v"])
        lvin[:, _IN_LEV.index("w"), :] = cols(w_lower_face)
        lvin[:, _IN_LEV.index("t"), :] = cols(atmosphere["temperature"])
        lvin[:, _IN_LEV.index("qv"), :] = cols(atmosphere["qv"])
        lvin[:, _IN_LEV.index("p"), :] = cols(atmosphere["pressure"])
        lvin[:, _IN_LEV.index("pi"), :] = cols(atmosphere["exner"])
        lvin[:, _IN_LEV.index("rho"), :] = cols(atmosphere["rho"])
        lvin[:, _IN_LEV.index("dz8w"), :] = cols(atmosphere["dz"])
        lvin[:, _IN_LEV.index("p8w"), :] = cols(
            atmosphere["p_interface"][:-1])
        lvin[:, _IN_LEV.index("rthften"), :] = held_lane("gf_rthdynten")
        lvin[:, _IN_LEV.index("rqvften"), :] = held_lane("gf_rqvdynten")
        lvin[:, _IN_LEV.index("rthraten"), :] = cols(rthraten)
        lvin[:, _IN_LEV.index("rthblten"), :] = held_lane("gf_rthblten")
        lvin[:, _IN_LEV.index("rqvblten"), :] = held_lane("gf_rqvblten")

        # WRF hands the scheme its model DT (module_cumulus_driver.F:1028)
        # -- the outer clock when the case integrates internal substeps,
        # the same idiom as the KF adapter.
        clock_dt = DTYPE(_model_clock_dt(cfg))
        scin = cp.zeros((n, len(_IN_SCA) + (4 if stochastic else 0)), dtype=DTYPE)
        scin[:, 0] = flat(state.ht)
        scin[:, 1] = flat(fields["hfx"])
        scin[:, 2] = flat(fields["qfx"])
        scin[:, 3] = flat(fields["xland"])
        scin[:, 4] = clock_dt
        if stochastic:
            # Native GFDRV clips four closure channels to [-1,1]. The
            # enabled specialization routes them into native rand_clos.
            closure = pattern.reshape(4, ncol).T
            if index is not None:
                closure = closure[index]
            scin[:, len(_IN_SCA):] = cp.clip(closure, -1.0, 1.0)
        # slots 6/7/8 stay 0: fzu is COMPUTED, not pinned, on this path.
        dx_column = (None if self._driver is None
                     else getattr(self._driver, "gf_dx_column", None))
        if dx_column is None:
            scin[:, 5] = DTYPE(cfg.dx)
        else:
            if dx_column.shape != (ny, nx):
                raise ValueError(
                    f"driver gf_dx_column must be [ny, nx]={ny, nx}, "
                    f"got {dx_column.shape}")
            scin[:, 5] = flat(cp.asarray(dx_column, dtype=DTYPE))

        iin = cp.empty((n, 3), dtype=cp.int32)
        iin[:, 0] = flat(fields["kpbl"])
        iin[:, 1] = np.int32(int(cfg.ishallow))
        iin[:, 2] = np.int32(int(cfg.clos_choice))

        # GFDRV writes every output slot, so clearing these slabs is redundant
        # for the launched columns; an inactive column's outputs are zero.
        lev = cp.empty((n, len(_OUT_LEV), nz), dtype=DTYPE)
        sca = cp.empty((n, len(_OUT_SCA)), dtype=DTYPE)
        isc = cp.empty((n, len(_OUT_ISCA)), dtype=cp.int32)

        module = _gf_module(nz, spp_conv=int(stochastic))
        fn = module.get_function("gf_gfdrv_stage")
        block = GF_BLOCK
        # The column arrays live in a global workspace sized to the threads
        # in flight, not in the per-thread local frame (which CUDA prices at
        # the card's whole resident-thread capacity).  Columns therefore go
        # in tiles of `tile`, each tile launched with the SAME one-thread-
        # one-column mapping the kernel has always had -- the slices below
        # are contiguous views, so the kernel indexes them exactly as before.
        tile = gf_tile_columns(fn, max(n, 1))
        ws = cp.empty(gf_workspace_floats(nz, tile), dtype=DTYPE)
        for lo in range(0, n, tile):
            hi = min(lo + tile, n)
            fn((((hi - lo) + block - 1) // block,), (block,),
               (lvin[lo:hi], scin[lo:hi], iin[lo:hi],
                lev[lo:hi], sca[lo:hi], isc[lo:hi], ws,
                np.int32(0),      # SHIPPED default: corrected k22 indexing
                np.int32(hi - lo), np.int32(nz)))
        del ws
        if index is not None:
            # Scatter the launched columns back; every other column is zero.
            full_lev = cp.zeros((ncol, len(_OUT_LEV), nz), dtype=DTYPE)
            full_sca = cp.zeros((ncol, len(_OUT_SCA)), dtype=DTYPE)
            full_isc = cp.zeros((ncol, len(_OUT_ISCA)), dtype=cp.int32)
            full_lev[index] = lev
            full_sca[index] = sca
            full_isc[index] = isc
            lev, sca, isc = full_lev, full_sca, full_isc
        if cfg.lightning_option == 11:
            cloud_top = state.scratch((ny, nx), "ltng_ktop_deep", dtype=np.int32)
            cloud_top[...] = isc[:, _OUT_ISCA.index("ktop_deep")].reshape(ny, nx)

        def grid(j):
            # (ncol, nz) slice -> (nz, ny, nx)
            return _mask_output_support(cp.ascontiguousarray(
                lev[:, j, :].T.reshape(nz, ny, nx)), cfg)

        rthcuten = grid(_OUT_LEV.index("rthcuten"))
        rqvcuten = grid(_OUT_LEV.index("rqvcuten"))
        rqc = grid(_OUT_LEV.index("rqccuten"))
        rqi = grid(_OUT_LEV.index("rqicuten"))
        history_rates = {
            "rqicuten": rqi,
            "rucuten": grid(_OUT_LEV.index("dudt")),
            "rvcuten": grid(_OUT_LEV.index("dvdt")),
        }
        if getattr(state, "qi", None) is None:
            from woof.wrf_exact import ENABLED
            # Stock cumulus_driver passes RQICUTEN to GFDRV even without
            # prognostic ice. GFDRV splits cold condensate into that slot;
            # calculate_phy_tend and phy_cu_ten leave it unused under F_QI.
            if not ENABLED:
                rqc = cp.ascontiguousarray(rqc + rqi)
            rqi = None
        rainc = cp.ascontiguousarray(
            sca[:, _OUT_SCA.index("raincv")].reshape(ny, nx))
        _mask_output_support(rainc, cfg)

        if cfg.cu_rad_feedback:
            from woof.core.g3 import run_g3_diag
            from woof.core.physics_inventory import cumulus_state_layout
            for key, (field_shape, dtype) in cumulus_state_layout(cfg).items():
                if key not in fields:
                    fields[key] = cp.zeros(field_shape, dtype=dtype)
            averages = {key:fields[key] for key in
                ('gd_cloud_a','qc_cu','gd_cloud2_a','qi_cu','gd_cldfr','raincv_a','raincv_b')}
            averages['clock'] = fields['g3_diag_clock']
            run_g3_diag(averages, cloud=grid(_OUT_LEV.index('gdc')),
                ice=grid(_OUT_LEV.index('gdc2')), rain=rainc,
                temperature=atmosphere['temperature'], pressure=atmosphere['pressure'],
                qv=atmosphere['qv'], qc=atmosphere['qc'], qi=atmosphere['qi'],
                fqi=int(getattr(state,'qi',None) is not None),
                dt=float(_model_clock_dt(cfg)), curr_secs=float(state.elapsed_seconds),
                avg_minutes=float(cfg.convtrans_avglen_m), adaptive=cfg.use_adaptive_time_step)

        return _native_result_type()(
            owner=self,
            rthcuten=rthcuten, rqvcuten=rqvcuten, rqccuten=rqc,
            rqicuten=rqi, rainc=rainc, history_rates=history_rates,
            # GFDRV's native bounds include its .001 level marker.
            cloud_base=(sca[:, _OUT_SCA.index('hbot')].reshape(ny, nx)
                        if 99 in (cfg.ra_lw_physics, cfg.ra_sw_physics) else None),
            cloud_top=(sca[:, _OUT_SCA.index('htop')].reshape(ny, nx)
                       if 99 in (cfg.ra_lw_physics, cfg.ra_sw_physics) else None))


# ===========================================================================
# Grell-Devenyi ensemble, cu_physics = 93
# ===========================================================================
# The older Grell ensemble (WRF v4.6.1 phys/module_cu_gd.F, GRELLDRV) lives
# in this module beside Grell-Freitas because it is the same family -- the
# maxiens/maxens/maxens2/maxens3 ensemble, the same advective/radiative/PBL
# forcing lanes, the same cu_diag fields -- and a scheme is admitted by its
# registry row and the cumulus dispatch table, not by an adapter file of its
# own.  The kernel is woof/core/kernels/gd.cu, graded word for word against
# WRF's own compiled module_cu_gd.o by tests/test_gd_cu93.py.
#
# What this seam hands the kernel, against WRF's cumulus_driver call
# (module_cumulus_driver.F:1095-1130):
#   * W is grid%w_2, the STAGGERED vertical velocity: GRELLDRV reads w(i,k,j)
#     for k = kts..ktf, the face below mass level k (omeg = -g*rho*w).  So
#     this adapter passes state.w[:-1], not the mass-level average the GF
#     adapter uses (its recorded deviation 3 does not apply here).
#   * P and P8W are p_hyd and p_hyd_w: atmosphere["pressure"] and
#     atmosphere["p_interface"] are exactly those (physics.py phy_prep).
#   * RTHFTEN/RQVFTEN are the dycore's pure advective rates; GRELLDRV folds
#     RTHRATEN + RTHBLTEN (times pi) and RQVBLTEN in itself
#     (module_cu_gd.F:219-224), which the kernel does, so the export stays
#     pure advection exactly as for GF.
#   * GSW is the radiation-written net surface shortwave (cap_max is 25 hPa
#     where gsw < 1, i.e. at night); XLAND is 1 land / 2 water.
#   * The edge band: GRELLDRV writes no convection in the four outermost
#     mass columns of a non-periodic domain (module_cu_gd.F:181-194,
#     310-327).  The band is derived from the call's own (ny, nx).  On a
#     tile-streamed run that is the tile buffer, and it is still right: a
#     clamped edge tile's buffer edge IS the domain edge, and an interior
#     seam's band columns are halo, which the scatter replaces with the
#     neighbour's interior (the halo radius is far wider than 4).
#
# Cadence and rain follow WRF's driver: the scheme runs when the cumulus
# step is due (any cudt -- GD has no NCA and WRF lets cudt hold its rates),
# its tendencies and PRATEC persist between calls, and RAINC accumulates
# PRATEC*DT every step (module_physics_addtendc.F:2141).  That is the
# driver's persistent-PRATEC contract with an all-zero NCA: every column is
# eligible on every due call and no hold ever expires.

GD_BLOCK = 64
#: The no-FTZ route: NVRTC directly, without CuPy's appended -ftz=true, and
#: without FMA contraction.  The kernel spells every operation with an _rn
#: intrinsic anyway; this keeps subnormal intermediates (small moisture
#: tendencies, the closure ensemble's tails) as gfortran keeps them.
GD_MODULE_OPTIONS = ("-std=c++17", "-fmad=false", "--ftz=false")
GD_MODULE_KEY = "woof.core.kernels:gd"
#: gd.cu's GD_NSLOT and GD_ENS: level slots and ensemble words per column.
GD_NSLOT = 76
GD_ENS = 144
#: gd.cu's input/output enum orders.
GD_IN_LEV = ("u", "v", "w", "t", "qv", "p", "pi", "rho", "rthften",
             "rqvften", "rthraten", "rthblten", "rqvblten")
GD_APR = ("apr_gr", "apr_w", "apr_mc", "apr_st", "apr_as", "apr_capma",
          "apr_capme", "apr_capmi")
GD_IN_SCA = ("p8w1", "ht", "xland", "gsw", "dt", "htop", "hbot") + GD_APR
GD_OUT_LEV = ("rthcuten", "rqvcuten", "rqccuten", "rqicuten", "gd_cloud",
              "gd_cloud2")
GD_OUT_SCA = ("raincv", "pratec", "htop", "hbot") + GD_APR + ("pret",)
GD_OUT_ISCA = ("ktop_deep", "kbcon", "ktop", "ierr", "k22")
#: WRF's module_model_constants as the cumulus driver hands them to
#: GRELLDRV: XLV, CP = 7*R_D/2, G, R_V (all float32).
from woof.core.constants import CP as _GD_CP, G as _GD_G, RV as _GD_RV, XLV as _GD_XLV

GD_XLV = np.float32(_GD_XLV)
GD_CP = np.float32(_GD_CP)
GD_G = np.float32(_GD_G)
GD_RV = np.float32(_GD_RV)
#: Blocks per SM a tile is sized for (the GF measurement; same shape).
GD_TILE_BLOCKS_PER_SM = 8


def gd_module_source() -> str:
    """The exact source NVRTC compiles for gd.cu (preamble + math header)."""
    from woof.core.kernels import module_source

    return module_source("gd")


def _gd_module():
    """gd.cu through the shared loader's cached no-FTZ route."""
    from woof.core.kernels import load_module

    return load_module("gd")


def gd_workspace_floats(nz: int, columns: int) -> int:
    """Workspace floats for ``columns`` columns in flight at ``nz`` levels."""
    blocks = (int(columns) + GD_BLOCK - 1) // GD_BLOCK
    return blocks * GD_BLOCK * (GD_NSLOT * (int(nz) + 2) + 2 * GD_ENS)


def gd_workspace_bytes(nz: int, columns: int) -> int:
    return 4 * gd_workspace_floats(nz, columns)


def gd_tile_columns(ncol: int, sms: int) -> int:
    """Columns kept in flight: enough to fill the card, no more."""
    return int(min(int(ncol), int(sms) * GD_TILE_BLOCKS_PER_SM * GD_BLOCK))


def gd_band_flags(ny: int, nx: int, periodic_x: bool,
                  periodic_y: bool) -> np.ndarray:
    """GRELLDRV's write window per mass column, (ny, nx) int32.

    2 inside both bands, 1 inside the j band but in the i edge band (WRF
    writes ktop_deep and zero tendencies there), 0 in the j edge band (WRF
    writes nothing).  ibeg = ids+4, iend = ide-5 unless periodic
    (module_cu_gd.F:181-194): with ide = nx+1, 0-based columns 4..nx-5.
    """
    i = np.arange(int(nx))
    j = np.arange(int(ny))
    iin = (np.ones(nx, bool) if periodic_x
           else (i >= 4) & (i <= int(nx) - 5))
    jin = (np.ones(ny, bool) if periodic_y
           else (j >= 4) & (j <= int(ny) - 5))
    return np.where(jin[:, None], np.where(iin[None, :], 2, 1),
                    0).astype(np.int32)


def gd_launch(lvin, scin, iin, *, nz: int, wrf_closure_count: int = 1,
              ens=None):
    """Run gd_grelldrv over every column of packed inputs; returns outputs.

    ``lvin`` (ncol, 13, nz), ``scin`` (ncol, 15) float32 and ``iin``
    (ncol, 2) int32 device arrays in gd.cu's orders.  ``wrf_closure_count``
    = 1 reproduces WRF's closure-count arithmetic and is the forecast
    default. The corrected-count experiment is available only to the oracle
    through an explicit zero argument.
    """
    import cupy as cp

    ncol = int(lvin.shape[0])
    lev = cp.empty((ncol, len(GD_OUT_LEV), nz), dtype=DTYPE)
    sca = cp.empty((ncol, len(GD_OUT_SCA)), dtype=DTYPE)
    isc = cp.empty((ncol, len(GD_OUT_ISCA)), dtype=cp.int32)
    fn = _gd_module().get_function("gd_grelldrv")
    sms = cp.cuda.Device().attributes["MultiProcessorCount"]
    tile = max(1, gd_tile_columns(ncol, sms))
    ws = cp.empty(gd_workspace_floats(nz, tile), dtype=DTYPE)
    null = np.uintp(0)
    for lo in range(0, ncol, tile):
        hi = min(lo + tile, ncol)
        fn((((hi - lo) + GD_BLOCK - 1) // GD_BLOCK,), (GD_BLOCK,),
           (lvin[lo:hi], scin[lo:hi], iin[lo:hi], lev[lo:hi], sca[lo:hi],
            isc[lo:hi], null if ens is None else ens[lo:hi], ws,
            GD_XLV, GD_CP, GD_G, GD_RV, np.int32(wrf_closure_count),
            np.int32(hi - lo), np.int32(nz)))
    del ws
    return lev, sca, isc


class GrellDevenyi:
    """Stateless-per-call ``cu_physics=93`` callable for :class:`PhysicsDriver`.

    Returns the driver's persistent-PRATEC result (``pratec`` with an
    all-zero ``nca_seconds``), so held rates and RAINC accumulation follow
    WRF's cumulus driver at any cudt.  ``cu_diag`` carries the per-call
    cu_diag fields GRELLDRV writes -- GD_CLOUD, GD_CLOUD2, ktop_deep,
    HTOP/HBOT and the eight APR_* closure-family precipitation accumulators
    -- for the cu_diag consumers. All views belong to driver.fields, whose
    inventory covers restart and checkpoint. Diagnostics remain local to
    each nest, as in the WRF Registry.
    """

    def __init__(self):
        self._driver = None

    @property
    def cu_diag(self):
        fields = {} if self._driver is None else self._driver.fields
        if "gd_lev" not in fields:
            return {}
        return {
            "gd_cloud": fields["gd_lev"][4],
            "gd_cloud2": fields["gd_lev"][5],
            "htop": fields["gd_sfc"][2], "hbot": fields["gd_sfc"][3],
            "ktop_deep": fields["gd_isfc"][0],
            **{name: fields["gd_sfc"][4 + n] for n, name in enumerate(GD_APR)},
        }

    def bind_driver(self, driver) -> None:
        self._driver = driver

    def release(self) -> int:
        """Drop the driver back-reference (the GF cycle; nothing to free)."""
        self._driver = None
        return 0

    def __call__(self, *, atmosphere, fields, state, cfg, **_ignored):
        import cupy as cp

        from woof.core.physics import CumulusResult

        nz, ny, nx = state.p.shape
        ncol = ny * nx

        def cols(a):
            return cp.asarray(a, dtype=DTYPE).reshape(nz, ncol).T

        zeros_lev = cp.zeros((ncol, nz), dtype=DTYPE)

        def held_lane(name):
            lane = (None if self._driver is None
                    else getattr(self._driver, name, None))
            if lane is None:
                return zeros_lev
            if lane.shape != (nz, ny, nx):
                raise ValueError(
                    f"driver {name} must be [nz, ny, nx]={nz, ny, nx}, "
                    f"got {lane.shape}")
            return cols(lane)

        if self._driver is not None:
            total = getattr(self._driver, "radiation_theta_forcing", None)
            rthraten = (total() if total is not None else
                        self._driver.rthratenlw + self._driver.rthratensw)
        else:
            rthraten = cp.zeros((nz, ny, nx), dtype=DTYPE)
        lvin = cp.empty((ncol, len(GD_IN_LEV), nz), dtype=DTYPE)
        lvin[:, 0, :] = cols(atmosphere["u"])
        lvin[:, 1, :] = cols(atmosphere["v"])
        lvin[:, 2, :] = cols(state.w[:-1])
        lvin[:, 3, :] = cols(atmosphere["temperature"])
        lvin[:, 4, :] = cols(atmosphere["qv"])
        lvin[:, 5, :] = cols(atmosphere["pressure"])
        lvin[:, 6, :] = cols(atmosphere["exner"])
        lvin[:, 7, :] = cols(atmosphere["rho"])
        lvin[:, 8, :] = held_lane("gf_rthdynten")
        lvin[:, 9, :] = held_lane("gf_rqvdynten")
        lvin[:, 10, :] = cols(rthraten)
        lvin[:, 11, :] = held_lane("gf_rthblten")
        lvin[:, 12, :] = held_lane("gf_rqvblten")

        diag = self.cu_diag
        if not diag or diag["htop"].shape != (ny, nx):
            diag = {name: cp.zeros((ny, nx), dtype=DTYPE) for name in GD_APR}
            diag["htop"] = cp.zeros((ny, nx), dtype=DTYPE)
            diag["hbot"] = cp.zeros((ny, nx), dtype=DTYPE)
        clock_dt = DTYPE(_model_clock_dt(cfg))
        gsw = fields.get("gsw")
        scin = cp.empty((ncol, len(GD_IN_SCA)), dtype=DTYPE)
        scin[:, 0] = atmosphere["p_interface"][0].reshape(ncol)
        scin[:, 1] = state.ht.reshape(ncol)
        scin[:, 2] = fields["xland"].reshape(ncol)
        scin[:, 3] = (0.0 if gsw is None else gsw.reshape(ncol))
        scin[:, 4] = clock_dt
        scin[:, 5] = diag["htop"].reshape(ncol)
        scin[:, 6] = diag["hbot"].reshape(ncol)
        for n, name in enumerate(GD_APR):
            scin[:, 7 + n] = diag[name].reshape(ncol)
        iin = cp.empty((ncol, 2), dtype=cp.int32)
        iin[:, 0] = fields["gd_band"].reshape(ncol)
        has_qi = getattr(state, "qi", None) is not None
        iin[:, 1] = np.int32(1 if has_qi else 0)

        lev, sca, isc = gd_launch(lvin, scin, iin, nz=nz)

        def grid(j):
            return cp.ascontiguousarray(lev[:, j, :].T.reshape(nz, ny, nx))

        def surf(j, a=sca):
            return cp.ascontiguousarray(a[:, j].reshape(ny, nx))

        rqi = grid(3) if has_qi else None
        fields["gd_lev"][...] = lev.transpose(1, 2, 0).reshape(6, nz, ny, nx)
        fields["gd_sfc"][...] = sca.T.reshape(len(GD_OUT_SCA), ny, nx)
        fields["gd_isfc"][...] = isc.T.reshape(len(GD_OUT_ISCA), ny, nx)
        from woof.core.g3 import run_g3_diag
        averages = {key: fields[key] for key in
            ('gd_cloud_a', 'qc_cu', 'gd_cloud2_a', 'qi_cu', 'gd_cldfr',
             'raincv_a', 'raincv_b')}
        averages['clock'] = fields['g3_diag_clock']
        shape = (nz, ny, nx)
        run_g3_diag(averages,
            cloud=self.cu_diag['gd_cloud'], ice=self.cu_diag['gd_cloud2'],
            rain=surf(0), temperature=atmosphere['temperature'],
            pressure=atmosphere['pressure'], qv=atmosphere['qv'],
            qc=atmosphere['qc'],
            qi=atmosphere['qi'] if has_qi else cp.zeros(shape, dtype=DTYPE),
            fqi=int(has_qi), dt=float(_model_clock_dt(cfg)),
            curr_secs=float(state.elapsed_seconds),
            avg_minutes=float(cfg.convtrans_avglen_m),
            adaptive=cfg.use_adaptive_time_step)
        return CumulusResult(
            rthcuten=grid(0), rqvcuten=grid(1), rqccuten=grid(2),
            rqicuten=rqi, pratec=surf(1),
            nca_seconds=cp.zeros((ny, nx), dtype=DTYPE))


__all__ = ["GrellFreitas", "GrellDevenyi"]
