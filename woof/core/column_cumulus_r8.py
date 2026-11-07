"""Column cumulus schemes admitted by table: one launcher, one row per scheme.

A cumulus scheme whose driver uses column CUDA kernels -- WRF's
``CU_*`` column driver, the scheme core and the lookup tables it reads,
transcribed into one CUDA translation unit -- needs no adapter of its own.
Its row in :data:`COLUMN_CUMULUS_SCHEMES` says which kernel, which lookup
tables, how much binary64 workspace a column takes, which WRF tendencies it
forms and where its random input comes from; :class:`ColumnCumulus` runs any
such row.  ``cu_physics = 95`` (WRF's old GFS simplified Arakawa-Schubert,
``phys/module_cu_osas.F``) is the first row.

THE INTERFACE IS WRF's ``CU_*`` CALL.  The kernel receives what
``module_cumulus_driver.F`` hands the scheme (T, QV, QC, QI, U, V, W, P,
P8W, PI, RHO, DZ8W, XLAND, DT, STEPCU) and returns RTHCUTEN, RQVCUTEN,
RQCCUTEN, RQICUTEN, RAINCV, PRATEC, HTOP and HBOT exactly as the Fortran
forms them.  The engine's atmosphere dictionary already carries every input
under the names New Tiedtke reads.

THE RANDOM INPUT.  The SAS family picks its cloud top from a uniform draw
(``XKT2``).  WRF draws it from a clock-seeded ``random_number`` on every
call and reseeds per J row from the same clock, so WRF is not reproducible
and rows share their sequence.  The strict arm (``GPUWM_WRF_EXACT=1``)
takes the draw as an injected input -- a verification harness hands the
oracle's values to :meth:`ColumnCumulus.inject_random`, otherwise it is WRF's
own REGTEST constant 0.1 -- and the default arm draws it from a counter-based
generator keyed on the model step and the column, which is reproducible,
independent of the launch decomposition and carries no state across a
restart.  That is a declared divergence with that cause.

THE LOOKUP TABLES.  ``gfs_funcphys`` is ``module_gfs_funcphys.F``'s
``gfuncphys`` (saturation vapour pressure, dewpoint, theta-e, moist adiabat,
LCL), built on the card by the scheme's own table kernels with WOOF's own
binary64 exp/log/pow and checked word for word against the digest of the
WRF oracle's tables before any column reads them.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from decimal import Decimal, localcontext
from types import MappingProxyType

import numpy as np

from woof.physics_vertical_contract import (
    OSAS_VERTICAL_LEVEL_BOUNDS, outside_vertical_bounds, refuse_vertical_levels)

#: Kernel threads per block, the house shape for column kernels.
_TPB = 64

#: A null device pointer argument (an 8-byte zero in the pointer's slot).
_NULL = np.uint64(0)

#: Column workspace budget per launch; the domain is processed in chunks
#: whose binary64 pool fits it.
COLUMN_CUMULUS_WORKSPACE_BUDGET_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class ColumnCumulusScheme:
    """One cumulus scheme run by :class:`ColumnCumulus`.

    ``r8_per_level`` and ``r8_extra`` size the binary64 workspace a column
    takes (``r8_per_level * kx + r8_extra`` words); the kernel flags an
    overflow and the launcher refuses the result by name.  ``flags`` are the
    shipped identity's kernel flags; the parity suites alone pass others.
    """

    @property
    def kernel_module(self):
        return self.module

    cu_physics: int
    label: str
    module: str
    entry: str
    tables: str
    r8_per_level: int
    r8_extra: int
    random_input: str
    strict_random_value: float
    flags: int
    strict_flags: int
    #: First-call level-count window (woof.physics_vertical_contract) and
    #: the breakage a column outside it meets.
    vertical_bounds: tuple
    vertical_breakage: str
    #: WRF parameters the scheme fixes in its driver (documentation and
    #: receipts; the kernel holds the same values).
    wrf_parameters: MappingProxyType
    #: Driver preparation, column physics and level-parallel output entries.
    launch_stages: tuple[str, ...] = ()


#: The admitted column cumulus schemes, keyed by WRF ``cu_physics``.
COLUMN_CUMULUS_SCHEMES = MappingProxyType({
    95: ColumnCumulusScheme(
        cu_physics=95,
        label="old GFS simplified Arakawa-Schubert (OSAS)",
        module="cu_osas",
        entry="cu_osas_columns",
        tables="gfs_funcphys",
        launch_stages=("prepare", "physics", "finish"),
        # CU_OSAS's 14 driver arrays (PRSI has kx+1 levels) and OSASCNV's
        # 28 automatic arrays, which are live together; SHALCV's three
        # come after OSASCNV has released its own.
        r8_per_level=42,
        r8_extra=1,
        random_input="xkt2",
        # module_cu_osas.F:283-286, the #ifdef REGTEST value.
        strict_random_value=0.1,
        flags=0,
        strict_flags=1,
        vertical_bounds=OSAS_VERTICAL_LEVEL_BOUNDS,
        vertical_breakage=(
            "SHALCV lifts its parcel from level 2 of the column's lower "
            "KM-1 levels (module_cu_osas.F:2218, :2438), which do not "
            "exist below three levels; select at least three model levels."),
        wrf_parameters=MappingProxyType({
            "jcap": 126, "ncloud": 1, "dpshc_cb": 30.0,
            "mommix": "NMM only; EM forms no momentum tendency"}),
    ),
})

# ---- gfs_funcphys: module_gfs_funcphys.F's gfuncphys tables ------------------
_GFS_NX = {"pvs": 7501, "tdp": 5001, "the": (241, 151), "ma": (151, 121),
           "tlcl": (151, 61)}
#: Words in the table buffer (16 index coefficients, then tbpvs, tbtdp,
#: tbthe, tbtma, tbqma, tbtlcl in Fortran order) -- cu_osas.cu OSAS_TB_WORDS.
GFS_FUNCPHYS_WORDS = 16 + 7501 + 5001 + 241 * 151 + 2 * 151 * 121 + 151 * 61

#: SHA-256 of the table buffer's little-endian words as the WRF v4.6.1
#: oracle builds them (tools/osas_wrf461_oracle, run_osas.F90's table dump,
#: coefficients first).  A card whose tables differ by one word is refused:
#: every saturation value the scheme forms would move with it.
GFS_FUNCPHYS_SHA256 = (
    "46b826ff2e7519cd89149ca9c351bc8aa5819f5e1a1812dea1f306f5e6735a61")


def _cr_pow(x: float, y: float) -> float:
    """``x**y`` correctly rounded, as gfortran folds a constant real power.

    ``0.04_krealfp**con_rocp`` and its siblings in gthe/gtma are constant
    expressions gfortran evaluates at compile time with MPFR, so the words
    WRF's tables start from are correctly rounded.  Evaluated here in
    80-digit decimal and rounded once to binary64.
    """
    with localcontext() as ctx:
        ctx.prec = 80
        return float(Decimal(x) ** Decimal(y))


def _gfs_rocp() -> float:
    rd = float(np.float32(2.8705e+2))
    cp_ = float(np.float32(1.0046e+3))
    return rd / cp_


def gfs_funcphys_limits() -> dict:
    """The folded y-range words of gthe and gtma."""
    rocp = _gfs_rocp()
    return {"the": (_cr_pow(0.04, rocp), _cr_pow(1.10, rocp)),
            "ma": (_cr_pow(0.01, rocp), _cr_pow(1.10, rocp))}


def table_sha256(words) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(words, dtype="<f8")).tobytes()).hexdigest()


_TABLE_CACHE: dict = {}


def gfs_funcphys_tables(module_name: str):
    """Build (once per card) and verify the gfuncphys tables on the device."""
    import cupy as cp

    from woof.core.kernels import load_module_int_defines

    device = cp.cuda.Device().id
    key = (module_name, device)
    cached = _TABLE_CACHE.get(key)
    if cached is not None:
        return cached
    module = load_module_int_defines(module_name, (("OSAS_TABLE_BUILD", 1),))
    tb = cp.zeros(GFS_FUNCPHYS_WORDS, dtype=cp.float64)
    limits = gfs_funcphys_limits()
    n = 7501
    module.get_function("osas_gpvs")(((n + 127) // 128,), (128,), (tb,))
    module.get_function("osas_gtdp")((2,), (128,), (tb,))
    n = 241 * 151
    module.get_function("osas_gthe")(((n + 127) // 128,), (128,),
                                     (tb, np.float64(limits["the"][0]),
                                      np.float64(limits["the"][1])))
    module.get_function("osas_gtma")(((121 + 127) // 128,), (128,),
                                     (tb, np.float64(limits["ma"][0]),
                                      np.float64(limits["ma"][1])))
    n = 151 * 61
    module.get_function("osas_gtlcl")(((n + 127) // 128,), (128,), (tb,))
    digest = table_sha256(tb.get())
    if GFS_FUNCPHYS_SHA256 is not None and digest != GFS_FUNCPHYS_SHA256:
        raise RuntimeError(
            "the gfuncphys lookup tables built on this card hash to "
            f"{digest}, not the WRF v4.6.1 oracle's {GFS_FUNCPHYS_SHA256}; "
            "every saturation, dewpoint, theta-e and moist-adiabat value the "
            "scheme reads would differ from WRF's, so the scheme is refused "
            "on this card rather than run on moved tables")
    _TABLE_CACHE[key] = tb
    return tb


# ---- the counter-based random input ---------------------------------------------
#: Seed of the default arm's draw.  Fixed: a reproducible run is the point.
COLUMN_RANDOM_SEED = 0x6f736173


def random_input(ncol: int, *, step: int, module_name: str, strict_value=None,
                 column_indices=None):
    """The per-column draw: counter-based, or the strict arm's constant."""
    import cupy as cp

    from woof.core.kernels import load_module

    out = cp.empty(ncol, dtype=cp.float64)
    kernel = load_module(module_name).get_function("cu_osas_xkt2")
    mode = 0 if strict_value is None else 1
    fixed = 0.0 if strict_value is None else float(strict_value)
    kernel(((ncol + 255) // 256,), (256,),
           (out, np.int64(ncol), np.uint64(COLUMN_RANDOM_SEED), np.int64(step),
            np.int32(mode), np.float64(fixed),
            _NULL if column_indices is None else column_indices))
    return out


def _mix64(z):
    """splitmix64's finaliser on uint64 arrays (wrapping arithmetic)."""
    z = z + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def random_input_host(ncol: int, *, step: int) -> np.ndarray:
    """The default arm's draw computed on the host, word for word the
    device's cu_osas_xkt2 (receipts and the oracle's default-arm replay)."""
    with np.errstate(over="ignore"):
        seed = np.array([COLUMN_RANDOM_SEED], dtype=np.uint64)
        stepw = np.array([step], dtype=np.uint64)
        key = _mix64(seed ^ _mix64(stepw))
        h = _mix64(key ^ np.arange(ncol, dtype=np.uint64))
    return (h >> np.uint64(11)).astype(np.float64) * 2.0 ** -53


# ---- the launch ------------------------------------------------------------------
LEVEL_INPUTS = ("t", "qv", "qc", "qi", "u", "v", "p", "pi", "rho", "dz8w")
IFACE_INPUTS = ("w", "p8w")
LEVEL_OUTPUTS = ("rthcuten", "rqvcuten", "rqccuten", "rqicuten")
SURFACE_OUTPUTS = ("raincv", "pratec", "htop", "hbot")


def chunk_columns(scheme: ColumnCumulusScheme, kx: int, ncol: int,
                  budget_bytes: int = COLUMN_CUMULUS_WORKSPACE_BUDGET_BYTES) -> int:
    per_column = 8 * (scheme.r8_per_level * kx + scheme.r8_extra)
    return max(1, min(ncol, max(_TPB, budget_bytes // per_column)))


def workspace_bytes(scheme: ColumnCumulusScheme, kx: int, ncol: int,
                    budget_bytes: int = COLUMN_CUMULUS_WORKSPACE_BUDGET_BYTES) -> int:
    """Device bytes one call allocates beyond its inputs and outputs."""
    chunk = chunk_columns(scheme, kx, ncol, budget_bytes)
    return (8 * (scheme.r8_per_level * kx + scheme.r8_extra) * chunk
            + 8 * ncol + 8 * GFS_FUNCPHYS_WORDS)


def run_columns(scheme: ColumnCumulusScheme, inputs: dict, random_draw, *,
                dt: float, stepcu: int, flags: int | None = None,
                want_qc: bool = True, want_qi: bool = True,
                stages: bool = False,
                budget_bytes: int = COLUMN_CUMULUS_WORKSPACE_BUDGET_BYTES) -> dict:
    """One scheme call over every column.

    ``inputs`` are float32 device arrays, level-major with the column
    fastest: :data:`LEVEL_INPUTS` ``(kx, ncol)``, :data:`IFACE_INPUTS`
    ``(kx+1, ncol)`` and ``xland`` ``(ncol,)``.  ``random_draw`` is the
    float64 ``(ncol,)`` random input.  Returns :data:`LEVEL_OUTPUTS`
    ``(kx, ncol)`` and :data:`SURFACE_OUTPUTS` ``(ncol,)``, plus ``stages``
    ``(ncol, 6*kx+5)`` binary64 when asked (the oracle's stage records).
    """
    import cupy as cp

    from woof.core.kernels import load_module

    kx, ncol = inputs["t"].shape
    if outside_vertical_bounds(kx, scheme.vertical_bounds):
        raise refuse_vertical_levels(
            f"{scheme.label} cumulus", scheme.vertical_bounds, kx,
            breakage=scheme.vertical_breakage)
    for name in LEVEL_INPUTS:
        _check(name, inputs[name], (kx, ncol), cp.float32)
    for name in IFACE_INPUTS:
        _check(name, inputs[name], (kx + 1, ncol), cp.float32)
    _check("xland", inputs["xland"], (ncol,), cp.float32)
    _check(scheme.random_input, random_draw, (ncol,), cp.float64)
    tb = gfs_funcphys_tables(scheme.module)
    out = {name: cp.zeros((kx, ncol), dtype=cp.float32) for name in LEVEL_OUTPUTS}
    out.update({name: cp.zeros(ncol, dtype=cp.float32) for name in SURFACE_OUTPUTS})
    stage_buf = (cp.zeros((ncol, 6 * kx + 5), dtype=cp.float64) if stages else None)
    chunk = chunk_columns(scheme, kx, ncol, budget_bytes)
    r8cap = scheme.r8_per_level * kx + scheme.r8_extra
    if r8cap < 42 * kx + 1:
        raise ValueError("CU_OSAS workspace requires 42*kx+1 binary64 words")
    pool = cp.empty(r8cap * chunk, dtype=cp.float64)
    err = cp.zeros(1, dtype=cp.int32)
    module = load_module(scheme.module)
    if scheme.launch_stages:
        entries = tuple((phase, module.get_function(scheme.entry + "_" + phase))
                        for phase in scheme.launch_stages)
    else:
        entries = (("column", module.get_function(scheme.entry)),)
    from woof import wrf_exact
    run_flags = ((scheme.strict_flags if wrf_exact.ENABLED else scheme.flags)
                 if flags is None else int(flags))
    fixed = (tuple(inputs[n] for n in LEVEL_INPUTS[:6])
             + (inputs["w"], inputs["p"], inputs["p8w"], inputs["pi"],
                inputs["rho"], inputs["dz8w"], inputs["xland"], random_draw,
                out["rthcuten"], out["rqvcuten"],
                out["rqccuten"] if want_qc else _NULL,
                out["rqicuten"] if want_qi else _NULL,
                out["raincv"], out["pratec"], out["htop"], out["hbot"], tb))
    for col0 in range(0, ncol, chunk):
        n = min(chunk, ncol - col0)
        args = fixed + (np.int64(ncol), np.int64(col0), np.int32(chunk),
                        np.int32(kx), np.float32(dt), np.int32(stepcu),
                        np.int32(run_flags), pool, np.int32(r8cap), err,
                        _NULL if stage_buf is None else stage_buf)
        # These launches share the stream and the bounded workspace. Finish
        # gives each level its own thread; the recurrences remain per column.
        for phase, kernel in entries:
            threads = 256 if phase == "finish" else _TPB
            count = n * kx if phase == "finish" else n
            kernel(((count + threads - 1) // threads,), (threads,), args)
    # Workspace size follows the complete automatic-array inventory.
    # Read the device error flag only in the opt-in stage replay.
    if stages and int(err.get()[0]):
        raise RuntimeError(
            f"the {scheme.label} column workspace overflowed its pool "
            f"({r8cap} binary64 words per column at kx={kx}); its r8_per_level "
            "in woof.core.column_cumulus.COLUMN_CUMULUS_SCHEMES is short, and "
            "the overflowing columns' outputs are not trustworthy")
    if stage_buf is not None:
        out["stages"] = stage_buf
    return out


def _check(name, array, shape, dtype):
    if (array.shape != shape or array.dtype != dtype
            or not array.flags.c_contiguous):
        raise ValueError(
            f"column cumulus input {name} must be a contiguous {np.dtype(dtype)} "
            f"array of shape {shape}, got {array.dtype} {array.shape}")


# ---- the cumulus callable ------------------------------------------------------
class ColumnCumulus:
    """The driver's cumulus callable for any row of :data:`COLUMN_CUMULUS_SCHEMES`.

    WHAT IT HANDS THE DRIVER.  WRF's held-rate contract: the four tendencies
    (held until the next call, as WRF holds RTHCUTEN), PRATEC and
    persistent_pratec=True.  The driver then adds PRATEC*DT to RAINC on every model step, which
    is module_physics_addtendc.F:2141 for every cumulus scheme, so a cumulus
    interval longer than the step (STEPCU > 1) books the scheme's rain
    exactly as WRF does. No per-column NCA expires the held rate.
    ``raincv`` carries the native interval increment for history and restart;
    the driver accumulates the held PRATEC independently.

    WHAT IT READS.  module_cumulus_driver.F's CU_OSAS argument list, from the
    atmosphere dictionary under the names New Tiedtke reads: temperature,
    qv, qc, qi (zeros where the moist package has none, as WRF's QI_CURR),
    u, v, pressure, exner, rho, dz, p_interface, the state's w and the
    surface xland.  DT is the model clock's step (WRF passes DT, and
    DELT = DT*STEPCU inside the scheme), STEPCU the driver's own interval.
    """

    def __init__(self, cu_physics: int) -> None:
        self.scheme = COLUMN_CUMULUS_SCHEMES[int(cu_physics)]
        self._injected = None
        self._driver = None
        self.last_htop = None
        self.last_hbot = None
        # Deterministic setup, gathered with geography rather than interpolated
        # with nest physics carriers. A tile must use the domain's column IDs.
        self.column_indices = None

    def ensure_carriers(self, state) -> None:
        """Allocate restart diagnostics before any nest/store inventory."""
        import cupy as cp
        shape = state.p.shape[1:]
        if self.column_indices is None:
            self.column_indices = cp.arange(shape[0] * shape[1], dtype=cp.uint64).reshape(shape)
        elif self.column_indices.shape != shape:
            raise ValueError("OSAS column index shape differs from domain")
        if self._injected is not None:
            self._injected = self._injected.reshape(shape)
        for name in ("last_htop", "last_hbot"):
            value = getattr(self, name)
            if value is None:
                setattr(self, name, cp.zeros(shape, dtype=cp.float32))
            elif value.shape != shape:
                raise ValueError(f"{name} shape {value.shape} differs from domain {shape}")

    def bind_driver(self, driver) -> None:
        self._driver = driver

    def inject_random(self, draw) -> None:
        """Strict arm only: the next calls read this per-column draw."""
        import cupy as cp
        self._injected = cp.ascontiguousarray(draw, dtype=cp.float64).reshape(-1)
        if self.column_indices is not None:
            self._injected = self._injected.reshape(self.column_indices.shape)

    def release(self) -> int:
        return 0

    def _stepcu(self, cfg) -> int:
        driver = self._driver
        if driver is not None and getattr(driver, "stepcu", None):
            return int(driver.stepcu)
        from woof.core.physics import _physics_interval_steps
        return int(_physics_interval_steps(cfg.cudt_minutes, cfg.dt))

    def __call__(self, *, atmosphere, fields, state, cfg):
        import cupy as cp

        from woof import wrf_exact
        from woof.core.physics import (PBL_RQI_MICROPHYSICS, CumulusResult,
                                        _model_clock_dt)

        self.ensure_carriers(state)
        nz, ny, nx = state.p.shape
        ncol = ny * nx

        def level(a):
            if a is None:
                return cp.zeros((nz, ncol), dtype=cp.float32)
            return cp.ascontiguousarray(a, dtype=cp.float32).reshape(nz, ncol)

        inputs = {
            "t": level(atmosphere["temperature"]), "qv": level(atmosphere["qv"]),
            "qc": level(atmosphere.get("qc")), "qi": level(atmosphere.get("qi")),
            "u": level(atmosphere["u"]), "v": level(atmosphere["v"]),
            "p": level(atmosphere["pressure"]), "pi": level(atmosphere["exner"]),
            "rho": level(atmosphere["rho"]), "dz8w": level(atmosphere["dz"]),
            "w": cp.ascontiguousarray(state.w, dtype=cp.float32).reshape(nz + 1, ncol),
            "p8w": cp.ascontiguousarray(atmosphere["p_interface"],
                                        dtype=cp.float32).reshape(nz + 1, ncol),
            "xland": cp.ascontiguousarray(fields["xland"],
                                          dtype=cp.float32).reshape(ncol),
        }
        dt = float(_model_clock_dt(cfg))
        stepcu = self._stepcu(cfg)
        if wrf_exact.ENABLED:
            if self._injected is not None:
                draw = cp.ascontiguousarray(self._injected, dtype=cp.float64).reshape(ncol)
            else:
                draw = random_input(ncol, step=0, module_name=self.scheme.module,
                                    strict_value=self.scheme.strict_random_value)
        else:
            epoch = float(getattr(state, "domain_start_offset", 0.0) or 0.0)
            step = int(np.floor((float(state.elapsed_seconds) - epoch)
                                / float(cfg.dt) + 0.5)) + 1
            draw = random_input(ncol, step=step, module_name=self.scheme.module,
                                column_indices=self.column_indices)
        has_qc = getattr(state, "qc", None) is not None
        has_qi = (getattr(state, "qi", None) is not None
                  and cfg.mp_physics in PBL_RQI_MICROPHYSICS)
        out = run_columns(self.scheme, inputs, draw, dt=dt, stepcu=stepcu,
                          want_qc=has_qc, want_qi=has_qi)
        self.last_htop[...] = out["htop"].reshape(ny, nx)
        self.last_hbot[...] = out["hbot"].reshape(ny, nx)

        def grid(a):
            return cp.ascontiguousarray(a.reshape(nz, ny, nx))

        return CumulusResult(
            rthcuten=grid(out["rthcuten"]), rqvcuten=grid(out["rqvcuten"]),
            rqccuten=grid(out["rqccuten"]) if has_qc else None,
            rqicuten=grid(out["rqicuten"]) if has_qi else None,
            pratec=out["pratec"].reshape(ny, nx), persistent_pratec=True,
            raincv=out["raincv"].reshape(ny, nx))
