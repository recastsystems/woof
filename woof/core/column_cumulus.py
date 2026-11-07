"""Table-driven column cumulus: one launcher for every one-kernel scheme.

A scheme whose whole WRF driver call runs as ONE per-column kernel is
admitted here by a row of :data:`COLUMN_CUMULUS_SCHEMES`, never by its own
adapter module: the row names the kernel unit and entry, the workspace it
needs, the namelist knobs it reads and how WRF maps ``mp_physics`` onto its
condensate classes.  :class:`ColumnCumulus` packs the engine's column state
into the kernel's level-major inputs, launches the entry over tiles of
columns with a workspace sized to the threads in flight, and returns the
driver's Task-1 :class:`~woof.core.physics.CumulusResult`.

The first row is New SAS (``cu_physics = 96``, ``woof/core/kernels/
nsas.cu``), graded word for word against WRF v4.6.1's own object by
``tests/test_nsas_column_oracle.py``.

THE KERNEL CONTRACT a row's entry must keep (``nsas_column`` is the
reference):

* level-major inputs ``a[k*ncol + i]``: ``t, qv, qc, qi, p, pi, rho, dz8w,
  u, v`` on mass levels, then ``p8w, w`` on the nz+1 interfaces;
* a per-column float table (:data:`SC_FIELDS`) and int table
  (:data:`IC_FIELDS`);
* six level-major outputs ``rthcuten, rqvcuten, rqccuten, rqicuten,
  rucuten, rvcuten`` (a null ``rqccuten``/``rqicuten`` is WRF's F_QC/F_QI
  false) and ``sfc[j*ncol + i]`` for ``raincv, pratec, hbot, htop``;
* a workspace of ``slots * (nz + 2) * tile`` floats and the launch scalars
  ``col0, ntile, ncol, nz``.

RAIN. WRF holds PRATEC between calls and accumulates PRATEC*DT on each
model-clock step (module_physics_addtendc.F:2141). The launcher returns the
persistent rate so the shared driver advances precipitation on that clock.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from types import MappingProxyType
from typing import Mapping

import numpy as np

# numpy.float32, which cupy.float32 is; woof.core.state would import CuPy
# into the CPU memory accounting (preflight.physics_array_shapes).
DTYPE = np.float32
from woof.core import column_cumulus_r8 as _r8

#: Per-column float table, in the kernel's column order.
SC_FIELDS = ("xland", "hpbl", "hfx", "qfx", "dx", "dt", "pgcon")
#: Per-column int table, in the kernel's column order.
IC_FIELDS = ("stepcu", "ncloud", "dx_factor")
#: Level-major mass and interface inputs, in the kernel's argument order.
MASS_INPUTS = ("t", "qv", "qc", "qi", "p", "pi", "rho", "dz8w", "u", "v")
IFACE_INPUTS = ("p8w", "w")
LEVEL_OUTPUTS = ("rthcuten", "rqvcuten", "rqccuten", "rqicuten",
                 "rucuten", "rvcuten")
SURFACE_OUTPUTS = ("raincv", "pratec", "hbot", "htop")

#: Threads per block, and the tile's granularity.
BLOCK = 64
#: Blocks per SM the tile is sized for (the workspace is charged per thread
#: in flight, so the tile is the card's width and no more).
TILE_BLOCKS_PER_SM = 4


def wrf_sas_ncloud(mp_physics: int) -> int:
    """module_cu_nsas.F:141-147: no cloud (0), cloud only (1), cloud+ice (2)."""
    mp = int(mp_physics)
    if mp == 0:
        return 0
    if mp in (1, 3):
        return 1
    return 2


@dataclass(frozen=True)
class ColumnCumulusScheme:
    """One row: everything the launcher needs to run a column scheme."""

    cu_physics: int
    name: str
    #: woof/core/kernels/<kernel_module>.cu
    kernel_module: str
    entry: str
    #: Workspace slots per level; the unit's own count is checked against
    #: this by tests/test_nsas_column_oracle.py.
    workspace_slots: int
    #: RunConfig field -> kernel per-column float (namelist knobs).
    config_floats: tuple[tuple[str, str], ...]
    #: RunConfig field -> kernel per-column int (namelist knobs).
    config_ints: tuple[tuple[str, str], ...]
    #: mp_physics -> the scheme's condensate classes (0, 1 or 2).
    ncloud: object
    #: The checkpoint/algorithm identity string.
    identity: str
    row_limits_entry: str | None = None


COLUMN_CUMULUS_SCHEMES = MappingProxyType({
    **_r8.COLUMN_CUMULUS_SCHEMES,
    96: ColumnCumulusScheme(
        cu_physics=96, name="New SAS (YSU) with NSCV shallow convection",
        kernel_module="nsas", entry="nsas_column", workspace_slots=50,
        config_floats=(("sas_pgcon", "pgcon"),),
        config_ints=(("nsas_dx_factor", "dx_factor"),),
        ncloud=wrf_sas_ncloud,
        identity="new-sas-nscv-wrf461-cu_nsas-v1",
        row_limits_entry="nsas_row_limits"),
})


def workspace_floats(scheme: ColumnCumulusScheme, nz: int, tile: int) -> int:
    return int(scheme.workspace_slots) * (int(nz) + 2) * int(tile)


def tile_columns(ncol: int, multiprocessor_count: int) -> int:
    """Columns in flight per launch: the card's width, never more than ncol."""
    tile = int(multiprocessor_count) * TILE_BLOCKS_PER_SM * BLOCK
    return max(1, min(int(ncol), tile))


def workspace_bytes(cu_physics: int, nz: int, ncol: int,
                    multiprocessor_count: int) -> int:
    """Device bytes the launcher allocates for one call (preflight pricing)."""
    scheme = COLUMN_CUMULUS_SCHEMES[int(cu_physics)]
    if isinstance(scheme, _r8.ColumnCumulusScheme):
        return _r8.workspace_bytes(scheme, nz, ncol)
    tile = tile_columns(ncol, multiprocessor_count)
    return 4 * workspace_floats(scheme, nz, tile)


def launch_columns(scheme: ColumnCumulusScheme, *, mass, iface, sc, ic,
                   outputs, sfc, nz: int, ncol: int, prep=None,
                   module=None, row_width: int = 1):
    """Launch ``scheme.entry`` over ``ncol`` columns in tiles.

    ``mass``/``iface`` map the kernel's input names to level-major CuPy
    arrays; ``outputs`` maps LEVEL_OUTPUTS to arrays or None; ``sfc`` is
    ``(4, ncol)``.  ``module`` overrides the compiled unit (the planted
    one-bit tests compile a perturbed copy of the source).
    """
    import cupy as cp

    if module is None:
        from woof.core.kernels import load_module
        module = load_module(scheme.kernel_module)
    if row_width < 1 or ncol % row_width:
        raise ValueError("row_width must divide ncol; partial rows change WRF cloud-search limits")
    fn = module.get_function(scheme.entry)
    limits = None
    if scheme.row_limits_entry is not None:
        nrow = ncol // row_width
        limits = cp.empty((nrow, 3), dtype=cp.int32)
        module.get_function(scheme.row_limits_entry)(
            ((nrow + BLOCK - 1) // BLOCK,), (BLOCK,),
            (mass["p"], iface["p8w"], limits, np.int32(ncol),
             np.int32(nz), np.int32(row_width)))
    sms = int(cp.cuda.Device().attributes["MultiProcessorCount"])
    tile = tile_columns(ncol, sms)
    ws = cp.empty(workspace_floats(scheme, nz, tile), dtype=DTYPE)
    null = np.uint64(0)
    args_in = tuple(mass[n] for n in MASS_INPUTS) + tuple(
        iface[n] for n in IFACE_INPUTS)
    outs = tuple(null if outputs.get(n) is None else outputs[n]
                 for n in LEVEL_OUTPUTS)
    for col0 in range(0, int(ncol), tile):
        width = min(tile, int(ncol) - col0)
        fn(((width + BLOCK - 1) // BLOCK,), (BLOCK,),
           args_in + (sc, ic) + outs
           + (sfc, null if prep is None else prep, ws,
              np.int32(col0), np.int32(width), np.int32(ncol),
              np.int32(nz), null if limits is None else limits,
              np.int32(row_width)))
    del ws


class ColumnCumulus:
    """The ``cu_physics`` callable for every row of the table.

    Stateless between calls (no hold: WRF's SAS family recomputes on its
    STEPCU cadence and the driver holds the rates), so ``release`` frees
    nothing but the driver back-reference.
    """

    def __new__(cls, cu_physics):
        if int(cu_physics) in _r8.COLUMN_CUMULUS_SCHEMES:
            return _r8.ColumnCumulus(cu_physics)
        return super().__new__(cls)

    def __init__(self, cu_physics: int):
        self.scheme = COLUMN_CUMULUS_SCHEMES[int(cu_physics)]
        self._driver = None
        self.last_htop = None
        self.last_hbot = None

    @classmethod
    def for_scheme(cls, cu_physics: int):
        return RegistryColumnCumulus.for_scheme(cu_physics)

    def bind_driver(self, driver) -> None:
        """Optional driver hook: PBL height and the cumulus cadence."""
        self._driver = driver

    def release(self) -> int:
        self._driver = None
        return 0

    def _stepcu(self) -> int:
        stepcu = getattr(self._driver, "stepcu", 1) if self._driver else 1
        return max(int(stepcu), 1)

    def __call__(self, *, atmosphere, fields, state, cfg):
        import cupy as cp

        from woof.core.kf import _model_clock_dt
        from woof.core.physics import CumulusResult

        scheme = self.scheme
        nz, ny, nx = state.p.shape
        ncol = ny * nx

        def lev(a, levels=nz):
            return cp.ascontiguousarray(a, dtype=DTYPE).reshape(levels, ncol)

        mass = {"t": lev(atmosphere["temperature"]), "qv": lev(atmosphere["qv"]),
                "qc": lev(atmosphere["qc"]), "qi": lev(atmosphere["qi"]),
                "p": lev(atmosphere["pressure"]), "pi": lev(atmosphere["exner"]),
                "rho": lev(atmosphere["rho"]), "dz8w": lev(atmosphere["dz"]),
                "u": lev(atmosphere["u"]), "v": lev(atmosphere["v"])}
        iface = {"p8w": lev(atmosphere["p_interface"], nz + 1),
                 "w": lev(state.w, nz + 1)}

        clock_dt = float(_model_clock_dt(cfg))
        stepcu = self._stepcu()
        floats = {"dx": float(cfg.dx), "dt": clock_dt}
        for field_name, slot in scheme.config_floats:
            floats[slot] = float(getattr(cfg, field_name))
        sc = cp.empty((ncol, len(SC_FIELDS)), dtype=DTYPE)
        sc[:, SC_FIELDS.index("xland")] = fields["xland"].reshape(ncol)
        sc[:, SC_FIELDS.index("hpbl")] = fields["pblh"].reshape(ncol)
        sc[:, SC_FIELDS.index("hfx")] = fields["hfx"].reshape(ncol)
        sc[:, SC_FIELDS.index("qfx")] = fields["qfx"].reshape(ncol)
        for slot, value in floats.items():
            sc[:, SC_FIELDS.index(slot)] = DTYPE(value)

        ncloud = int(scheme.ncloud(cfg.mp_physics))
        ints = {"stepcu": stepcu, "ncloud": ncloud}
        for field_name, slot in scheme.config_ints:
            ints[slot] = int(getattr(cfg, field_name))
        ic = cp.empty((ncol, len(IC_FIELDS)), dtype=cp.int32)
        for slot, value in ints.items():
            ic[:, IC_FIELDS.index(slot)] = np.int32(value)

        # WRF forms RQCCUTEN only where the state carries QC (F_QC) and
        # RQICUTEN only where it carries QI (F_QI), module_cu_nsas.F:285-303.
        has_qc = getattr(state, "qc", None) is not None
        has_qi = getattr(state, "qi", None) is not None
        out = {n: cp.empty((nz, ncol), dtype=DTYPE) for n in LEVEL_OUTPUTS}
        if not has_qc:
            out["rqccuten"] = None
        if not has_qi:
            out["rqicuten"] = None
        sfc = cp.empty((len(SURFACE_OUTPUTS), ncol), dtype=DTYPE)
        launch_columns(scheme, mass=mass, iface=iface, sc=sc, ic=ic,
                       outputs=out, sfc=sfc, nz=nz, ncol=ncol, row_width=nx)

        def grid(a):
            return None if a is None else a.reshape(nz, ny, nx)

        rqc = grid(out["rqccuten"])
        rqi = grid(out["rqicuten"])
        pratec = sfc[SURFACE_OUTPUTS.index("pratec")].reshape(ny, nx)
        return CumulusResult(
            rthcuten=grid(out["rthcuten"]), rqvcuten=grid(out["rqvcuten"]),
            rqccuten=rqc, rqicuten=rqi,
            rucuten=grid(out["rucuten"]), rvcuten=grid(out["rvcuten"]),
            pratec=cp.ascontiguousarray(pratec), persistent_pratec=True,
            # CU_NSAS assigns both native bounds (module_cu_nsas.F:263-264).
            cloud_base=(sfc[SURFACE_OUTPUTS.index('hbot')].reshape(ny, nx)
                        if 99 in (cfg.ra_lw_physics, cfg.ra_sw_physics) else None),
            cloud_top=(sfc[SURFACE_OUTPUTS.index('htop')].reshape(ny, nx)
                       if 99 in (cfg.ra_lw_physics, cfg.ra_sw_physics) else None))


# Registry-described kernels have a distinct argument contract from NSAS.

#: Threads per block for every column kernel on this route; the workspace
#: is interleaved per tile column, so any block size is admissible and this
#: one does not change the arithmetic.
_DEFAULT_BLOCK = 128

#: Blocks per SM a tile is sized for (the gf.cu tile idea: the workspace is
#: charged per column IN FLIGHT, never per column on the grid).
_TILE_BLOCKS_PER_SM = 8


@dataclass(frozen=True)
class ColumnKernelSpec:
    """One ``consumers.column_kernel`` row, parsed and checked."""

    cu_physics: int
    module: str
    entry: str
    workspace_slots: int
    block: int
    tables: Mapping | None
    level_inputs: tuple[tuple[str, str], ...]
    interface_inputs: tuple[tuple[str, str], ...]
    column_inputs: tuple[tuple[str, str], ...]
    carried: tuple[tuple[str, float], ...]
    level_outputs: tuple[str, ...]
    column_outputs: tuple[str, ...]
    int_column_outputs: tuple[str, ...]
    result: Mapping[str, str]
    vertical_minimum: int
    identity: str
    persist_outputs: tuple[str, ...]
    feedback: Mapping | None

    @classmethod
    def from_row(cls, cu_physics: int, row: Mapping) -> "ColumnKernelSpec":
        def pairs(key):
            return tuple((str(a), str(b)) for a, b in row.get(key, ()))

        tables = row.get("tables")
        spec = cls(
            cu_physics=int(cu_physics),
            module=str(row["module"]),
            entry=str(row["entry"]),
            workspace_slots=int(row["workspace_slots"]),
            block=int(row.get("block", _DEFAULT_BLOCK)),
            tables=dict(tables) if tables else None,
            level_inputs=pairs("level_inputs"),
            interface_inputs=pairs("interface_inputs"),
            column_inputs=pairs("column_inputs"),
            carried=tuple((str(c["field"]), float(c["cold_start"]))
                          for c in row.get("carried", ())),
            level_outputs=tuple(row["level_outputs"]),
            column_outputs=tuple(row["column_outputs"]),
            int_column_outputs=tuple(row.get("int_column_outputs", ())),
            result=dict(row["result"]),
            vertical_minimum=int(row.get("vertical_minimum", 1)),
            identity=str(row["identity"]),
            persist_outputs=tuple(row.get("persist_outputs", ())),
            feedback=dict(row["feedback"]) if row.get("feedback") else None,
        )
        outputs = set(spec.level_outputs) | set(spec.column_outputs)
        missing = [name for name in spec.result.values() if name not in outputs]
        if missing:
            raise ValueError(
                f"column_kernel row for cu_physics={cu_physics} maps result "
                f"fields to outputs the kernel does not write: {missing}")
        for required in ("rthcuten", "rqvcuten", "rainc"):
            if required not in spec.result:
                raise ValueError(
                    f"column_kernel row for cu_physics={cu_physics} names no "
                    f"{required!r} result: the driver couples rthcuten and "
                    "rqvcuten and accumulates rainc on every due call")
        return spec


def column_kernel_spec(cu_physics: int) -> ColumnKernelSpec | None:
    """The registry's column-kernel row for ``cu_physics``, or ``None``."""
    from woof.physics_registry import consumer_rows_by_selector

    row = consumer_rows_by_selector("cumulus", "column_kernel").get(int(cu_physics))
    if not row:
        return None
    return ColumnKernelSpec.from_row(int(cu_physics), row)


def column_kernel_selectors() -> frozenset[int]:
    """Every ``cu_physics`` value the registry admits on this route."""
    from woof.physics_registry import consumer_rows_by_selector

    return frozenset(int(value) for value, row in
                     consumer_rows_by_selector("cumulus", "column_kernel").items()
                     if row)


# ---------------------------------------------------------------- tables
_TABLE_CACHE: dict[tuple[int, str], object] = {}


#: share/module_model_constants.F values a table entry point may take, by
#: their WRF names, as the float32 words WRF's PARAMETERs fold to
#: (cp = 7.*r_d/2. is exact in float32).
WRF_MODEL_CONSTANTS = {
    "r_d": float(DTYPE(287.0)),
    "cp": float(DTYPE(DTYPE(7.0) * DTYPE(287.0) / DTYPE(2.0))),
}


def _table_constant(name: str) -> float:
    try:
        return WRF_MODEL_CONSTANTS[name]
    except KeyError:
        raise ValueError(
            f"column_kernel tables row names model constant {name!r}, which "
            f"this route does not publish (known: {sorted(WRF_MODEL_CONSTANTS)})"
        ) from None


def table_words_sha256(words: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(words, dtype="<f4").tobytes()).hexdigest()


def column_tables(spec: ColumnKernelSpec, *, verify: bool = True):
    """Build (once per device) and return the scheme's lookup tables.

    The table kernel is the transcribed table routine itself (for BMJ:
    BMJINIT and SPLINE), run on the device that will read the tables, in
    the same arithmetic as the column kernel.  The words are hashed against
    the row's pin before any column reads them.  The breakage this refusal
    prevents: a table off by one word moves the moist adiabat and the
    saturation point of every convecting column, silently, at every call.
    """
    import cupy as cp
    from woof.core.kernels import load_module

    if not spec.tables:
        return cp.zeros(1, dtype=DTYPE)
    key = (cp.cuda.Device().id, spec.module)
    cached = _TABLE_CACHE.get(key)
    if cached is not None:
        return cached
    t = spec.tables
    words = int(t["floats"])
    grid, block = int(t["grid"]), int(t["block"])
    tab = cp.zeros(words, dtype=DTYPE)
    scratch = cp.zeros(grid * block * int(t["scratch_floats_per_thread"]),
                       dtype=DTYPE)
    fn = load_module(spec.module).get_function(str(t["entry"]))
    consts = tuple(DTYPE(_table_constant(name)) for name in t["constants"])
    fn((grid,), (block,), (tab, scratch) + consts)
    del scratch
    if verify:
        digest = table_words_sha256(cp.asnumpy(tab))
        if digest != t["sha256"]:
            raise RuntimeError(
                f"{spec.module} lookup tables hash to {digest}, not the "
                f"pinned {t['sha256']} (registry consumers.column_kernel."
                "tables.sha256): every convecting column would read a "
                "different moist adiabat than the one graded against WRF. "
                "Refusing to run the scheme on them.")
    _TABLE_CACHE[key] = tab
    return tab


# ---------------------------------------------------------------- launch
def registry_tile_columns(fn, ncol: int, block: int) -> int:
    import cupy as cp

    dev = cp.cuda.Device()
    sms = dev.attributes["MultiProcessorCount"]
    per_sm = _TILE_BLOCKS_PER_SM
    try:
        from cupy_backends.cuda.api import driver

        resident = driver.occupancyMaxActiveBlocksPerMultiprocessor(
            fn.kernel.ptr, block, 0)
        per_sm = max(1, min(per_sm, int(resident)))
    except Exception:                              # noqa: BLE001
        pass
    return int(min(int(ncol), sms * per_sm * block))


def run_columns(spec: ColumnKernelSpec, *, level, interface, column, carried,
                dt: float, stepcu: int = 1, rad_feedback: bool = False,
                tables=None, ws_tile: int | None = None):
    """Launch the column kernel over level-major ``(nz, ncol)`` inputs.

    ``carried`` arrays are updated IN PLACE (they are the scheme's state).
    Returns ``{output name: array}``, level outputs ``(nz, ncol)``, column
    outputs ``(ncol,)``.
    """
    import cupy as cp
    from woof.core.kernels import load_module

    first = level[spec.level_inputs[0][0]]
    nz, ncol = (int(v) for v in first.shape)
    if nz < spec.vertical_minimum:
        raise ValueError(
            f"{spec.module} needs at least {spec.vertical_minimum} levels, "
            f"got {nz}")

    def dev(a, shape, name):
        arr = cp.ascontiguousarray(cp.asarray(a, dtype=DTYPE))
        if arr.shape != shape:
            raise ValueError(f"{spec.module} input {name}: shape {arr.shape}, "
                             f"expected {shape}")
        return arr

    args = []
    for name, _ in spec.level_inputs:
        args.append(dev(level[name], (nz, ncol), name))
    for name, _ in spec.interface_inputs:
        args.append(dev(interface[name], (nz + 1, ncol), name))
    for name, _ in spec.column_inputs:
        args.append(dev(column[name], (ncol,), name))
    for name, _ in spec.carried:
        arr = carried[name]
        if (not isinstance(arr, cp.ndarray) or arr.dtype != DTYPE
                or arr.shape != (ncol,) or not arr.flags.c_contiguous):
            raise ValueError(
                f"{spec.module} carried field {name} must be a contiguous "
                f"float32 device array of shape {(ncol,)} (updated in place)")
        args.append(arr)
    if tables is None:
        tables = column_tables(spec)
    out = {}
    for name in spec.level_outputs:
        out[name] = cp.empty((nz, ncol), dtype=DTYPE)
    for name in spec.column_outputs:
        out[name] = cp.empty((ncol,), dtype=DTYPE)
    for name in spec.int_column_outputs:
        out[name] = cp.empty((ncol,), dtype=cp.int32)
    fn = load_module(spec.module).get_function(spec.entry)
    block = spec.block
    tile = ws_tile or registry_tile_columns(fn, ncol, block)
    ws = cp.empty(spec.workspace_slots * nz * tile, dtype=DTYPE)
    tail = ([out[n] for n in spec.level_outputs]
            + [out[n] for n in spec.column_outputs]
            + [out[n] for n in spec.int_column_outputs])
    for col0 in range(0, ncol, tile):
        width = min(tile, ncol - col0)
        fn(((width + block - 1) // block,), (block,),
           tuple(args) + (tables, ws) + tuple(tail)
           + (DTYPE(dt), np.int32(stepcu), np.int32(1 if rad_feedback else 0),
              np.int32(col0), np.int32(tile), np.int32(ncol), np.int32(nz)))
    del ws
    return out


# ---------------------------------------------------------- engine callable
class RegistryColumnCumulus:
    """The engine callable for every ``consumers.column_kernel`` scheme.

    Stateless apart from the bound driver: the carried per-column state
    lives in the driver's ``fields`` (so the restart writer serialises it
    with every other field), the tables in a per-device cache.
    """

    def __init__(self, spec: ColumnKernelSpec):
        self.spec = spec
        self._driver = None
        # BMJINIT and its hash check belong to initialization. Forecast
        # calls reuse device tables and do not copy them to the host.
        column_tables(spec)

    @classmethod
    def for_scheme(cls, cu_physics: int) -> "RegistryColumnCumulus":
        spec = column_kernel_spec(cu_physics)
        if spec is None:
            raise ValueError(
                f"cu_physics={cu_physics} publishes no consumers.column_kernel "
                "row in woof/physics_registry_v2.json")
        return cls(spec)

    def restart_identity(self):
        return {"column_kernel": self.spec.identity,
                "cu_physics": self.spec.cu_physics}

    def bind_driver(self, driver) -> None:
        self._driver = driver

    def release(self) -> int:
        self._driver = None
        return 0

    def __call__(self, *, atmosphere, fields, state, cfg, **_ignored):
        import cupy as cp
        from woof.core.kf import _model_clock_dt
        from woof.core.physics import CumulusResult

        spec = self.spec
        nz, ny, nx = state.p.shape
        ncol = ny * nx

        def lev(key):
            return cp.asarray(atmosphere[key], dtype=DTYPE).reshape(nz, ncol)

        level = {name: lev(key) for name, key in spec.level_inputs}
        interface = {name: cp.asarray(atmosphere[key], dtype=DTYPE)
                     .reshape(nz + 1, ncol)
                     for name, key in spec.interface_inputs}
        column = {name: cp.asarray(fields[key], dtype=DTYPE).reshape(ncol)
                  for name, key in spec.column_inputs}
        carried = {}
        for name, _cold in spec.carried:
            arr = fields[name]
            if arr.shape != (ny, nx) or not arr.flags.c_contiguous:
                raise ValueError(f"fields[{name!r}] must be a contiguous "
                                 f"[ny, nx]={ny, nx} array")
            carried[name] = arr.reshape(ncol)          # a view: in place
        stepcu = int(getattr(self._driver, "stepcu", 1) or 1)
        dt = DTYPE(_model_clock_dt(cfg))
        out = run_columns(
            spec, level=level, interface=interface, column=column,
            carried=carried, dt=float(dt), stepcu=stepcu,
            rad_feedback=bool(spec.feedback and
                              getattr(cfg, spec.feedback["parameter"], False)))
        for name in spec.persist_outputs:
            fields[name][...] = out[name].reshape(fields[name].shape)
        # This buffer is accumulated by the existing advance_ppt clock on
        # every step, including non-due steps. This route leaves the KF
        # timer inactive, so its expiry mask never clears the held rates.
        self._driver.cu_pratec[...] = out[spec.result["rainc"]].reshape(ny, nx)
        self._driver.cu_raincv[...] = out["raincv"].reshape(ny, nx)

        def grid3(name):
            return cp.ascontiguousarray(out[name].reshape(nz, ny, nx))

        return CumulusResult(
            rthcuten=grid3(spec.result["rthcuten"]),
            rqvcuten=grid3(spec.result["rqvcuten"]))



__all__ = ["COLUMN_CUMULUS_SCHEMES", "ColumnCumulus", "ColumnCumulusScheme",
           "launch_columns", "workspace_bytes", "wrf_sas_ncloud",
           "RegistryColumnCumulus", "ColumnKernelSpec", "column_kernel_spec",
           "column_kernel_selectors", "column_tables", "run_columns",
           "table_words_sha256"]


def merge_column_cumulus_clouds(qc, qi, fraction, *, fields, cfg, columns):
    """Merge carried convective clouds into temporary radiation copies."""
    if int(getattr(cfg, 'shcu_physics', 0)) == 5:
        import cupy as cp
        qc = columns(fields['cw_rad']) + qc
        if fraction is not None:
            fraction = cp.maximum(columns(fields['cldfra_sh']), fraction)
    spec = column_kernel_spec(int(getattr(cfg, "cu_physics", 0)))
    row = None if spec is None else spec.feedback
    if not row or not getattr(cfg, row["parameter"], False):
        return qc, qi, fraction
    import cupy as cp
    from woof.core.kernels import load_module
    arrays = [cp.ascontiguousarray(columns(fields[row[name]]), dtype=DTYPE)
              for name in ("liquid", "ice", "fraction")]
    if any(a.shape != qc.shape for a in arrays) or qi.shape != qc.shape:
        raise ValueError("convective and radiation cloud fields must share shape")
    qc, qi = cp.ascontiguousarray(qc).copy(), cp.ascontiguousarray(qi).copy()
    if fraction is not None:
        fraction = cp.ascontiguousarray(fraction).copy()
        if fraction.shape != qc.shape:
            raise ValueError("radiation cloud fraction must share condensate shape")
    load_module(spec.module).get_function(row["entry"])(
        ((qc.size + 127) // 128,), (128,),
        (qc, qi, np.uint64(0) if fraction is None else fraction,
         *arrays, np.int64(qc.size)))
    return qc, qi, fraction


def _phase_presence(state):
    return tuple(getattr(state, name, None) is not None for name in ("qr", "qi", "qs"))


class MskfColumnCumulus:
    """MSKF registry contract, independent of the New SAS kernel ABI."""

    def __init__(self, selector: int = 11, *, scheme: int | None = None,
                 kf_edrates: bool = False):
        # woof.core.physics builds every KF_FAMILY_CU_SCHEMES adapter (1, 11,
        # 99) as factory(scheme=..., kf_edrates=...); without this signature
        # cu_physics=11 died at physics init with a TypeError (0 steps).
        # kf_edrates is read from the run config at each call (step below).
        import importlib
        from woof.physics_registry import physics_registry
        if scheme is not None:
            selector = int(scheme)
        self.kf_edrates = bool(kf_edrates)
        rows=physics_registry()['components']['cumulus']['options']
        option=next(v for v in rows.values() if v['selectors']['cu_physics']==selector)
        self.contract=option['extensions']['column_contract']
        self.module=importlib.import_module(self.contract['module'])
        self.w0avg=None
        self._history_state=None
        self._history_time=None

    def ensure_trigger_history(self, state):
        import cupy as cp
        shape=(int(state.w.shape[0])-1,)+tuple(state.w.shape[1:])
        if self.w0avg is None or self._history_state is not state or self.w0avg.shape!=shape:
            self.w0avg=cp.zeros(shape,cp.float32)
            for name in self.contract['outputs_3d']:
                setattr(self,name,cp.zeros(shape,cp.float32))
            for name in self.contract['outputs_2d']:
                setattr(self,name,cp.zeros(shape[1:],cp.float32))
            self.cu_act=cp.zeros(shape[1:],cp.int32)
            self._history_state=state
            self._history_time=None
        return self.w0avg

    def history_fields(self):
        """Publish the actual module carriers without allocating copies."""
        if self.w0avg is None:
            return {}
        from woof.io.wrf_output_schema import MSKF_OUTPUT_FIELDS
        return {schema.netcdf_name: getattr(self, key)
                for key, schema in MSKF_OUTPUT_FIELDS.items()}

    @property
    def radiation_feedback(self):
        if self.w0avg is None:
            return None
        return {name:getattr(self,name) for name in self.contract['radiation_feedback']}

    def update_trigger_history(self, *, state, cfg):
        from woof.core.physics import _model_clock_dt
        self.ensure_trigger_history(state)
        dt=_model_clock_dt(cfg)
        steps=max(int(np.floor(cfg.cudt_minutes*60.0/dt+0.5)),1)
        self.module.update_w0avg(self.w0avg,state.w,dt=dt,cudt_minutes=cfg.cudt_minutes,
                                stepcu=steps,adaptive=bool(cfg.use_adaptive_time_step))
        self._history_time=float(state.elapsed_seconds)

    def __call__(self, *, atmosphere, fields, state, cfg):
        import cupy as cp
        from woof.core.physics import CumulusResult, _model_clock_dt
        if self._history_state is not state or self._history_time!=float(state.elapsed_seconds):
            self.update_trigger_history(state=state,cfg=cfg)
        inputs={key:cp.ascontiguousarray(atmosphere[source],dtype=cp.float32)
                for key,source in self.contract['atmosphere_inputs'].items()}
        inputs.update({key:fields[source] for key,source in self.contract['surface_inputs'].items()})
        inputs['w0avg']=self.w0avg
        outputs={name:getattr(self,name) for name in self.contract['outputs_3d']+self.contract['outputs_2d']}
        outputs['cu_act']=self.cu_act
        outputs['nca'][...]=state.physics.cu_nca
        phase=getattr(self.module,self.contract['phase_factory'])(cfg.mp_physics in self.contract['warm_rain_selectors'],
                                   *_phase_presence(state))
        if cfg.aercu_opt>0:
            from woof.core.cumulus_aerosol import update_monthly_aerosol
            update_monthly_aerosol(state,cfg,inputs['p'])
        aerosol=getattr(state,'aerocu',None) if cfg.aercu_opt>0 else None
        self.module.raise_on_status(getattr(self.module,self.contract['launcher'])(
            inputs,outputs,dx=cfg.dx,dt=_model_clock_dt(cfg),cudt_minutes=cfg.cudt_minutes,
            phase=phase,aercu_opt=cfg.aercu_opt,aercu_fct=cfg.aercu_fct,
            kf_edrates=cfg.kf_edrates,aerocu=aerosol))
        mapping=self.contract['result_mapping']
        values={name:outputs[source] for name,source in mapping.items()}
        for result_name,species in self.contract['optional_tendencies'].items():
            if getattr(state,species,None) is None:
                values[result_name]=None
        return CumulusResult(**values)
