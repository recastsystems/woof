"""Original WRF v4.6.1 cumulus, cu_physics=6.

The column kernel follows module_cu_tiedtke.F. It is independent of New
Tiedtke and the cu16 closure emulation. tools/tiedtke_wrf461_oracle/check.py
--gpu holds it to the original Fortran word for word (docs/cumulus-cu6.md).
"""
from __future__ import annotations
import math
import numpy as np

INPUTS = ("u3d", "v3d", "w", "t3d", "qv3d", "qc3d", "qi3d", "pi3d",
          "rho3d", "qvften", "qvpblten", "dz8w", "pcps", "p8w", "xland",
          "qfx", "znu")
LEVEL_OUTPUTS = ("rthcuten", "rqvcuten", "rqccuten", "rqicuten", "rucuten", "rvcuten")
SURFACE_OUTPUTS = ("raincv", "pratec")
REFUSAL = ("cu6 refused an undefined WRF column: module_cu_tiedtke.F:2282/2289 "
           "writes flux arrays at vertical index 0 when the cloud top reaches "
           "level 3. Returning those tendencies would depend on out-of-bounds memory.")


def workspace_shapes(nz, ncol):
    """Conservative call-tree peaks, including the driver's live arrays.

    transcribe.workspace_peak computes them independently from every local
    array declaration and the call graph. One extra float covers the small
    vertical grids where a different child routine sets the maximum.
    """
    if nz < 4 or ncol < 1:
        raise ValueError("cu6 needs nz>=4 and ncol>=1: the cloud-base search reads interior levels")
    # The last integer row records the first ascent's active-column flag.
    return ((58 * nz + 25, ncol), (nz + 9, ncol), (4, ncol))


def workspace_bytes(nz, ncol):
    f, i, b = workspace_shapes(nz, ncol)
    return int(np.prod(f) * 4 + np.prod(i) * 4 + np.prod(b))


def column_step(inputs, *, dt, itimestep=2, stepcu=1, f_qc=True, f_qi=True,
                workspace=None, outputs=None, row_width=None, refused=None,
                guess=1, libmvec=False, referee_glibc=None):
    """Launch the complete original driver on bottom-up float32 columns.

    znu is shared (nz,), surfaces are (ncol,), levels (nz,ncol), interfaces
    (nz+1,ncol). Optional inactive tendencies retain their supplied words.
    Passing no output buffers initializes those optional fields to zero.
    row_width splits the columns into consecutive WRF i rows of that many
    columns; F:960's early-return predicate is evaluated per row. None
    treats the whole launch as one row.

    refused, a device bool scalar, defers the out-of-bounds refusal: the
    column flags are OR-ed into it without a host read and the caller
    raises after its last launch. None reads them here and raises.

    guess is the first launch's path for every column of a multi-column
    row: 1 continues past F:960, 0 takes the early return. Columns whose
    row decides otherwise are recomputed, so either guess gives the same
    words; it only moves cost. Continuation is the default: an inactive
    column's continuation is nearly free (5.44 ms for a whole-domain pass on
    NCAR's 12 km CONUS grid against 5.4 ms with the early return), while a
    wrong early-return guess costs those columns a second ascent. On that
    grid every row holds an active column, so nothing is recomputed.

    libmvec selects the referee build. False is gfortran at -O0 (or any
    build that does not vectorize tiecnv), the scalar expf everywhere, as
    tools/tiedtke_wrf461_oracle builds it. True is WRF's stock gfortran
    -O2 -ftree-vectorize build (configure option 34, gfortran 15.2, glibc
    2.43): tiecnv's conversion loop runs four columns of each row at a time
    and F:588's exp is libmvec's _ZGVbN4v_expf there, the scalar expf in
    the row's last (width mod 4) columns. The row is row_width columns, or
    the whole launch when row_width is None, as WRF's ite-its+1.

    referee_glibc is an independent (major, minor) tuple for F:2939's
    tangent: before 2.41 the Sun routine, from 2.41 the correctly rounded
    routine. Its default follows GPUWM_WRF_GLIBC in strict proof mode;
    fast numerics keeps the newer routine. Vectorization never selects
    the tangent generation.
    """
    import cupy as cp
    from woof.core.kernels import load_module

    if set(inputs) != set(INPUTS):
        raise ValueError(f"cu6 input fields must be {INPUTS}")
    if inputs["t3d"].ndim != 2:
        raise ValueError("cu6 t3d must have shape (nz,ncol)")
    nz, ncol = map(int, inputs["t3d"].shape)
    shapes = workspace_shapes(nz, ncol)
    if not math.isfinite(dt) or dt <= 0 or stepcu < 1 or itimestep < 1:
        raise ValueError("cu6 requires finite positive dt and positive stepcu/itimestep: zero delt divides tendencies by zero")
    for name in INPUTS:
        expected = ((nz + 1, ncol) if name in ("w", "p8w") else
                    (ncol,) if name in ("xland", "qfx") else
                    (nz,) if name == "znu" else (nz, ncol))
        a = inputs[name]
        if not isinstance(a, cp.ndarray) or a.dtype != np.float32 or not a.flags.c_contiguous or a.shape != expected:
            raise ValueError(f"cu6 {name} must be contiguous device float32 {expected}: kernel indexing uses this layout")
    if workspace is None:
        workspace = (cp.empty(shapes[0], dtype=cp.float32),
                     cp.empty(shapes[1], dtype=cp.int32),
                     cp.empty(shapes[2], dtype=cp.bool_))
    for a, shape, dtype in zip(workspace, shapes, (cp.float32, cp.int32, cp.bool_), strict=True):
        if a.shape != shape or a.dtype != dtype or not a.flags.c_contiguous:
            raise ValueError(f"cu6 workspace must be contiguous {dtype} {shape}: parent and child arrays must not overlap")
    if outputs is None:
        outputs = {n: cp.zeros((nz, ncol), dtype=cp.float32) for n in LEVEL_OUTPUTS}
        outputs.update({n: cp.empty(ncol, dtype=cp.float32) for n in SURFACE_OUTPUTS})
        outputs["cu_act_flag"] = cp.empty(ncol, dtype=cp.int32)
    expected_keys = set(LEVEL_OUTPUTS + SURFACE_OUTPUTS + ("cu_act_flag",))
    if set(outputs) != expected_keys:
        raise ValueError("cu6 output fields differ from the WRF driver boundary")
    for name, a in outputs.items():
        shape = (nz, ncol) if name in LEVEL_OUTPUTS else (ncol,)
        dtype = cp.int32 if name == "cu_act_flag" else cp.float32
        if a.shape != shape or a.dtype != dtype or not a.flags.c_contiguous:
            raise ValueError(f"cu6 output {name} must be contiguous {dtype} {shape}")
    width = ncol if row_width is None else int(row_width)
    if referee_glibc is None:
        from woof import wrf_exact
        referee_glibc = (wrf_exact.referee_glibc() if wrf_exact.ENABLED else (2, 43))
    old_tan = tuple(referee_glibc) < (2, 41)
    if width < 1 or ncol % width:
        raise ValueError(f"cu6 row_width {row_width} must divide the {ncol} columns: "
                         "F:960's early return is decided per complete WRF i row")
    def args(force_continue):
        return tuple(inputs[n] for n in INPUTS) + tuple(outputs[n] for n in
            LEVEL_OUTPUTS + SURFACE_OUTPUTS + ("cu_act_flag",)) + tuple(workspace) + (
            np.float32(dt), np.int32(itimestep), np.int32(stepcu), np.int32(nz), np.int32(ncol),
            np.int32(f_qc), np.int32(f_qi), np.int32(force_continue),
            np.int32(width if libmvec else 0), np.int32(old_tan))
    kernel = load_module("tiedtke").get_function("tiedtke_column")
    grid = ((ncol + 31) // 32,)
    if ncol == 1 or width == 1:
        # A one-column row's predicate is its own column's: one launch.
        kernel(grid, (32,), args(0))
    else:
        # F:960 is a row-wide early return: a row's columns stop only when
        # the first ascent found no active column anywhere in that WRF i
        # row. The predicate is not known until every column of the row has
        # ascended, so the first launch runs every column down one path
        # (the guess) and the second recomputes, from the pristine inputs
        # and in place, only the inactive columns whose row took the other
        # path. An active column never returns early, so either path is
        # its own. Mode 2 reads each column's path from its activity word:
        # 0 early return, 1 continue, -1 keep the words already written.
        # Rows never mix, and no host read decides anything.
        if guess not in (0, 1):
            raise ValueError(f"cu6 guess {guess!r} must be 1 (continue past F:960) or 0 (early return)")
        activity = workspace[1][-1].reshape(-1, width)
        activity.fill(guess)
        kernel(grid, (32,), args(2))
        active = activity != 0
        row = active.any(axis=1, keepdims=True)
        redo = ~active & (row != bool(guess))
        activity[...] = cp.where(redo, row.astype(cp.int32), cp.int32(-1))
        kernel(grid, (32,), args(2))
    bad = (outputs["cu_act_flag"] < 0).any()
    if refused is not None:
        cp.logical_or(refused, bad, out=refused)
    elif bool(bad):
        raise ValueError(REFUSAL)
    return outputs


def stock_vector_referee():
    """Whether F:588 takes the stock vectorized build's lanes.

    WOOF_WRF_LIBMVEC=1 selects WRF 4.6.1 as its stock gfortran configure
    (option 34, -O2 -ftree-vectorize) builds it with gfortran 15.2 on glibc
    2.43, the referee of the WRF-JAX paper comparison: tiecnv's conversion
    loop runs four columns at a time through libmvec's expf.  Unset or 0 is
    the scalar referee (-O0 or -fno-tree-vectorize), the build the published
    whole-hour record and tools/tiedtke_wrf461_oracle use.  The two differ
    in 3.6 per cent of F:588's exp words and nowhere else in cu6.
    """
    import os

    return os.environ.get("WOOF_WRF_LIBMVEC", "0") == "1"


#: Device bytes the persistent cu6 arenas of one domain may hold.  A launch
#: walks as many complete j rows as fit.  The column kernel needs ~255
#: registers, so an SM keeps at most 8 of its warps resident: a 64 MiB arena
#: (13 rows of NCAR's 12 km CONUS grid, 173 warps) left a 170-SM card about
#: one warp per SM and cost 38 ms per call where the whole domain in one
#: launch (1.5 GB at 50 levels) costs 11.4 ms (RTX 5090, V9, 2026-10-09).
TIEDTKE_WORKSPACE_BUDGET_BYTES = 2 << 30


def rows_per_launch(nz, nx, ny):
    """Complete j rows per launch: at least one, at most the domain.

    When the domain needs more than one launch the rows are spread evenly,
    so the last launch is not a small tail that idles the card.
    """
    fit = max(1, min(int(ny), TIEDTKE_WORKSPACE_BUDGET_BYTES // workspace_bytes(nz, nx)))
    launches = -(-int(ny) // fit)
    return -(-int(ny) // launches)


class Tiedtke:
    """cu6 adapter. Runs every model step and returns one rain increment."""

    def __init__(self):
        self._driver = None

    def bind_driver(self, driver):
        self._driver = driver

    def release(self):
        self._driver = None
        return 0  # The arenas are per call, from the pool; nothing is held.

    def __call__(self, *, atmosphere, fields, state, cfg):
        import cupy as cp
        from woof.core.physics import CumulusResult, _model_clock_dt
        from woof.core.physics_inventory import PBL_RQI_MICROPHYSICS

        nz, ny, nx = state.p.shape
        ncol = ny * nx
        def flat(a):
            return cp.ascontiguousarray(a, dtype=cp.float32).reshape(-1, ncol)
        def lane(name):
            a = None if self._driver is None else getattr(self._driver, name, None)
            if a is None:
                return cp.zeros((nz, ncol), dtype=cp.float32)
            from woof.wrf_exact import ENABLED
            if (ENABLED and name == "gf_rqvdynten"
                    and a is getattr(state, "rqvften", None)):
                # module_em.F:1372 set_tend, then phy_prep_part2 :5301.
                # The last RK transport is coupled; WRF decouples with
                # the acoustic mass carried into this physics call.
                from woof.state_serialization_contract import WRF_MUTS_SLOT
                mut = state.existing_scratch(WRF_MUTS_SLOT)
                if mut is None:
                    mut = state.total_mu()
                a = a / (state.c1h[:, None, None] * mut[None]
                         + state.c2h[:, None, None])
            return flat(a)
        # F:582. Original tiecnv adds these two lanes itself. No radiation or
        # temperature-forcing fold from cu16 is applied to cu6.
        inputs = {"u3d": flat(atmosphere["u"]), "v3d": flat(atmosphere["v"]),
                  "w": flat(state.w), "t3d": flat(atmosphere["temperature"]),
                  "qv3d": flat(atmosphere["qv"]), "qc3d": flat(atmosphere["qc"]),
                  "qi3d": flat(atmosphere["qi"]), "pi3d": flat(atmosphere["exner"]),
                  "rho3d": flat(atmosphere["rho"]), "qvften": lane("gf_rqvdynten"),
                  "qvpblten": lane("gf_rqvblten"), "dz8w": flat(atmosphere["dz"]),
                  "pcps": flat(atmosphere["pressure"]), "p8w": flat(atmosphere["p_interface"]),
                  "xland": cp.ascontiguousarray(fields["xland"]).reshape(ncol),
                  "qfx": cp.ascontiguousarray(fields["qfx"]).reshape(ncol),
                  "znu": cp.ascontiguousarray(state.znu)}
        f_qi = cfg.mp_physics in PBL_RQI_MICROPHYSICS
        # Every column writes every active rate at every level; an inactive
        # optional rate keeps the zero WRF's driver leaves it.
        result = {name: (cp.empty if f_qi or name != "rqicuten" else cp.zeros)(
                      (nz, ncol), dtype=cp.float32) for name in LEVEL_OUTPUTS}
        rain = cp.empty(ncol, dtype=cp.float32)
        dt = _model_clock_dt(cfg)
        epoch = float(getattr(state, "domain_start_offset", 0.0) or 0.0)
        itimestep = int(np.floor((float(state.elapsed_seconds) - epoch) / cfg.dt + 0.5)) + 1
        # WRF calls tiecnv once per j row. The early-return predicate must
        # never combine different rows or split a row across independent
        # tiles; column_step decides it per row of width nx. A row is one
        # WRF tile row of a single-process run, so under the stock
        # vectorized referee its F:588 lanes are that build's.
        libmvec = stock_vector_referee()
        # The arenas are drawn from the pool for this call, as GF's are:
        # every word is written before it is read inside one launch, so
        # nothing in them crosses a step. DomainState.scratch slots would
        # have entered every restart manifest (the first adapter's
        # 'tiedtke_b' raised RestartManifestError on the first real run).
        rows = rows_per_launch(nz, nx, ny)
        shapes = workspace_shapes(nz, rows * nx)
        arenas = tuple(cp.empty(int(np.prod(shape)), dtype=dtype)
                       for shape, dtype in zip(shapes, (cp.float32, cp.int32, cp.bool_),
                                               strict=True))
        # One host read per step, after the last launch, decides the refusal.
        refused = cp.zeros((), dtype=cp.bool_)
        for r0 in range(0, ny, rows):
            lo, hi = r0 * nx, min(ny, r0 + rows) * nx
            n = hi - lo
            whole = n == ncol
            chunk = inputs if whole else {
                name: a if name == "znu" else cp.ascontiguousarray(
                    a[lo:hi] if a.ndim == 1 else a[:, lo:hi])
                for name, a in inputs.items()}
            workspace = tuple(a[:shape[0] * n].reshape(shape[0], n)
                              for a, shape in zip(arenas, shapes, strict=True))
            # A launch over the whole domain writes the result in place.
            out = dict(result, raincv=rain) if whole else {
                name: result[name][:, lo:hi].copy() for name in LEVEL_OUTPUTS}
            out.update(raincv=out.get("raincv", cp.empty(n, dtype=cp.float32)),
                       pratec=cp.empty(n, dtype=cp.float32),
                       cu_act_flag=cp.empty(n, dtype=cp.int32))
            column_step(chunk, dt=dt, itimestep=itimestep, f_qi=f_qi,
                        workspace=workspace, outputs=out, row_width=nx,
                        refused=refused, libmvec=libmvec)
            if not whole:
                for name in LEVEL_OUTPUTS:
                    result[name][:, lo:hi] = out[name]
                rain[lo:hi] = out["raincv"]
        if bool(refused):
            raise ValueError(REFUSAL)
        # RAINC grows by RAINCV here and by PRATEC*DT in WRF's advance_ppt
        # (module_physics_addtendc.F:2141), with PRATEC = rn/(stepcu*dt)
        # (F:404).  At stepcu = 1, the only cadence cu6 runs, those are the
        # same word: rn = fl(a*dt) (F:645), and fl(fl(fl(a*dt)/dt)*dt) =
        # fl(a*dt) for every float32 a measured (418,110,244 values from
        # 1e-14 to 10, dt = 9-90 s), and on all 249,001 columns of the
        # captured WRF step.
        return CumulusResult(**{name: cp.ascontiguousarray(a.reshape(nz, ny, nx))
                               for name, a in result.items()}, rainc=rain.reshape(ny, nx))
