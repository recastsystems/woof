"""CUDA port of the complete MPAS-A v8.4.2 isobaric diagnostic module.

Pool arrays use (level,entity), matching the dycore, and scalar arrays use
(scalar,level,cell). cellsOnVertex remains native one-based, including the
nCells+1 sentinel. Python only orchestrates device kernels and pool binding.
"""
from woof.hex import numerics as _hex_numerics
from functools import lru_cache
import os

LEVELS = (50, 100, 200, 250, 500, 700, 850, 925)
FAMILIES = ("temperature", "relhum", "dewpoint", "uzonal", "umeridional",
            "height", "w", "vorticity")
FIELDS = tuple(f"{family}_{level}hPa" for family in FAMILIES for level in LEVELS) + (
    "mslp", "t_isobaric", "z_isobaric", "meanT_500_300")


@lru_cache(maxsize=None)
def _module(device):
    import cupy as cp
    from ._artifact import cuda_source, authenticate, artifact_architecture, OPTIONS
    source = cuda_source()
    with cp.cuda.Device(device):
        module = cp.cuda.function.Module()
        artifact = os.environ.get("HEX_MPASDIAG_CUDA_ARTIFACT")
        if artifact:
            architecture = artifact_architecture(artifact, cp.cuda.Device(device).compute_capability)
            path = authenticate(artifact, source, architecture)
            module.load(path.read_bytes())
        else:
            # p5's gpu-extended-ftz-failure.json witnesses RawModule's
            # appended -ftz=true overriding --ftz=false. Load direct NVRTC
            # output so native subnormal interpolation words are retained.
            from cupy.cuda import compiler
            binary, _ = compiler.compile_using_nvrtc(source, options=_hex_numerics.options(OPTIONS),
                         arch=cp.cuda.Device(device).compute_capability)
            module.load(binary)
        return module


def _kernel(name):
    import cupy as cp
    return _module(cp.cuda.runtime.getDevice()).get_function(name)


def _launch(name, count, args):
    if count:
        _kernel(name)(((count + 127) // 128,), (128,), args)


def _real(array, name, shape=None):
    import cupy as cp
    if not isinstance(array, cp.ndarray) or array.dtype != cp.float32:
        raise TypeError(f"{name} must be a CUDA float32 array (native RKIND)")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name}: expected {shape}, got {array.shape}")
    return cp.ascontiguousarray(array)


def interp_tofixed_pressure(pres_in, field_in, pres_out):
    """MPAS v8.4.2 mpas_isobaric_diagnostics.F:1005-1091.

    Inputs are (column,level), increasing pressure. The native collective
    search is preserved even for unordered targets and nonmonotone inputs.
    """
    import cupy as cp
    import numpy as np
    p = _real(pres_in, "pres_in")
    if p.ndim != 2 or p.shape[1] < 2:
        raise ValueError("fixed-pressure interpolation requires at least two levels")
    f = _real(field_in, "field_in", p.shape)
    po = _real(pres_out, "pres_out")
    if po.ndim != 2 or po.shape[0] != p.shape[0]:
        raise ValueError("pres_out column count must match pres_in")
    nc, ni = p.shape
    out = cp.empty_like(po)
    no = po.shape[1]
    if nc and no:
        # Whole-grid form of the one-block collective search (_kernels.py):
        # three short launches per output level instead of one 128-thread
        # block walking every column (about 1.2 s a frame at 216k cells).
        upper = cp.zeros(nc, dtype=cp.int32)
        start = cp.full(no + 1, ni - 1, dtype=cp.int32)
        start[0] = 0  # min over the zero-initialised kupper
        count = cp.zeros(no * ni, dtype=cp.int32)
        stop = cp.empty(2 * no, dtype=cp.int32)
        blocks = ((nc + 255) // 256,)
        i32 = np.int32
        nc32, ni32, no32 = i32(nc), i32(ni), i32(no)
        for k32 in map(i32, range(no)):
            _kernel("fixed_pressure_count")(blocks, (256,), (nc32, ni32, no32, k32, p, po, upper, start, count))
            _kernel("fixed_pressure_scan")((1,), (1,), (nc32, ni32, k32, start, count, stop))
            _kernel("fixed_pressure_apply")(blocks, (256,), (nc32, ni32, no32, k32, p, f, po, out, upper, start, stop))
    return out


def compute_layer_mean(field_in, press_in, p1, p2):
    """MPAS v8.4.2 mpas_isobaric_diagnostics.F:1245-1351."""
    import cupy as cp
    p = _real(press_in, "press_in")
    if p.ndim != 2 or p.shape[1] < 2:
        raise ValueError("layer mean requires at least two levels")
    f = _real(field_in, "field_in", p.shape)
    out = cp.empty(p.shape[0], dtype=cp.float32)
    _launch("layer_mean", p.shape[0], (p.shape[0], p.shape[1], cp.float32(p1),
                                     cp.float32(p2), f, p, out))
    return out


def compute_slp(p, temperature, height, qv, *, pascals=False):
    """MPAS v8.4.2 mpas_isobaric_diagnostics.F:1094-1227.

    Arrays are (column,level); p is hPa, height has one extra level.
    Failure of any column zeros every column, as in native MPAS.
    """
    import cupy as cp
    p = _real(p, "p")
    if p.ndim != 2 or p.shape[1] < 2:
        raise ValueError("SLP requires at least two levels")
    nc, nl = p.shape
    temp = _real(temperature, "temperature", p.shape)
    h = _real(height, "height", (nc, nl + 1))
    q = _real(qv, "qv", p.shape)
    out = cp.zeros(nc, dtype=cp.float32)
    failed = cp.zeros(1, dtype=cp.int32)
    _launch("slp_columns", nc, (nc, nl, p, temp, h, q, out, failed))
    _launch("slp_finish", nc, (nc, out, failed, cp.float32(100 if pascals else 1)))
    return out


class IsobaricDiagnostics:
    def __init__(self, *, exchange_pressure_halo):
        # The exchange must complete on the launch stream before kernels run.
        self.exchange_pressure_halo = exchange_pressure_halo

    def setup(self, pools, simulation_clock, utils):
        """MPAS v8.4.2 mpas_isobaric_diagnostics.F:55-71."""
        # Retain the live pool provider: state-level rotation may replace a
        # view between calls, as well as update arrays inside the same view.
        self.pools = pools
        self.clock, self.utils = simulation_clock, utils

    def compute(self):
        """MPAS v8.4.2 mpas_isobaric_diagnostics.F:85-309."""
        wanted = {name for name in FIELDS if self.utils.field_will_be_written(name)}
        if wanted:
            self.interp_diagnostics(wanted)

    def interp_diagnostics(self, wanted):
        """MPAS v8.4.2 mpas_isobaric_diagnostics.F:313-1001.

        Bound state is native time level 1. All family levels are updated
        when any member is selected; height/w share their native condition.
        """
        m, s, d = (self.pools[name] for name in ("mesh", "state", "diag"))
        need = {family: any(f"{family}_{level}hPa" in wanted for level in LEVELS)
                for family in FAMILIES}
        thermo = any(need[x] for x in ("temperature", "relhum", "dewpoint")) or "mslp" in wanted
        # mpas_isobaric_diagnostics.F:643-658,922-925 omit additional-T
        # initialization and read the whole uninitialized array. Cross-host/rank
        # strict executable witnesses are in native-temperature-undefined.json.
        # DECLARED NATIVE UNDEFINED: use the intended in-bounds temperature
        # formula for this request combination, per NATIVE-DEFECT-POLICY.md.
        thermo = thermo or bool(wanted & {"t_isobaric", "meanT_500_300"})
        import cupy as cp
        nc, nl, nv = m["nCells"], m["nVertLevels"], m["nVertices"]
        if nl < 2:
            raise ValueError("native interface pressure requires at least two mass levels")
        self.exchange_pressure_halo(d["pressure_p"])
        def mass(a, name):
            return _real(a, name, (nl, nc + 1))
        pp, pb = mass(d["pressure_p"], "pressure_p"), mass(d["pressure_base"], "pressure_base")
        theta, exner = mass(s["theta_m"], "theta_m"), mass(d["exner"], "exner")
        padded_qv = getattr(s, "padded_qv", None)
        if padded_qv is not None:
            # Resident pools hand over the moisture slot alone (the only scalar read here), not a padded copy
            # of every scalar (resident_output_diagnostics_v842 _Pools.padded_qv).
            q = mass(padded_qv(), "qv")
        else:
            scalars = _real(s["scalars"], "scalars")
            if scalars.ndim != 3 or scalars.shape[1:] != (nl, nc + 1):
                raise ValueError("scalars must have native (scalar,level,nCells+1) shape")
            q = mass(scalars[s["index_qv"] - 1], "qv")
        h = _real(m["zgrid"], "zgrid", (nl + 1, nc + 1))
        p = cp.empty((nl, nc + 1), cp.float32)
        temp, dew = (cp.empty((nl, nc), cp.float32) for _ in range(2))
        pi = cp.empty((nl + 1, nc), cp.float32)
        pv = cp.empty((nl, nv), cp.float32)
        _launch("prepare_mass", (nc + 1) * nl, (nc + 1, nl, nc + 1, int(thermo),
                pp, pb, theta, exner, q, p, temp, dew))
        _launch("prepare_interfaces", nc * (nl + 1), (nc, nl, nc + 1, p, h, pi))
        cov = m["cellsOnVertex"]
        if not isinstance(cov, cp.ndarray) or cov.dtype != cp.int32 or cov.shape != (m["vertexDegree"], nv):
            raise ValueError("cellsOnVertex must be native one-based CUDA int32 (degree,vertex)")
        cov = cp.ascontiguousarray(cov)
        kite = _real(m["kiteAreasOnVertex"], "kiteAreasOnVertex", cov.shape)
        area = _real(m["areaTriangle"], "areaTriangle", (nv,))
        _launch("prepare_vertices", nv * nl, (nc, nv, nl, m["vertexDegree"], cov, kite, area, p, pv))
        targets = cp.broadcast_to(cp.asarray(LEVELS, dtype=cp.float32), (nc, 8)).copy()
        def columns(a):
            return cp.ascontiguousarray(a[::-1].T)
        pin = columns(p[:, :nc])
        def family(name, values, pressure=pin):
            result = interp_tofixed_pressure(pressure, columns(values), targets if name != "vorticity"
                else cp.broadcast_to(cp.asarray(LEVELS, dtype=cp.float32), (nv, 8)).copy())
            for j, level in enumerate(LEVELS):
                d[f"{name}_{level}hPa"][:result.shape[0]] = result[:, j]
        for name, values in (("temperature", temp), ("dewpoint", dew)):
            if need[name]:
                family(name, values)
        for name, key in (("relhum", "relhum"), ("uzonal", "uReconstructZonal"),
                          ("umeridional", "uReconstructMeridional")):
            if need[name]:
                family(name, mass(d[key], key)[:, :nc])
        if need["height"] or need["w"]:
            family("height", h[:, :nc], columns(pi))
            family("w", _real(s["w"], "w", h.shape)[:, :nc], columns(pi))
        if need["vorticity"]:
            vort = _real(d["vorticity"], "vorticity", (nl, nv + 1))
            family("vorticity", vort[:, :nv], columns(pv))
        if "mslp" in wanted:
            d["mslp"][:nc] = compute_slp(cp.ascontiguousarray(p[:, :nc].T),
                cp.ascontiguousarray(temp.T), cp.ascontiguousarray(h[:, :nc].T),
                cp.ascontiguousarray(q[:, :nc].T), pascals=True)
        if wanted & {"t_isobaric", "meanT_500_300"}:
            # Native first divides Pa by 100, then multiplies by 100 again.
            ppascals = pin * cp.float32(100)
            if "t_isobaric" in wanted:
                d["t_iso_levels"][:5] = cp.asarray((30000, 35000, 40000, 45000, 50000), cp.float32)
                levels = cp.broadcast_to(d["t_iso_levels"], (nc, m["nIsoLevelsT"])).copy()
                d["t_isobaric"][:, :nc] = interp_tofixed_pressure(ppascals, columns(temp), levels).T
            if "meanT_500_300" in wanted:
                d["meanT_500_300"][:nc] = compute_layer_mean(columns(temp), ppascals, 50000, 30000)
        if "z_isobaric" in wanted:
            d["z_iso_levels"][:13] = cp.asarray(tuple(range(30000, 90001, 5000)), cp.float32)
            levels = cp.broadcast_to(d["z_iso_levels"], (nc, m["nIsoLevelsZ"])).copy()
            d["z_isobaric"][:, :nc] = interp_tofixed_pressure(columns(pi) * cp.float32(100),
                                                           columns(h[:, :nc]), levels).T


def compute_output_diagnostics(mesh, state, diag):
    """MPAS v8.4.2 mpas_atm_core.F:916-965, using live state level 1.

    Native initial output: output diagnostics; manager reset/update/compute;
    stream write; manager reset; stream alarm reset. Subsequent outputs:
    advance clock; output diagnostics if any output alarm; manager
    update/compute; write; manager reset; stream alarm reset. Scheduling is
    supplied by the forecast driver, never inferred from output filenames.
    """
    nc, nl = mesh["nCells"], mesh["nVertLevels"]
    shape = (nl, nc + 1)
    args = [_real(state["theta_m"], "theta_m", shape),
            _real(state["rho_zz"], "rho_zz", shape), _real(mesh["zz"], "zz", shape),
            _real(state["scalars"][state["index_qv"] - 1], "qv", shape),
            _real(diag["pressure_base"], "pressure_base", shape),
            _real(diag["pressure_p"], "pressure_p", shape)]
    for name in ("theta", "rho", "pressure"):
        array = _real(diag[name], name, shape)
        if array is not diag[name]:
            raise ValueError(f"output {name} must be C contiguous for in-place native pool updates")
        args.append(array)
    _launch("output_diagnostics", nc * nl, (nc, nl, *args))
