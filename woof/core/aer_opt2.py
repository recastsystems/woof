"""WRF 4.6.1 user aerosol optics. Arithmetic lives in the shared CUDA unit.

The packaged table bytes are pinned before any device allocation. The loader
records the assembled CUDA unit under woof.core.kernels:aer_opt2.

Subnormal route into the frozen SW engine: rrtmg_sw.cu's spcvmc asymmetry
combination (zgcc) divides by an aerosol numerator, and the hardware division
can flush a subnormal operand.  aer_opt = 0 hands neutral 0/1/0 optics and
aer_opt = 3's range is bounded by its producer's physical number floors; this
producer (built off FTZ) refuses a subnormal nonzero aerosol numerator before
the chain.  The note lives here, not in rrtmg_sw.cu, so the frozen SW
translation unit keeps its sealed bytes.
"""
from pathlib import Path
import hashlib
import json
import math
import numpy as np

TABLES_SHA256 = "d37fb560e9238af21848417a6474e1a444ca809706ae31ed5226266fca5ea8e9"
TABLE_PATH = Path(__file__).resolve().parents[1] / "data/wrf_radiation/aer_opt2.json"
INPUT_FIELDS = ("AOD5502D", "ANGEXP2D", "AERSSA2D", "AERASY2D")
OPTION_KEYS = ("aer_aod550_opt", "aer_angexp_opt", "aer_ssa_opt", "aer_asy_opt")
VALUE_KEYS = ("aer_aod550_val", "aer_angexp_val", "aer_ssa_val", "aer_asy_val")
DEFAULT_OPTIONS = (1, 1, 1, 1)
DEFAULT_VALUES = (0.12, 1.3, 0.85, 0.9)


def load_tables(path=None):
    path = TABLE_PATH if path is None else Path(path)
    from woof.tables import table_path, CorruptTableAssets
    try:
        path = table_path("aer-opt2", path)
    except CorruptTableAssets as error:
        raise CorruptTableAssets(f"aer_opt=2 spectral table hash differs: {error}") from error
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != TABLES_SHA256:
        raise ValueError("aer_opt=2 spectral table hash differs; altered aerosol coefficients would change shortwave optics")
    data = json.loads(payload)
    for scheme in ("rrtmg", "goddard"):
        data[scheme] = {k: np.asarray(v, np.float32) for k, v in data[scheme].items()}
    data["rh"] = np.asarray(data["rh"], np.float32)
    return data


def validate_options(*, aer_type=1, options=DEFAULT_OPTIONS, values=DEFAULT_VALUES):
    if type(aer_type) is not int or aer_type not in (1, 2, 3):
        raise ValueError("aer_type must be 1 (rural), 2 (urban) or 3 (maritime)")
    if len(options) != 4 or len(values) != 4:
        raise ValueError("four aerosol options and values are required")
    for i, (opt, val) in enumerate(zip(options, values)):
        if type(opt) is not int or opt not in ((1, 2) if i == 0 else (1, 2, 3)):
            raise ValueError(f"{OPTION_KEYS[i]} has no WRF aerosol operation for {opt!r}")
        if opt == 1 and (not math.isfinite(val) or
                        (i == 0 and val < 0) or
                        (i in (2, 3) and not 0 <= val <= 1)):
            raise ValueError(f"{VALUE_KEYS[i]} is outside WRF's fixed-value bounds")


def optics_device(p, t, qv, dz8w, ht, *, aer_type=1, options=DEFAULT_OPTIONS,
                  values=DEFAULT_VALUES, inputs=None, nlayers=None, scheme="rrtmg"):
    """Return (tau, ssa, asy, band10_aod, 2D diagnostics) on the device.

    p/t/qv/dz8w are contiguous float32 (column, model layer), bottom first.
    tau/ssa/asy use the SW engine layout (column, band, wrapper layer).
    The added wrapper layers retain 0/1/0, as WRF RRTMG_SWRAD initializes them.
    inputs contains already ingested column arrays for the selected opt=2 keys.
    """
    import cupy as cp
    from woof.core.kernels import get_kernel
    validate_options(aer_type=aer_type, options=options, values=values)
    if scheme not in ("rrtmg", "goddard"):
        raise ValueError("aerosol shortwave scheme must be rrtmg or goddard")
    shape = getattr(p, "shape", ())
    if len(shape) != 2:
        raise ValueError("aerosol columns must have shape (ncol, nz)")
    nc, nz = shape
    if nz < 1:
        raise ValueError("an aerosol column needs at least one model layer")
    for name, array in zip(("p", "t", "qv", "dz8w"), (p,t,qv,dz8w)):
        if (not isinstance(array, cp.ndarray) or array.dtype != cp.float32 or
                array.shape != shape or not array.flags.c_contiguous):
            raise ValueError(f"{name} must be contiguous float32 columns")
        invalid = ~cp.isfinite(array)
        if name in ("p", "t", "dz8w"):
            invalid |= array <= 0
        if bool(cp.any(invalid).item()):
            raise ValueError(f"{name} contains nonfinite or nonpositive aerosol column inputs")
    ht = cp.ascontiguousarray(ht, dtype=cp.float32).reshape(-1)
    if ht.shape != (nc,) or bool(cp.any(~cp.isfinite(ht)).item()):
        raise ValueError("aerosol terrain must contain one finite height per column")
    nl = nz if nlayers is None else int(nlayers)
    if nl < nz:
        raise ValueError("aerosol wrapper layers cannot truncate the model column")
    tables = load_tables()
    band = tables[scheme]
    nb = 14 if scheme == "rrtmg" else 11
    inp = cp.empty((nc, 4), dtype=cp.float32)
    inputs = {} if inputs is None else inputs
    for i, (key,opt,val) in enumerate(zip(INPUT_FIELDS, options, values)):
        if opt == 2:
            if key not in inputs:
                raise ValueError(f"aer_opt=2 {OPTION_KEYS[i]}=2 requires ingested {key}; missing aerosol input would replace the requested optics")
            a = cp.asarray(inputs[key], dtype=cp.float32).reshape(-1)
            if a.shape != (nc,):
                raise ValueError(f"{key} must contain one value per column")
            invalid = ~cp.isfinite(a)
            if i == 0:
                invalid |= a < 0
            if i == 2:
                invalid |= (a < 0) | (a > 1)
            if i == 3:
                invalid |= (a < -1) | (a > 1)
            if bool(cp.any(invalid).item()):
                raise ValueError(f"{key} has values outside WRF's gridded aerosol bounds")
            inp[:,i] = a
        else:
            inp[:,i] = np.float32(val)
    out = [cp.empty((nc,nb,nl),cp.float32) for _ in range(3)]
    aod = cp.empty((nc,nz),cp.float32)
    diag = cp.empty((nc,4),cp.float32)
    table_arrays = [cp.asarray(band[k]) for k in ("raod","ssa","asy","lower_wvl","upper_wvl")]
    get_kernel("aer_opt2", "aer_opt2_columns")(
        ((nc+63)//64,), (64,),
        (np.int32(nc),np.int32(nz),np.int32(nl),np.int32(nb),np.int32(aer_type),
         *[np.int32(x) for x in options],p,t,qv,dz8w,ht,inp,*table_arrays,
         cp.asarray(tables["rh"]),*out,aod,diag))
    bad=cp.zeros(1,cp.int32)
    get_kernel("aer_opt2","aer_opt2_normal_numerator")(
        ((out[0].size+255)//256,),(256,),
        (np.int32(out[0].size),*out,bad))
    if int(bad.item()):
        raise ValueError("aer_opt=2 produces a subnormal aerosol asymmetry numerator; legacy SW hardware division can flush this operand, so this column is refused instead of changing its radiation")
    return (*out,aod,diag)


def auxiliary_inputs(cfg, start_time, elapsed_seconds, shape):
    """Select WRF auxinput15 through the shared Rust NetCDF reader.

    Python chooses the file and time index. The bridge decodes its fields.
    Missing selected input or a mismatched grid is refused before optics run.
    """
    from datetime import timedelta
    from woof.netcdf_bridge import open_dataset
    required=[key for key,opt in zip(INPUT_FIELDS,(getattr(cfg,k) for k in OPTION_KEYS)) if opt==2]
    if not required:
        return {}
    interval=int(cfg.aer_auxinput15_interval)
    offset=float(elapsed_seconds)
    if interval:
        offset=int(offset//interval)*interval
    valid=start_time+timedelta(seconds=offset)
    name=cfg.aer_auxinput15.replace('<domain>',f'{int(cfg.grid_id):02d}').replace('<date>',valid.strftime('%Y-%m-%d_%H:%M:%S'))
    with open_dataset(name) as dataset:
        missing=[key for key in required if key not in dataset.variables]
        if missing:
            raise ValueError(f"auxinput15 {name} lacks {', '.join(missing)}; selected aerosol fields cannot be replaced by fixed values")
        index=0
        if 'Times' in dataset.variables:
            rows=dataset.variables['Times'][:]
            times=[b''.join(row).decode('ascii').rstrip('\x00 ') for row in rows]
            matches=[i for i,text in enumerate(times) if text<=valid.strftime('%Y-%m-%d_%H:%M:%S')]
            if not matches:
                raise ValueError(f"auxinput15 {name} has no record at or before the radiation time")
            index=matches[-1]
        out={}
        for key in required:
            variable=dataset.variables[key]
            dims=tuple(variable.dimensions)
            if dims not in (('Time','south_north','west_east'),('south_north','west_east')):
                raise ValueError(f"auxinput15 {key} must use the WRF horizontal mass grid, got {dims}")
            value=variable[index] if len(dims)==3 else variable[:]
            if value.shape!=shape:
                raise ValueError(f"auxinput15 {key} shape {value.shape} differs from the radiation grid {shape}")
            out[key]=value
        return out
