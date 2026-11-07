"""The Pleim-Xiu land-surface model (``sf_surface_physics = 7``).

A port of WRF v4.6.1 ``phys/module_sf_pxlsm.F`` -- the ``pxlsm`` driver,
VEGELAND, SOILPROP, PXSNOW, SURFPX, QFLUX and SMASS -- as one CUDA column
kernel, :file:`woof/core/kernels/pxlsm.cu`, with the land-use tables of
``phys/module_sf_pxlsm_data.F`` packaged as
:file:`woof/data/pxlsm/pxlsm_tables.json` under a SHA-256 pin.

The scheme carries a two-layer soil (a 1 cm surface layer and a 99 cm root
zone, WRF's ``init_soil_depth_7``), a canopy water store, and a diagnosed
2 m temperature and mixing ratio of its own, which it writes into T2/Q2
(WRF's surface driver calls no SFCDIAGS after it,
module_surface_driver.F:3598-3697).  Its optional indirect soil nudging
(``pxlsm_soil_nudge = 1``) moves soil moisture and the deep soil
temperature toward analysed 2 m temperature and humidity, interpolated in
time between the analysis before and after the current time.

Verification: tests/test_pxlsm_wrf461_parity.py grades the kernel against
tools/pxlsm_wrf461_oracle (WRF as built) word for word under
``GPUWM_WRF_EXACT=1`` and measures the default arm's distance per field.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType

import numpy as np

from woof.core.device_cache import cuda_cache

#: WRF's sf_surface_physics selector for this scheme (PXLSMSCHEME).
PXLSM_SCHEME = 7

#: The soil column WRF's init_soil_depth_7 builds (module_soil_pre.F:
#: 1247-1267): two layers, DZS = (0.01, 0.99) m, centres ZS = (0.005,
#: 0.505) m.  It is fatal at any other layer count, and so is woof.
NUM_SOIL_LAYERS = 2
WRF_SUPPORTED_NUM_SOIL_LAYERS = (2,)
PX_LAYER_THICKNESS_M = np.array([0.01, 0.99], dtype=np.float32)
PX_LAYER_CENTRE_M = np.array([0.005, 0.505], dtype=np.float32)

#: The packaged tables and their pin.  The bytes are produced by
#: tools/pxlsm_wrf461_oracle/pack_tables.py from the gfortran object of the
#: pinned module_sf_pxlsm_data.F; tests/test_pxlsm_tables.py holds every
#: word equal to that object's dump.
PXLSM_TABLES_RELATIVE = Path("data/pxlsm/pxlsm_tables.json")
PXLSM_TABLES_SHA256 = "42c59c74b2d5531afd81997eb4ad9edaa297ae1c9b07021e091c79de2f00d3c2"

#: Row order of the packed device table, the kernel's PXT_* indices.
PX_TABLE_ROWS = ("RSMIN", "Z00", "VEG0", "VEGMN0", "LAI0", "LAIMN0",
                 "SNUP0", "ALBF", "SNOALB")


@dataclass(frozen=True)
class PxLandUseDataset:
    """One of VEGELAND's LAND_USE_TYPE arms (module_sf_pxlsm.F:749-814)."""
    table: str          # dataset key in the packaged tables
    kwat: int           # VEGELAND's water category (1-based)
    kernel_kind: int    # pxlsm.cu PX_LU_*


#: pxlsm chooses the land-use tables by the CATEGORY COUNT alone
#: (module_sf_pxlsm.F:365-381), and so does this table: 50 NLCD50, 40
#: NLCD40, 20/21/61 MODIS, 24 USGS, 28 USGS28.  Any other count is WRF's
#: own fatal ('Unknown Land Use Category') and woof's refusal.
PX_DATASETS = MappingProxyType({
    24: PxLandUseDataset("USGS", 16, 0),
    28: PxLandUseDataset("USGS28", 16, 1),
    50: PxLandUseDataset("NLCD50", 1, 2),
    40: PxLandUseDataset("NLCD40", 17, 3),
    20: PxLandUseDataset("MODIS", 17, 4),
    21: PxLandUseDataset("MODIS", 17, 4),
    61: PxLandUseDataset("MODIS", 17, 4),
})

#: Soil categories pxlsm's SOILPROP reads (STATSGO, module_sf_pxlsm.F:1763).
PX_SOIL_CATEGORIES = 16

#: What woof does differently from the WRF source, and why.  Every item is
#: also stated in the kernel header and in the registry option's warnings.
PXLSM_DIVERGENCES = (
    "RSTMIN on a land column with no leaf area in any of its land-use "
    "categories (every present category's LAI0 is zero -- a land column "
    "covered by a water category other than VEGELAND's KWAT, which real.exe "
    "never produces because it folds lakes into ISWATER) is WRF's 0/0 "
    "(module_sf_pxlsm.F:997), a NaN that reaches TSK.  woof gives such a "
    "column its dominant category's RSMIN.  Every column with leaf area is "
    "WRF's word; tools/pxlsm_wrf461_oracle's modis21-lake-defect case pins "
    "both behaviours.",
    "pxlsm reads eight soil parameters (B, C1SAT, C2R, AS, JP, C3, WRES, "
    "CGSAT) that are never set on a water column and so hold the previous "
    "column's values.  woof defines them; on open water no statement reads "
    "them and on a water column turned to ice they reach only "
    "(WSAT/W2CG)**(0.5*B/ALN10) with WSAT = W2CG = 1, which is 1 for every "
    "B, so no output can differ.",

)


def tables_path() -> Path:
    return Path(__file__).resolve().parents[1] / PXLSM_TABLES_RELATIVE


@lru_cache(maxsize=1)
def load_pxlsm_tables() -> MappingProxyType:
    """The packaged tables, verified against their pin, as float32 rows."""
    path = tables_path()
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != PXLSM_TABLES_SHA256:
        raise ValueError(
            f"{path} has SHA-256 {digest}, not the pinned "
            f"{PXLSM_TABLES_SHA256}: the PX land-use tables are not the ones "
            "the WRF v4.6.1 module_sf_pxlsm_data.F object holds, and every "
            "vegetation, roughness and albedo the scheme forms would move. "
            "Restore the packaged file.")
    document = json.loads(data)
    if tuple(document["rows"]) != PX_TABLE_ROWS:
        raise ValueError(f"{path}: rows {document['rows']} are not "
                         f"{PX_TABLE_ROWS}")
    out = {}
    for name, entry in document["datasets"].items():
        count = int(entry["categories"])
        rows = np.stack([np.asarray(entry[row], dtype=np.float32)
                         for row in PX_TABLE_ROWS])
        if rows.shape != (len(PX_TABLE_ROWS), count):
            raise ValueError(f"{path}: {name} rows are not {count} long")
        rows.setflags(write=False)
        out[name] = rows
    return MappingProxyType(out)


def px_dataset(nlcat: int) -> PxLandUseDataset:
    """VEGELAND's dataset for ``nlcat`` land-use categories, or refuse."""
    try:
        return PX_DATASETS[int(nlcat)]
    except KeyError:
        raise ValueError(
            f"the Pleim-Xiu LSM has land-use tables for 20, 21 or 61 "
            f"(MODIS), 24 (USGS), 28 (USGS28), 40 (NLCD40) and 50 (NLCD50) "
            f"categories, not {nlcat}: pxlsm selects its tables by the "
            "category count alone (module_sf_pxlsm.F:365-381) and WRF stops "
            "with 'Unknown Land Use Category' on any other, because there "
            "is no table to weight the land-use fractions with.") from None


def px_device_table(nlcat: int) -> np.ndarray:
    """The (9, nlcat) float32 table the kernel reads for ``nlcat``."""
    dataset = px_dataset(nlcat)
    rows = load_pxlsm_tables()[dataset.table]
    if rows.shape[1] < nlcat:
        raise ValueError(f"the packaged {dataset.table} table has "
                         f"{rows.shape[1]} categories, fewer than {nlcat}")
    return np.ascontiguousarray(rows[:, :int(nlcat)], dtype=np.float32)


@cuda_cache(ready=True)
def _device_table(nlcat: int):
    """Upload each immutable table once per device, with stream ordering.

    The driver must prime this at initialization before graph capture.
    The unbounded cache has only seven category keys per device and never
    evicts a table or synchronizes a device in the step path.
    """
    import cupy as cp
    return cp.asarray(px_device_table(nlcat))


def prepare_pxlsm(nlcat: int) -> None:
    """Prime tables and the compiled kernel before entering the step loop."""
    from woof.core.kernels import get_kernel
    _device_table(nlcat)
    get_kernel("pxlsm", "pxlsm_column")


#: Kernel argument order (woof/core/kernels/pxlsm.cu pxlsm_column).
PXLSM_ATMOSPHERE = ("dz8w1", "dz8w2", "qv3d1", "qv3d2", "t3d1", "th3d1",
                    "rho1")
PXLSM_INPUTS = ("psfc", "gsw", "glw", "rainbl", "emiss", "shdmin", "shdmax",
                "pblh", "rmol", "ust", "mavail", "sst",
                "t2_ndg_old", "t2_ndg_new", "q2_ndg_old", "q2_ndg_new",
                "sn_ndg_old", "sn_ndg_new", "vegfra", "imperv", "canfra")
PXLSM_CARRIED = ("ra", "rs", "lai", "znt", "qsfc", "capg", "canwat", "qfx",
                 "hfx", "lh", "psih", "vegf_px", "snow", "snoalb", "snowh",
                 "snowc", "albedo", "xland", "xice")
PXLSM_PUBLISHED = ("grdflx", "tsk", "t2", "q2", "t2obs", "q2obs", "lai_px",
                   "wwlt_px", "wfc_px", "wsat_px", "clay_px", "csand_px",
                   "fmsand_px")

PXLSM_BLOCK = 128


def validate_pxlsm_call(*, dt, dtbl, curr_secs, itimestep, smois_init,
                       soil_nudge, modis_veg, anal_interval, domain_shape,
                       tile_origin, tile_shape, dzs):
    """Reject scalar geometry that would divide by zero or misclassify edges.

    Runs before CUDA import or allocation. DT is the model step; DTBL is
    the surface interval used to convert accumulated rain to a rate.
    """
    for name, value in (("dt", dt), ("dtbl", dtbl)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"pxlsm {name} must be finite and positive: "
                             "the substep and precipitation-rate divisions need it")
    if not np.isfinite(curr_secs) or curr_secs < 0:
        raise ValueError("pxlsm curr_secs must be finite and non-negative")
    for name, value, minimum in (("itimestep", itimestep, 1),
                                 ("smois_init", smois_init, 0),
                                 ("anal_interval", anal_interval, 0)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"pxlsm {name} must be an integer >= {minimum}")
    for name, value in (("soil_nudge", soil_nudge), ("modis_veg", modis_veg)):
        if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
            raise ValueError(f"pxlsm {name} must be 0 or 1: other nudging "
                             "values can read analysis operands never initialized")
    if soil_nudge == 1 and anal_interval <= 0:
        raise ValueError("pxlsm soil nudging needs a positive analysis interval: "
                         "the analysis interpolation modulus is undefined at zero")
    for name, shape, minimum in (("domain_shape", domain_shape, 1),
                                 ("tile_shape", tile_shape, 1),
                                 ("tile_origin", tile_origin, 0)):
        if len(shape) != 2 or any(isinstance(v, bool) or not isinstance(v, int)
                                  or v < minimum for v in shape):
            raise ValueError(f"pxlsm {name} must contain two integers >= {minimum}")
    if any(o + n > d for o, n, d in zip(tile_origin, tile_shape, domain_shape)):
        raise ValueError("pxlsm tile extends outside domain_shape: nudging "
                         "would use the wrong domain boundary")
    if len(dzs) != NUM_SOIL_LAYERS or any(not np.isfinite(v) or v <= 0 for v in dzs):
        raise ValueError("pxlsm dzs must contain two positive finite layer "
                         "thicknesses: SURFPX divides by each layer depth")


def launch_pxlsm(fields: dict, atmosphere: dict, *, landusef, soilcbot,
                 dt: float, dtbl: float, itimestep: int, curr_secs: float,
                 smois_init: int, soil_nudge: int, modis_veg: int,
                 anal_interval: int, domain_shape: tuple[int, int],
                 tile_origin: tuple[int, int] = (0, 0),
                 dzs=PX_LAYER_THICKNESS_M, device_clock=None) -> None:
    """Run one pxlsm call on device arrays, in place.

    ``fields`` maps every name of :data:`PXLSM_INPUTS`,
    :data:`PXLSM_CARRIED` and :data:`PXLSM_PUBLISHED` to a C-contiguous
    ``(ny, nx)`` float32 array, ``isltyp`` to ``(ny, nx)`` int32 and
    ``smois``/``tslb`` to ``(2, ny, nx)`` float32.  ``atmosphere`` maps the
    names of :data:`PXLSM_ATMOSPHERE` to ``(ny, nx)`` float32 planes of the
    two lowest mass levels.  ``landusef`` is ``(nlcat, ny, nx)`` and
    ``soilcbot`` ``(16, ny, nx)``.  ``domain_shape`` is the WHOLE domain's
    (ny, nx) and ``tile_origin`` this array's (j, i) offset in it: pxlsm
    switches soil nudging off on the outer two rows and columns of the
    domain (module_sf_pxlsm.F:601-602), not of a tile.
    """
    validate_pxlsm_call(
        dt=dt, dtbl=dtbl, curr_secs=curr_secs, itimestep=itimestep,
        smois_init=smois_init, soil_nudge=soil_nudge, modis_veg=modis_veg,
        anal_interval=anal_interval, domain_shape=domain_shape,
        tile_origin=tile_origin, tile_shape=fields["tsk"].shape, dzs=dzs)
    import cupy as cp

    from woof.core.kernels import get_kernel

    ny, nx = fields["tsk"].shape
    ncol = ny * nx
    nlcat = int(landusef.shape[0])
    dataset = px_dataset(nlcat)
    if soil_nudge == 1 and int(anal_interval) <= 0:
        raise ValueError(
            "pxlsm_soil_nudge=1 needs a positive analysis interval: pxlsm "
            "interpolates the analysed 2 m temperature and humidity between "
            "the analyses before and after the current time with "
            "MOD(CURR_SECS, interval) (module_sf_pxlsm.F:399-409), which is "
            "undefined at zero -- WRF stops here too (:390-396)")
    if (isinstance(itimestep, bool) or not isinstance(itimestep, int)
            or itimestep < 1):
        raise ValueError("pxlsm itimestep must be a positive integer")
    table = _device_table(nlcat)

    def plane(name, source):
        a = source[name]
        if a.shape != (ny, nx) or a.dtype != cp.float32 or not a.flags.c_contiguous:
            raise ValueError(f"pxlsm {name}: expected C-contiguous ({ny},{nx}) float32")
        return a

    def stack(name, a, depth, dtype=cp.float32):
        if a.shape != (depth, ny, nx) or a.dtype != dtype or not a.flags.c_contiguous:
            raise ValueError(f"pxlsm {name}: expected C-contiguous "
                             f"({depth},{ny},{nx}) {dtype.__name__}")
        return a

    args = [np.int32(ncol), np.int32(nx), np.int32(ny),
            np.int32(tile_origin[1]), np.int32(tile_origin[0]),
            np.int32(domain_shape[1] + 1), np.int32(domain_shape[0] + 1),
            np.int32(nlcat), np.int32(dataset.kwat),
            np.int32(dataset.kernel_kind), np.int32(itimestep),
            np.int32(smois_init), np.int32(soil_nudge), np.int32(modis_veg),
            np.float32(dt), np.float32(dtbl), np.float32(curr_secs),
            np.int32(anal_interval),
            np.float32(dzs[0]), np.float32(dzs[1]),
            table,
            stack("landusef", landusef, nlcat),
            stack("soilcbot", soilcbot, PX_SOIL_CATEGORIES)]
    args += [plane(name, atmosphere) for name in PXLSM_ATMOSPHERE]
    args += [plane(name, fields) for name in PXLSM_INPUTS]
    args += [plane(name, fields) for name in PXLSM_CARRIED]
    args += [plane(name, fields) for name in PXLSM_PUBLISHED]
    isltyp = fields["isltyp"]
    if (isltyp.shape != (ny, nx) or isltyp.dtype != cp.int32
            or not isltyp.flags.c_contiguous):
        raise ValueError(f"pxlsm isltyp: expected ({ny},{nx}) int32")
    args.append(isltyp)
    args.append(stack("smois", fields["smois"], NUM_SOIL_LAYERS))
    args.append(stack("tslb", fields["tslb"], NUM_SOIL_LAYERS))
    args.append(np.uint64(0) if device_clock is None else device_clock)
    kernel = get_kernel("pxlsm", "pxlsm_column")
    blocks = (ncol + PXLSM_BLOCK - 1) // PXLSM_BLOCK
    kernel((blocks,), (PXLSM_BLOCK,), tuple(args))


__all__ = [
    "PXLSM_SCHEME", "NUM_SOIL_LAYERS", "WRF_SUPPORTED_NUM_SOIL_LAYERS",
    "PX_LAYER_THICKNESS_M", "PX_LAYER_CENTRE_M", "PXLSM_TABLES_RELATIVE",
    "PXLSM_TABLES_SHA256", "PX_TABLE_ROWS", "PX_DATASETS", "PX_SOIL_CATEGORIES",
    "PXLSM_DIVERGENCES", "PXLSM_ATMOSPHERE", "PXLSM_INPUTS", "PXLSM_CARRIED",
    "PXLSM_PUBLISHED", "load_pxlsm_tables", "px_dataset", "px_device_table",
    "launch_pxlsm", "prepare_pxlsm", "tables_path", "validate_pxlsm_call",
]
