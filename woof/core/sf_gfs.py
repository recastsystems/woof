"""GFS surface layer, ``sf_sfclay_physics = 3`` (WRF v4.6.1 module_sf_gfs.F).

CUDA launcher around ``kernels/sf_gfs.cu``: SF_GFS over the executed part of
PROGTM, in binary64 as the scheme's ``kind_phys`` is, with GFUNCPHYS's FPVS
saturation table packaged from WRF's construction and checked
through the shared asset reader before anything reads it.  The CPU
reference is :mod:`woof.verify.sfgfs_ref`; both are graded word for word
against ``tools/sfgfs_wrf461_oracle`` (tests/test_sfgfs_*_parity.py).

WHAT IT READS: the lowest model level's u, v, T, qv and p, the surface
pressure, TSK and XLAND (and, under FRACTIONAL_SEAICE = 1, XICE and the
SST carrier).  WHAT IT WRITES: the shared surface-seam set
(BR, CHS, CHS2, CPM, CQS2, FLHC, FLQC, GZ1OZ0, HFX, LH, PSIM, PSIH, QFX,
QGH, QSFC, U10, V10, WSPD) and its carried state ZNT and UST.  It writes no
MOL, ZOL, RMOL, REGIME, FM/FH, T2, Q2 or TH2: WRF's SF_GFS has none, so
those keep whatever the run allocated, exactly as in WRF.

Fast arithmetic is the default. GPUWM_NUMERICS=strict or GPUWM_WRF_EXACT=1
selects the rounding-pinned native column arithmetic, without changing the
binary64 column declaration.

DECLARED DIVERGENCES (WRF defects repaired, each measured by the oracle):

* the FRACTIONAL_SEAICE wrapper's open-water call is given XLAND_SEA, the
  open-water mask WRF builds for it, where WRF v4.6.1 passes XLAND
  (module_surface_driver.F:5751 vs :5772) and so computes the open-water
  fraction of a sea-ice cell as land; every other *_seaice_wrapper in the
  file passes XLAND_SEA.  On the oracle's columns the repaired call is
  0 ULP from WRF's statements with that one argument corrected
  (tools/sfgfs_wrf461_oracle arm C); the shipped call (arm B) differs only
  on sea-ice cells real.exe marked XLAND = 1.
* the wrapper publishes the open-water skin temperature it computed
  (TSK_SEA) to the carrier the post-LSM reblend reads.  WRF keeps it local
  to the GFS wrapper (:5670), so under GFS the surface driver's own TSK_SEA,
  an automatic array (:1370), reaches the RUC reblend (:3561) unset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from woof.core.device_cache import cuda_cache
from woof.core.state import DTYPE

#: WRF's selector value.
GFS_SFCLAY_SCHEME = 3

#: Inputs, (ny, nx) float32, in kernel argument order.
GFS_SFCLAY_INPUTS = ("u1", "v1", "t1", "qv1", "p1", "psfc", "tsk", "xland")
#: Carried state the scheme reads and rewrites.
GFS_SFCLAY_INOUT = ("znt", "ust")
#: The surface-seam fields SF_GFS writes, in kernel argument order.
GFS_SFCLAY_OUTPUTS = ("br", "chs", "chs2", "cpm", "cqs2", "flhc", "flqc",
                      "gz1oz0", "hfx", "lh", "psim", "psih", "qfx", "qgh",
                      "qsfc", "u10", "v10", "wspd")
#: The open-water carriers the FRACTIONAL_SEAICE wrapper writes for the
#: post-LSM reblend, in kernel argument order.
GFS_SEA_CARRIERS = ("chs_sea", "chs2_sea", "cpm_sea", "cqs2_sea",
                    "flhc_sea", "flqc_sea", "hfx_sea", "lh_sea", "qfx_sea",
                    "qgh_sea", "qsfc_sea", "ust_sea", "znt_sea")
#: Fields the wrapper blends by ice fraction right after its two calls
#: (module_surface_driver.F:5782-5813); the rest wait for the LSM.
GFS_SEAICE_BLENDED = ("br", "gz1oz0", "psim", "psih", "u10", "v10", "wspd")

#: tbpvs(1:7501), c1xpvs, c2xpvs as WRF's GFUNCPHYS leaves them, binary64
#: little-endian: the digest of the words the oracle read out of
#: module_gfs_funcphys's own storage (tests/data/oracles/sfgfs/
#: fpvs_table.bin). The shared asset reader checks these words before
#: the table is uploaded to the card.
FPVS_TABLE_SHA256 = (
    "bef4a54b636c57f74951560d3f5f0a65e12337cbedeae1ac17be1376f66e2235")
#: Table length: GFS_NXPVS words then c1xpvs and c2xpvs.
FPVS_TABLE_WORDS = 7501 + 2

_TPB = 128


def wrf_surface_constants() -> dict[str, np.float32]:
    """The REAL constants the surface driver hands SF_GFS.

    module_model_constants.F's PARAMETERs, folded in float32 one operation
    at a time as gfortran folds a default-kind PARAMETER expression (the
    rule woof.core.constants documents): CP=7.*R_D/2., RCP=R_D/CP,
    EP_1=R_V/R_D-1., EP_2=R_D/R_V, XLV, KARMAN.
    """
    f = np.float32
    r_d, r_v = f(287.0), f(461.6)
    cp = f(f(f(7.0) * r_d) / f(2.0))
    return {"cp": cp, "rovcp": f(r_d / cp), "r": r_d, "xlv": f(2.5e6),
            "ep1": f(f(r_v / r_d) - f(1.0)), "ep2": f(r_d / r_v),
            "karman": f(0.4)}


@cuda_cache(maxsize=None, ready=True)
def _module_and_table():
    """Upload the manifest-checked WRF table through the shared card cache."""
    import cupy as cp
    from woof.core.kernels import load_module
    from woof.table_assets import read_binary_table
    path = Path(__file__).resolve().parents[1] / "data" / "sfgfs" / "fpvs_table.bin"
    data = read_binary_table(path)
    values = np.frombuffer(data, dtype="<f8")
    return load_module("sf_gfs"), cp.asarray(values)


def fpvs_table():
    """The checked FPVS table on the current card (device array)."""
    return _module_and_table()[1]


def _check(name, array, shape):
    if (array.shape != shape or array.dtype != DTYPE
            or not array.flags.c_contiguous):
        raise ValueError(
            f"the GFS surface layer requires {name} as a contiguous float32 "
            f"array of shape {shape}, got {array.dtype} {array.shape}")


def launch_gfs_sfclay(inputs, state, outputs, *, isfflx: int,
                      seaice: dict | None = None) -> None:
    """Run SF_GFS (or its FRACTIONAL_SEAICE wrapper) over every column.

    ``inputs`` holds :data:`GFS_SFCLAY_INPUTS`, ``state`` the carried
    ``znt``/``ust`` (updated in place) and ``outputs`` every name in
    :data:`GFS_SFCLAY_OUTPUTS`, all ``(ny, nx)`` float32 device arrays.

    ``seaice`` selects the FRACTIONAL_SEAICE = 1 wrapper
    (module_surface_driver.F:2324-2338): it holds ``xice``, the ``sst``
    carrier (INOUT), every :data:`GFS_SEA_CARRIERS` array (OUT),
    ``xice_threshold`` and WRF's one-based ``itimestep``.
    """
    module, table = _module_and_table()
    shape = inputs["u1"].shape
    for name in GFS_SFCLAY_INPUTS:
        _check(name, inputs[name], shape)
    for name in GFS_SFCLAY_INOUT:
        _check(name, state[name], shape)
    for name in GFS_SFCLAY_OUTPUTS:
        _check(name, outputs[name], shape)
    n = int(np.prod(shape))
    if n == 0:
        return
    k = wrf_surface_constants()
    consts = (k["cp"], k["rovcp"], k["r"], k["xlv"], k["ep1"], k["ep2"],
              k["karman"], np.int32(isfflx))
    blocks = (n + _TPB - 1) // _TPB
    args_in = tuple(inputs[name] for name in GFS_SFCLAY_INPUTS)
    args_out = tuple(outputs[name] for name in GFS_SFCLAY_OUTPUTS)
    if seaice is None:
        kernel = module.get_function("gfs_sfclay")
        kernel((blocks,), (_TPB,),
               (np.int32(n), *args_in, state["znt"], state["ust"],
                *args_out, table, *consts))
        return
    _check("xice", seaice["xice"], shape)
    _check("sst", seaice["sst"], shape)
    for name in GFS_SEA_CARRIERS:
        _check(name, seaice[name], shape)
    kernel = module.get_function("gfs_sfclay_seaice")
    kernel((blocks,), (_TPB,),
           (np.int32(n), *args_in, seaice["xice"], seaice["sst"],
            state["znt"], state["ust"], *args_out,
            *(seaice[name] for name in GFS_SEA_CARRIERS), table, *consts,
            np.float32(seaice["xice_threshold"]),
            np.int32(seaice["itimestep"])))


__all__ = ["FPVS_TABLE_SHA256", "FPVS_TABLE_WORDS", "GFS_SEAICE_BLENDED",
           "GFS_SEA_CARRIERS", "GFS_SFCLAY_INOUT", "GFS_SFCLAY_INPUTS",
           "GFS_SFCLAY_OUTPUTS", "GFS_SFCLAY_SCHEME", "fpvs_table",
           "launch_gfs_sfclay", "wrf_surface_constants"]
