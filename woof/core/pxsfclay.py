"""Pleim-Xiu surface layer, ``sf_sfclay_physics = 7``, on the device.

CUDA launcher around :mod:`woof.core.kernels.pxsfclay` (``pxsfclay.cu``),
WRF v4.6.1 ``phys/module_sf_pxsfclay.F`` (PXSFCLAY over PXSFCLAY1D) and the
fractional sea-ice wrapper ``PXSFCLAY_SEAICE_WRAPPER`` with its helper
``get_local_ice_tsk`` (``phys/module_surface_driver.F:6687-6972``,
``:7117-7208``).  ``pxsfclayinit`` has an empty body: the scheme carries no
table and reads no file, so there is nothing to load beside the kernel.

Strict arithmetic (``GPUWM_WRF_EXACT=1``) preserves WRF statement order,
subnormals and scalar libm for module replay. The default uses CUDA math
and permits contraction. Both modes launch the same column physics.

What the scheme publishes (WRF's INOUT/OUT set) is
:data:`woof.core.physics_inventory.PX_SFCLAY_STATE`; it writes no FM, FH,
USTM, TH2, T2 or Q2; in WRF the land model or its SFCDIAGS forms the 2 m
fields from CHS2/CQS2, and the PX land model forms its own RA/RS from the
PSIH, UST, ZNT and RMOL this layer leaves behind.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Mapping

import numpy as np

from woof.core.physics_inventory import (PX_SFCLAY_INPUTS, PX_SFCLAY_SEA,
                                          PX_SFCLAY_STATE)

MODULE_OPTIONS = ("-std=c++17",)
MODULE_KEY = "woof.core.pxsfclay:pxsfclay"
ASSET_BYTES = 1486
ASSET_SHA256 = "98383ff4ad9f694dde50b98482778d3abfe9eec155f8061253a4730f49f015fc"
_BLOCK = 128


def constant_document(root: Path | None = None) -> dict:
    """Load the pinned words through the existing table-asset validator."""
    import json
    from woof.data_assets import data_path
    from woof.core.thompson_contract import TableAsset, read_validated_table_asset

    directory = data_path("pxsfclay") if root is None else Path(root)
    asset = TableAsset("constants.json", ASSET_BYTES, ASSET_SHA256)
    try:
        raw = read_validated_table_asset(directory, asset)
    except (OSError, ValueError) as error:
        raise ValueError("PX constants refused: changed coefficient words would "
                         "change surface fluxes and restart identity. " + str(error)) from error
    return json.loads(raw)


@lru_cache(maxsize=1)
def wrf_constants() -> dict[str, np.float32]:
    """WRF module_model_constants words from the hash-validated asset."""
    return {key: np.float32(float.fromhex(value))
            for key, value in constant_document()["driver"].items()}


def source_constant_prefix(root: Path | None = None) -> str:
    """Exact constant words, with no literal lookup tables in the kernel.

    ``root`` is the constants directory of the tree being composed (A193);
    the imported package's when omitted.
    """
    from woof.wrf_exact import ENABLED, referee_glibc
    doc = constant_document(root)
    rows = [f"#define {key} {value}f"
            for key, value in doc["scheme"].items()]
    for key, values in doc["atan"].items():
        words = ", ".join(value + "f" for value in values)
        rows.append(f"__constant__ float WOOF_ATAN_{key.upper()}[{len(values)}] = {{{words}}};")
    if ENABLED and referee_glibc() >= (2, 41):
        rows.append("#define PX_MODERN_ATAN 1")
    return "\n".join(rows) + "\n"


def module_source(kernel_dir: Path | None = None) -> str:
    """The shared loader's exact source string."""
    from woof.core.kernels import module_source as shared_source
    if kernel_dir is None:
        return shared_source("pxsfclay")
    return shared_source("pxsfclay", kernel_dir=kernel_dir)


def compile_module(source: str):
    """Compile through the shared no-FTZ route, including mutation controls."""
    from woof.core.kernels import compile_noftz_module
    return compile_noftz_module("pxsfclay", source, MODULE_KEY)


@lru_cache(maxsize=None)
def _module(device: int):
    import cupy as cp
    from woof.core.kernels import load_module
    with cp.cuda.Device(device):
        return load_module("pxsfclay")


def _kernel(name: str, module=None):
    import cupy as cp

    if module is None:
        module = _module(cp.cuda.Device().id)
    return module.get_function(name)


def _checked(arrays, shape):
    import cupy as cp

    for array in arrays:
        if (array.shape != shape or array.dtype != cp.float32
                or not array.flags.c_contiguous):
            raise ValueError("the PX surface layer needs same-shape "
                             "contiguous float32 surface arrays")
    return arrays


def _constant_args(dx: float, itimestep: int, n: int) -> tuple:
    c = wrf_constants()
    return (c["cp"], c["g"], c["rovcp"], c["r"], c["xlv"], np.float32(dx),
            c["svp1"], c["svp2"], c["svp3"], c["svpt0"], c["ep1"], c["ep2"],
            c["karman"], np.int32(itimestep), np.int32(n))


def plain_arguments(inputs: Mapping, state: Mapping, *, dx: float,
                    itimestep: int) -> tuple:
    """``pxsfclay_column``'s argument tuple, validated."""
    shape = inputs["psfc"].shape
    ins = _checked([inputs[name] for name in PX_SFCLAY_INPUTS], shape)
    st = _checked([state[name] for name in PX_SFCLAY_STATE], shape)
    n = int(np.prod(shape))
    return tuple(ins) + tuple(st) + _constant_args(dx, itimestep, n)


def seaice_arguments(inputs: Mapping, state: Mapping, sea: Mapping, *,
                     xice, sst, dx: float, itimestep: int,
                     xice_threshold: float, tice2tsk_if2cold: bool) -> tuple:
    """``pxsfclay_seaice_column``'s argument tuple, validated."""
    shape = inputs["psfc"].shape
    ins = _checked([inputs[name] for name in PX_SFCLAY_INPUTS], shape)
    st = _checked([state[name] for name in PX_SFCLAY_STATE], shape)
    sw = _checked([sea[name] for name in PX_SFCLAY_SEA], shape)
    _checked([xice, sst], shape)
    n = int(np.prod(shape))
    return (tuple(ins) + (xice, sst) + tuple(st) + tuple(sw)
            + (np.float32(xice_threshold), np.int32(bool(tice2tsk_if2cold)))
            + _constant_args(dx, itimestep, n))


def _grid(n: int) -> tuple[tuple[int], tuple[int]]:
    return ((n + _BLOCK - 1) // _BLOCK,), (_BLOCK,)


def launch_pxsfclay(inputs: Mapping, state: Mapping, *, dx: float,
                    itimestep: int, module=None) -> None:
    """One PXSFCLAY call in place (module_surface_driver.F:2163).

    ``inputs`` holds :data:`PX_SFCLAY_INPUTS` (the lowest-level wind,
    temperature, theta, vapour, pressure, layer depth, and the surface
    pressure, skin temperature, PBL height, moisture availability and land
    mask); ``state`` holds :data:`PX_SFCLAY_STATE`, WRF's INOUT words read
    and rewritten in place, U10/V10 written.  ``itimestep`` is WRF's
    one-based step count: on step 1 QSFC is the saturation value everywhere.
    """
    args = plain_arguments(inputs, state, dx=dx, itimestep=itimestep)
    grid, block = _grid(int(args[-1]))
    _kernel("pxsfclay_column", module)(grid, block, args)


def launch_pxsfclay_seaice(inputs: Mapping, state: Mapping, sea: Mapping, *,
                           xice, sst, dx: float, itimestep: int,
                           xice_threshold: float, tice2tsk_if2cold: bool,
                           module=None) -> None:
    """One PXSFCLAY_SEAICE_WRAPPER call in place (:2149).

    The land/ice call runs on the diagnosed ice skin temperature, the
    open-water call on SST under every cell with ``xice_threshold <= XICE
    <= 1``, and the INOUT set and U10/V10 leave as the XICE-weighted blend.
    ``sea`` receives the open-water CHS, CHS2, CPM, CQS2, FLHC, FLQC, HFX,
    LH, QFX, QGH, QSFC and skin temperature on those cells only -- the
    words the post-LSM blend reads -- and is untouched elsewhere.  ``sst``
    is INOUT: get_local_ice_tsk floors and, on the first three steps,
    caps it on ice cells.
    """
    args = seaice_arguments(inputs, state, sea, xice=xice, sst=sst, dx=dx,
                            itimestep=itimestep,
                            xice_threshold=xice_threshold,
                            tice2tsk_if2cold=tice2tsk_if2cold)
    grid, block = _grid(int(args[-1]))
    _kernel("pxsfclay_seaice_column", module)(grid, block, args)


__all__ = ["MODULE_OPTIONS", "compile_module", "launch_pxsfclay",
           "launch_pxsfclay_seaice", "module_source", "plain_arguments",
           "seaice_arguments", "wrf_constants"]
