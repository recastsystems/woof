"""The Eta-family surface layers: Eta similarity (``sf_sfclay_physics = 2``)
and QNSE (``sf_sfclay_physics = 4``).

CUDA launcher around :mod:`woof.core.kernels.myjsfc` and
:mod:`woof.core.kernels.qnsesfc`.  WRF treats the two as one family -- its
surface driver sets the same ``myj`` flag for both
(phys/module_surface_driver.F:1898-1899) -- because QNSE's
phys/module_sf_qnsesfc.F is module_sf_myjsfc.F with Sukoriansky's quasi-normal
scale elimination functions: the same argument list, the same INOUT state
(AKHS/AKMS/THZ0/QZ0/UZ0/VZ0/QSFC/USTAR/ZNT) and the same published set.  So
one launcher runs both, and what separates them is the row of
:data:`ETA_FAMILY_SURFACE_LAYERS` the selector names: which kernel, which
QNSESFCINIT/MYJSFCINIT table set, which extra surface inputs and table
scalars.  Admitting another member is a row, not a launcher.

Eta similarity: the device mirror of the float32 CPU authority
``woof.verify.myj_ref.np_myjsfc_column``; both halves interpolate the SAME
host-built tables (:func:`woof.core.myjsfc_tables.build_psi_tables`, a
transcription of ``MYJSFCINIT``, phys/module_sf_myjsfc.F:1174-1299).
Conformance status: implemented-unverified (``tests/test_myj_port.py``).

Conformance status: BIT-IDENTICAL to WRF v4.6.1's MYJSFC and MYJSFCINIT
on the 224-column oracle (tools/myjsfc_wrf461_oracle,
tests/test_myjsfc_wrf461_parity.py) under the strict build and under
default arithmetic; registry maturity stays implemented-unverified because
no matched forecast trajectory exists.  ``surface["ht"]`` is WRF's HT:
MYJSFC's interface heights start at the terrain height (ZINT(KTE+1)=HT,
:165), PBLH and ZSL are differences of them, and the float32 words depend
on it.
QNSE: graded word for word against the WRF v4.6.1 column oracle
(tools/qnsesfc_wrf461_oracle, tests/test_qnsesfc_wrf461_parity.py), its
tables against QNSESFCINIT's own dump
(:func:`woof.core.myjsfc_tables.build_qnse_psi_tables`).  It reads WRF's
HT: QNSESFC's interface heights start at the terrain height
(ZINT(KTE+1)=HT, module_sf_qnsesfc.F:168), and its stable land branch
reads HT again through TOPOTERM=TOPOFAC*HT**2 (:668).  It also READS the
incoming ``hfx`` output slot: its land branch re-diagnoses THZ0 from the
land-surface model's HFX (:744).

Pairing (who may consume these layers) is :func:`woof.config.
validate_myj_pairing`'s, not this module's.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

from woof.core.device_cache import cuda_cache

import cupy as cp
import numpy as np

from woof.core.kernels import get_kernel, get_kernel_int_defines
from woof.core.myjsfc_tables import build_psi_tables, build_qnse_psi_tables
# WRF INOUT surface state and pure outputs, in kernel argument order.  The
# tables live in the runtime-free inventory module so the VRAM estimator
# prices the same names this launcher binds without importing cupy.
from woof.core.physics_inventory import (MYJ_SFCLAY_INOUT,
                                          MYJ_SFCLAY_OUTPUTS)
from woof.core.state import DTYPE

_TPB = 128

#: Read-only column inputs, in kernel argument order.
_COLUMN_INPUTS = ("dz", "tke")
#: Read-only surface inputs, in kernel argument order (the Eta row's).
_SURFACE_INPUTS = ("u1", "v1", "t1", "th1", "qv1", "qc1", "p1", "psfc",
                   "tsk", "xland", "mavail", "z0base", "ht")
_TABLE_ARRAYS = ("psim1", "psih1", "psim2", "psih2")


class EtaFamilySurfaceLayer(NamedTuple):
    """One WRF surface layer of the Eta family, as the launcher runs it."""

    #: Kernel translation unit and entry point.
    module: str
    entry: str
    #: The host-side INIT transcription whose tables the kernel reads.
    tables: Callable
    #: Read-only surface inputs, in kernel argument order.
    surface_inputs: tuple[str, ...]
    #: Table scalars passed by value after the four arrays, in order.
    table_scalars: tuple[str, ...]


#: WRF ``sf_sfclay_physics`` value -> its row.
ETA_FAMILY_SURFACE_LAYERS: dict[int, EtaFamilySurfaceLayer] = {
    # module_sf_myjsfc.F (MYJSFCSCHEME).
    2: EtaFamilySurfaceLayer(
        module="myjsfc", entry="myjsfc_column", tables=build_psi_tables,
        surface_inputs=_SURFACE_INPUTS,
        table_scalars=("dzeta1", "dzeta2", "ztmin1", "ztmax1", "ztmin2",
                       "ztmax2", "fh01", "fh02")),
    # module_sf_qnsesfc.F (QNSESFCSCHEME).  HT is an input (see the module
    # docstring); FH01/FH02 are set by QNSESFCINIT and read by nothing, the
    # QNSE SIMH factor being PRT0 (:589).
    4: EtaFamilySurfaceLayer(
        module="qnsesfc", entry="qnsesfc_column",
        tables=build_qnse_psi_tables,
        surface_inputs=_SURFACE_INPUTS,
        table_scalars=("dzeta1", "dzeta2", "ztmin1", "ztmax1", "ztmin2",
                       "ztmax2")),
}


def seed_eta_family_10m_winds(fields, sf_sfclay_physics) -> None:
    """Cold-start U10E/V10E, as WRF's phy_init sets them.

    WRF v4.6.1 phys/module_physics_init.F:1318-1327, inside the
    ``IF(.not.restart)`` block (:1156-1348), copies U10/V10 into the
    "special output from myjsfc and qnsesfc" for either Eta-family surface
    layer before the first surface-layer call, so the initial history frame
    carries U10E = U10 and V10E = V10.  WOOF's own cold starts (source
    analysis, child case, HRRR prep) call this once their U10/V10 hold the
    start winds; the wrfinput path applies the same rule inline in
    woof.ingest.wrfinput.initialize_wrfinput_physics (its CPU-only test
    substitutes numpy for cupy).  A restart never seeds them.
    """
    if int(sf_sfclay_physics) in ETA_FAMILY_SURFACE_LAYERS and "u10e" in fields:
        fields["u10e"][...] = fields["u10"]
        fields["v10e"][...] = fields["v10"]


@cuda_cache(maxsize=None, ready=True)
def _device_tables(correctly_rounded_atan: bool = False, sf_sfclay_physics: int = 2):
    """Upload one scheme's INIT tables once per process.

    The scalars ride along as Python floats; the kernel takes them by
    value.  The cache keeps the four 10001-word arrays resident for the
    life of the process rather than re-uploading 160 kB every call.
    ``correctly_rounded_atan`` is the referee WRF's libm generation
    (:func:`woof.wrf_exact.referee_correctly_rounded_trig`).
    """
    row = ETA_FAMILY_SURFACE_LAYERS[sf_sfclay_physics]
    tables = row.tables(correctly_rounded_atan)
    arrays = tuple(cp.asarray(tables[name], dtype=DTYPE)
                   for name in _TABLE_ARRAYS)
    scalars = tuple(DTYPE(tables[name]) for name in row.table_scalars)
    return arrays, scalars


def launch_myj_sfclay(columns, surface, state, outputs, *,
                      itimestep: int, ocean_z0=None, sf_sfclay_physics: int = 2) -> None:
    """Run one MYJ surface-layer step in place.

    ``columns`` supplies the full bottom-up ``(nz, ny, nx)`` ``dz`` and
    ``tke`` arrays the PBLH scan needs (module_sf_myjsfc.F:263-277,
    module_sf_qnsesfc.F:258-275); ``surface`` the lowest-model-level and
    static fields (``ht`` too for QNSE); ``state`` the nine WRF INOUT
    values; ``outputs`` the outputs (QNSE also reads ``hfx``).  Every array
    is contiguous float32 with the same ``(ny, nx)`` surface shape.
    """
    try:
        row = ETA_FAMILY_SURFACE_LAYERS[int(sf_sfclay_physics)]
    except KeyError:
        raise ValueError(
            f"sf_sfclay_physics={sf_sfclay_physics} is not an Eta-family "
            f"surface layer (rows: {sorted(ETA_FAMILY_SURFACE_LAYERS)})"
        ) from None
    shape = surface["psfc"].shape
    nz = int(columns["dz"].shape[0])
    if nz < 2:
        raise ValueError("the Eta-family surface layers need at least two "
                         "levels: their PBLH scan walks the TKE column")
    for name in _COLUMN_INPUTS:
        array = columns[name]
        if (array.shape != (nz, *shape) or array.dtype != DTYPE
                or not array.flags.c_contiguous):
            raise ValueError(
                f"launch_myj_sfclay needs a contiguous float32 "
                f"({nz}, {shape[0]}, {shape[1]}) {name!r} column")
    named = (
        [columns[name] for name in _COLUMN_INPUTS]
        + [surface[name] for name in row.surface_inputs]
        + [state[name] for name in MYJ_SFCLAY_INOUT]
        + [outputs[name] for name in MYJ_SFCLAY_OUTPUTS])
    for array in named[len(_COLUMN_INPUTS):]:
        if (array.shape != shape or array.dtype != DTYPE
                or not array.flags.c_contiguous):
            raise ValueError("launch_myj_sfclay requires same-shape "
                             "contiguous float32 surface arrays")
    ocean_args = ()
    if ocean_z0 is not None:
        if int(sf_sfclay_physics) != 2:
            raise ValueError("wave roughness is implemented only for the MYJ surface layer")
        if (not isinstance(ocean_z0, cp.ndarray)
                or ocean_z0.shape != shape or ocean_z0.dtype != DTYPE
                or not ocean_z0.flags.c_contiguous
                or ocean_z0.device.id != surface["psfc"].device.id):
            raise ValueError("MYJ wave roughness requires contiguous float32 on the surface device/grid")
        ocean_args = (ocean_z0,)
    from woof.wrf_exact import referee_correctly_rounded_trig
    arrays, scalars = _device_tables(referee_correctly_rounded_trig(), int(sf_sfclay_physics))
    n = int(np.prod(shape))
    blocks = (n + _TPB - 1) // _TPB
    kernel = (get_kernel_int_defines("myjsfc", "myjsfc_column",
              (("OCEAN_WAVE_ROUGHNESS", 1),)) if ocean_z0 is not None
              else get_kernel(row.module, row.entry))
    kernel((blocks,), (_TPB,), tuple(named) + arrays + scalars + ocean_args
           + (np.int32(itimestep), np.int32(nz), np.int32(n)))
