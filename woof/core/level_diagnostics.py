"""CUDA leaves for WRF 4.7.1 pressure- and height-level diagnostics.

The resident driver and auxhist22/23 writer use this explicit device-array API.
No transfers, scalar device reads or synchronization occur in these launches.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

FIELDS = ("U", "V", "T", "RH", "GHT", "S", "TD", "Q")


@dataclass(frozen=True)
class LevelInputs:
    u: object
    v: object
    theta_perturbation: object
    qv: object
    ph: object
    phb: object
    pressure_perturbation: object
    pressure_base: object

    def shape(self):
        nz, ny, nx = self.theta_perturbation.shape
        if min(nz, ny, nx) < 1:
            raise ValueError("level diagnostics require nonempty columns")
        for name in ("theta_perturbation", "qv", "pressure_perturbation", "pressure_base"):
            _array(getattr(self, name), (nz, ny, nx), name)
        for name in ("ph", "phb"):
            _array(getattr(self, name), (nz + 1, ny, nx), name)
        _array(self.u, (nz, ny, nx + 1), "u")
        _array(self.v, (nz, ny + 1, nx), "v")
        return nz, ny, nx

    def arguments(self):
        return (self.u, self.v, self.theta_perturbation, self.qv, self.ph, self.phb,
                self.pressure_perturbation, self.pressure_base)


def _array(value, shape, name):
    if (getattr(value, "shape", None) != shape or value.dtype != np.dtype("float32")
            or not value.flags.c_contiguous or not hasattr(value, "__cuda_array_interface__")):
        raise ValueError(f"{name} must be a contiguous float32 CUDA array of shape {shape}")


def _targets(levels):
    if getattr(levels, "ndim", None) != 1 or levels.size < 1:
        raise ValueError("levels must be a nonempty one-dimensional CUDA array")
    _array(levels, levels.shape, "levels")
    return int(levels.size)


def pressure_levels(inputs: LevelInputs, hydrostatic_pressure, face_pressure,
                    levels, output, *, pressure_kind=2, extrapolate=1, missing=-999.,
                    bounds=None):
    """Fill eight (field,target,j,i) planes, preserving native search order.

    ``pressure_kind=1`` uses total pp+pb; 2 uses hydrostatic mass pressure.
    Geopotential height always uses hydrostatic face pressure, as native does.
    The caller owns target ordering, output storage and stream scheduling.
    """
    if pressure_kind not in (1, 2) or extrapolate not in (1, 2):
        raise ValueError("pressure_kind and extrapolate must each be 1 or 2")
    nz, ny, nx = inputs.shape()
    nl = _targets(levels)
    _array(hydrostatic_pressure, (nz, ny, nx), "hydrostatic_pressure")
    _array(face_pressure, (nz + 1, ny, nx), "face_pressure")
    _array(output, (len(FIELDS), nl, ny, nx), "output")
    if bounds is not None:
        _array(bounds, (2, ny, nx), "pressure endpoint bounds")
    from woof.core.kernels import get_kernel
    get_kernel("level_diagnostics", "pressure_levels")(((ny * nx + 127) // 128,), (128,),
        (*inputs.arguments(), hydrostatic_pressure, face_pressure, levels, output,
         np.int32(nz), np.int32(ny), np.int32(nx), np.int32(nl),
         np.int32(pressure_kind), np.int32(extrapolate), np.float32(missing),
         output if bounds is None else bounds, np.int32(bounds is not None)))


def height_levels(inputs: LevelInputs, terrain, levels, output, *, missing=-999.):
    """Fill nine planes, with pressure last; values <1 request abs(z)+terrain.

    Native ZLD always uses total pressure and does not extrapolate below ground.
    Its pressure/extrapolation namelist arguments are unused in WRF itself.
    """
    nz, ny, nx = inputs.shape()
    nl = _targets(levels)
    _array(terrain, (ny, nx), "terrain")
    _array(output, (len(FIELDS) + 1, nl, ny, nx), "output")
    from woof.core.kernels import get_kernel
    get_kernel("level_diagnostics", "height_levels")(((ny * nx + 127) // 128,), (128,),
        (*inputs.arguments(), terrain, levels, output, np.int32(nz), np.int32(ny),
         np.int32(nx), np.int32(nl), np.float32(missing)))
