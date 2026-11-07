"""Shared ocean solver ABI v1. No CPU array conversion is permitted here.

All planes are CUDA float32 C-contiguous (ny, nx), row j then column i.
Coordinates are cell centres. Rectilinear geographic grids run south to
north and west to east; forcing preparation normalizes axis order. Distances are metres; dt is seconds. Solvers
and driver enqueue on the caller's current CUDA stream and device. They may
not synchronize or copy to host per step. Arrays remain owned by the caller
until its stream has completed. State/output buffers persist across steps.
Winds/stresses are east/north; directions travel towards degrees clockwise
from true north. Elevation and eta use the bathymetry vertical datum, in m.
The caller supplies grid rotation when projecting earth-relative forcing.
Shared output planes must be finite, including calm and dry cells. Undefined
calm-wave period and direction use zero. A dry cell's stored eta may equal its
bed elevation; water-level products include only ocean or inundated cells.
"""
from __future__ import annotations
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Mapping, Protocol, runtime_checkable
import math

ABI = "gpuwm.ocean.device/v1"

@runtime_checkable
class DeviceArray(Protocol):
    @property
    def __cuda_array_interface__(self) -> dict: ...

@dataclass(frozen=True)
class OceanGrid:
    nx: int
    ny: int
    dx: float
    dy: float
    latitude: DeviceArray
    longitude: DeviceArray
    cosalpha: DeviceArray
    sinalpha: DeviceArray

    @property
    def shape(self) -> tuple[int, int]:
        return (self.ny, self.nx)

    def validate(self) -> None:
        if self.nx < 2 or self.ny < 2 or not all(math.isfinite(x) and x > 0 for x in (self.dx,self.dy)):
            raise ValueError("ocean grid needs at least 2x2 cells and finite positive dx/dy")
        for name in ("latitude", "longitude", "cosalpha", "sinalpha"):
            validate_plane(getattr(self,name), self.shape, name)

@dataclass(frozen=True)
class Bathymetry:
    elevation: DeviceArray  # signed bed elevation; negative below datum
    depth: DeviceArray      # max(-elevation, 0), positive down
    wet: DeviceArray        # float32 1 ocean-connected wet cell, 0 otherwise
    coastal: DeviceArray    # float32 1 adjacent to shoreline, 0 otherwise
    datum: str             # e.g. EGM2008; never silently treated as a tidal datum
    source: str            # pinned preparation manifest path

@dataclass(frozen=True)
class AtmosphereForcing:
    u10: DeviceArray        # earth east, m/s
    v10: DeviceArray        # earth north, m/s
    psfc: DeviceArray       # absolute pressure, Pa
    ustar: DeviceArray      # friction velocity, m/s
    tau_x: DeviceArray      # stress on water, east, N/m2
    tau_y: DeviceArray      # stress on water, north, N/m2
    rho_air: DeviceArray    # kg/m3
    time_seconds: float

@dataclass(frozen=True)
class WaveOutputs:
    hs: DeviceArray         # significant wave height, m
    tp: DeviceArray         # peak period, s; zero for calm cells
    direction: DeviceArray  # propagation towards, degrees clockwise from north; zero when calm

@dataclass(frozen=True)
class SurgeOutputs:
    eta: DeviceArray        # water level above datum, m; dry points may store bed elevation
    inundation: DeviceArray # max(eta-elevation,0) on initially dry land, m

def validate_plane(array: DeviceArray, shape: tuple[int,int], name: str) -> None:
    try:
        a = array.__cuda_array_interface__
    except AttributeError as exc:
        raise TypeError(f"{name} must be a CUDA array; host copies are forbidden") from exc
    if tuple(a["shape"]) != shape or a["typestr"] not in ("<f4", "=f4", "|f4"):
        raise ValueError(f"{name} must be float32 with shape {shape}")
    strides=a.get("strides")
    if strides is not None and tuple(strides) != (shape[1]*4,4):
        raise ValueError(f"{name} must be C-contiguous")
    if not a["data"][0] or a["data"][1]:
        raise ValueError(f"{name} needs a non-null writable device buffer")

def validate_fields(value: Any, grid: OceanGrid) -> None:
    items=({f.name:getattr(value,f.name) for f in fields(value)} if is_dataclass(value)
           else value if isinstance(value,Mapping) else vars(value))
    for name,item in items.items():
        if name not in ("datum", "source", "time_seconds"):
            validate_plane(item,grid.shape,name)

@runtime_checkable
class WaveSolver(Protocol):
    def set_forcing(self, forcing: AtmosphereForcing) -> None: ...
    def set_water_level(self, eta: DeviceArray) -> None: ...
    def step(self, dt: float) -> None: ...
    @property
    def outputs(self) -> WaveOutputs: ...

@runtime_checkable
class SurgeSolver(Protocol):
    def set_forcing(self, forcing: AtmosphereForcing) -> None: ...
    def set_waves(self, waves: WaveOutputs) -> None: ...
    def step(self, dt: float) -> None: ...
    @property
    def outputs(self) -> SurgeOutputs: ...

# Both sibling modules export create_solver(grid, bathymetry, *, config).
# config is a mapping of solver-specific scalar options. No case adapters.
class SolverFactory(Protocol):
    def __call__(self, grid: OceanGrid, bathymetry: Bathymetry, *,
                 config: Mapping[str, Any]) -> WaveSolver | SurgeSolver: ...
