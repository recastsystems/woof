"""WRF's five-layer thermal diffusion land surface (``sf_surface_physics=1``).

The column is ``phys/module_sf_slab.F`` SUBROUTINE ``SLAB1D`` of WRF v4.7.1
(byte-identical to v4.6.1),
transcribed to one CUDA thread per surface column in
``woof/core/kernels/slab.cu`` and graded word for word against the
byte-unmodified module (``tools/slab_wrf461_oracle``,
``tests/test_slab_wrf461_oracle.py``).  WRF calls it from the
``CASE (SLABSCHEME)`` arm of ``module_surface_driver.F`` (:2543-2583):

1. ``CQS2 = CQS2*MAVAIL`` (the 2 m moisture diagnostic sees the moisture
   availability);
2. ``SLAB`` with ``ROVCP=rcp``, ``DELTSM=dtbl``, ``RADIATION=(ra_lw_physics
   > 0)`` and WRF's own physical constants;
3. ``SFCEVP`` accumulation with ``DTBL`` and the shared ``ACHFX``/``ACLHF``
   accumulation with atmospheric ``DT``;
4. ``SFCDIAGS`` -- the same routine Noah's arm calls, which the driver
   already owns (``PhysicsDriver._refresh_surface_diagnostics``).

Its soil geometry is ``init_soil_depth_1`` (``share/module_soil_pre.F``):
five layers whose thicknesses double from 1 cm, centres 0.005, 0.02, 0.05,
0.11 and 0.23 m.  The deepest layer is the lower boundary and is never
integrated; real.exe sets it to TMN, the source soil temperature closest to
30 cm (:func:`woof.ingest.ruc_soil.preprocess_land_surface_soil`).

Declared divergence: the default caps the top soil layer under snow at every
nonzero IFSNOW, matching the skin cap. WRF caps that layer only at IFSNOW=1,
leaving a measured 16 K skin/soil split at IFSNOW=2. GPUWM_WRF_EXACT=1
preserves WRF behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: init_soil_depth_1 is fatal at any other count ("The SLAB thermal diffusion
#: LSM uses 5 layers.", share/module_soil_pre.F), and module_check_a_mundo.F
#: set_physics_rconfigs resolves num_soil_layers=5 for SLABSCHEME.
NUM_SOIL_LAYERS = 5

#: Values covered by the WRF column fixture. All integer values are accepted.
#: WRF caps TSLB(1) only at 1 but TSK at every nonzero value. The default
#: corrects the top-layer cap for every nonzero value; WRF-exact preserves it.
ADMITTED_IFSNOW = (0, 1, 2)

#: The 2-D surface planes the scheme adds to the driver's ``fields`` beyond
#: the shared surface/Noah inventory: GSW (the net shortwave it reads, a
#: radiation-cadence carrier), CAPG (its heat capacity, rewritten from THC
#: every call), THC (LANDUSE.TBL THERIN/100, set once by landuse_init), and
#: SFCEVP (surface_driver's accumulated evaporation, a restart-only field).
SLAB_FIELDS_2D = ("gsw", "capg", "thc", "sfcevp")

f32 = np.float32


def soil_depths() -> tuple[np.ndarray, np.ndarray]:
    """``(zs, dzs)`` exactly as ``init_soil_depth_1`` builds them, float32.

    ``dzs(1)=.01``, ``zs(1)=.5*dzs(1)``, then ``dzs(l)=2*dzs(l-1)`` and
    ``zs(l)=zs(l-1)+.5*dzs(l-1)+.5*dzs(l)`` -- every operation a REAL one.
    """
    dzs = np.zeros(NUM_SOIL_LAYERS, f32)
    zs = np.zeros(NUM_SOIL_LAYERS, f32)
    dzs[0] = f32(0.01)
    zs[0] = f32(0.5) * dzs[0]
    for layer in range(1, NUM_SOIL_LAYERS):
        dzs[layer] = f32(2) * dzs[layer - 1]
        zs[layer] = (zs[layer - 1] + f32(0.5) * dzs[layer - 1]) \
            + f32(0.5) * dzs[layer]
    return zs, dzs


#: The layer centres and thicknesses, metres.
SOIL_LAYER_CENTRES_M, SOIL_LAYER_THICKNESS_M = soil_depths()

#: WRF's share/module_model_constants.F values, folded the way gfortran
#: folds them (each a REAL PARAMETER).  The oracle writes the values WRF
#: used into its output header and the test holds these equal, bit for bit.
R_D = f32(287.0)
R_V = f32(461.6)
CP = f32(7.0) * R_D / f32(2.0)
WRF_CONSTANTS = {
    "svp1": f32(0.6112), "svp2": f32(17.67), "svp3": f32(29.65),
    "svpt0": f32(273.15), "ep2": R_D / R_V, "karman": f32(0.4),
    "eomeg": f32(7.2921e-5), "stbolt": f32(5.67051e-8),
    "xlv": f32(2.5e6), "rcp": R_D / CP, "p1000mb": f32(100000.0),
}


@dataclass(frozen=True)
class SlabParameters:
    """What one slab run integrates with, beyond its fields.

    The driver attribute restart identity binds
    (``LAND_SURFACE_PARAMETER_SOURCES[1]``): the soil geometry, the snow
    switch and the physical constants the kernel is handed.  THC itself is a
    field (``fields["thc"]``) and rides the checkpoint with the rest.
    """

    zs: tuple
    dzs: tuple
    ifsnow: int
    constants: tuple
    # Both the snow cap and converted-ice initialization depend on this arm.
    wrf_exact: bool = False
    numerics: str = "fast"

    @classmethod
    def from_config(cls, cfg) -> "SlabParameters":
        from woof.wrf_exact import ENABLED
        from woof.numerics import mode
        zs, dzs = soil_depths()
        return cls(tuple(float(v) for v in zs), tuple(float(v) for v in dzs),
                   require_admitted_ifsnow(getattr(cfg, "ifsnow", 1)),
                   tuple(sorted((name, float(value))
                                for name, value in WRF_CONSTANTS.items())),
                   wrf_exact=ENABLED, numerics=mode())

    def device_geometry(self):
        """``(zs, dzs)`` as device arrays, uploaded once per bundle."""
        cached = _DEVICE_GEOMETRY.get(id(self))
        if cached is None or cached[0] is not self:
            import cupy as cp
            cached = (self, cp.asarray(np.asarray(self.zs, f32)),
                      cp.asarray(np.asarray(self.dzs, f32)))
            _DEVICE_GEOMETRY[id(self)] = cached
        return cached[1], cached[2]


_DEVICE_GEOMETRY: dict = {}


def initial_soil_temperature(tsk, tmn, land, water_temperature):
    """``init_soil_1_real`` (share/module_soil_pre.F:1287-1340), float32.

    Land: the profile linear in depth from the skin at ``zs(1)`` to TMN at
    ``zs(5)``.  Water: every layer the water temperature (SST where the
    source has one, else TSK).  Returns ``(5, ny, nx)`` float32.
    """
    from woof.slab_init_bridge import profile
    return profile(tsk, tmn, land, water_temperature)


def require_admitted_ifsnow(ifsnow) -> int:
    if isinstance(ifsnow, (bool, np.bool_)) or not isinstance(ifsnow, (int, np.integer)):
        raise ValueError(f"ifsnow must be an integer, got {ifsnow!r}")
    return int(ifsnow)


_THREADS = 128


def _kernel():
    from woof.core.kernels import load_module
    return load_module("slab").get_function("slab_columns")


def launch_slab(fields, *, t1, qv1, p1, dt: float, ifsnow: int,
                radiation: bool, zs, dzs, constants=None) -> None:
    """Advance every column of ``fields`` one WRF surface call.

    ``fields`` carries the driver's 2-D device arrays (``flhc``, ``flqc``,
    ``psfc``, ``xland``, ``tmn``, ``hfx``, ``qfx``, ``lh``, ``tsk``,
    ``qsfc``, ``chklowq``, ``gsw``, ``glw``, ``capg``, ``thc``, ``snowc``,
    ``emiss``, ``mavail``) and ``tslb`` as ``(5, ny, nx)``; ``t1``, ``qv1``
    and ``p1`` are the lowest model level's temperature, vapour mixing ratio
    and pressure.  ``zs``/``dzs`` are device copies of :func:`soil_depths`.
    ``constants`` overrides :data:`WRF_CONSTANTS` (the oracle's planted
    one-ULP check uses it); a run never passes it.
    """
    k = WRF_CONSTANTS if constants is None else constants
    ncol = int(fields["tsk"].size)
    if fields["tslb"].shape[0] != NUM_SOIL_LAYERS:
        raise ValueError(
            f"the slab scheme integrates {NUM_SOIL_LAYERS} soil layers; TSLB "
            f"has {fields['tslb'].shape[0]}")
    ifsnow = require_admitted_ifsnow(ifsnow)
    _kernel()(
        ((ncol + _THREADS - 1) // _THREADS,), (_THREADS,),
        (t1, qv1, p1, fields["flhc"], fields["flqc"], fields["psfc"],
         fields["xland"], fields["tmn"], fields["hfx"], fields["qfx"],
         fields["lh"], fields["tsk"], fields["qsfc"], fields["chklowq"],
         fields["gsw"], fields["glw"], fields["capg"], fields["thc"],
         fields["snowc"], fields["emiss"], fields["mavail"],
         fields["tslb"], zs, dzs,
         f32(dt), f32(k["rcp"]), f32(k["xlv"]), np.int32(ifsnow),
         f32(k["svp1"]), f32(k["svp2"]), f32(k["svp3"]), f32(k["svpt0"]),
         f32(k["ep2"]), f32(k["stbolt"]), np.int32(bool(radiation)),
         f32(k["p1000mb"]), np.int64(ncol)))


__all__ = ["ADMITTED_IFSNOW", "NUM_SOIL_LAYERS", "SLAB_FIELDS_2D",
           "SOIL_LAYER_CENTRES_M", "SlabParameters",
           "SOIL_LAYER_THICKNESS_M", "WRF_CONSTANTS", "initial_soil_temperature",
           "launch_slab", "require_admitted_ifsnow", "soil_depths"]
