"""Milbrandt-Yau double-moment microphysics (WRF ``mp_physics = 9``).

The classic tornado/hail-literature two-moment scheme: mass AND number are
prognostic for all six hydrometeors, with graupel and hail carried as
SEPARATE categories rather than one rimed-ice slot selected by a switch
(the Morrison mp=10 arrangement).  Six masses (qc, qr, qi, qs, qg, qh) and
six numbers (nc, nr, ni, ns, ng, nh) are transported.

Transcription authority: the byte-frozen WRF v4.6.1 Fortran at
``phys/module_mp_milbrandt2mom.F`` -- ``mp_milbrandt2mom_main`` :841-3485,
the sedimentation trio :564-836, the helper functions :31-433 and
:3489-3525, and the 3-D wrapper ``mp_milbrandt2mom_driver`` :3559-3703.
The staging, the fixed switches and every documented divergence live in
``woof/core/kernels/milbrandt2.cu``'s header; this module is the launcher
and the :class:`~woof.core.state.DomainState` adapter.

WHAT THE WRAPPER FIXES.  ``mp_milbrandt2mom_driver`` hard-codes CCNtype=2
(continental), and precipDiag/sedi/warmphase/autoconv/icephase/snow all ON
(:3612-3623); the scheme body hard-codes snowSpherical=.false.,
primIceNucl=1 (Meyers + contact) and grpl/hail/rainAccr/iceDep ON
(:1168-1176).  Those are the only mp=9 identity WRF can produce, so woof
compiles them in and ``woof/config.py`` refuses by name any request to
move them (the MYNN pinning pattern).

SURFACE PRESSURE.  The wrapper feeds the scheme ``p_sfc = p8w(i,kms,j)``
(:3646) and the scheme rebuilds its working pressure as ``PS*sigma`` where
``sigma = p/p_sfc`` (:3648, :1216).  ``p8w(i,1,j)`` is WRF's
linear-in-height extrapolation of the FULL EOS pressure to the surface,
``w1*p(1) + w2*p(2)`` with ``w1 = (z_at_w(1)-z(2))/(z(1)-z(2))``
(moist_physics_prep_em, module_big_step_utilities_em.F:5566-5574) --
reproduced in :func:`_surface_pressure` from the same geopotential the
adapter already builds.  The sigma round trip is NOT the identity in FP32
and it is the pressure every later statement uses, so it is reproduced
rather than short-circuited to ``state.p``.

EVIDENCE.  Column smoke against the shipped seams plus float64
self-consistency; NO oracle comparison against the WRF Fortran has been
run.  The registry row says exactly that.
"""

from __future__ import annotations

import cupy as cp
import numpy as np

from woof.config import RunConfig
from woof.core import constants as c
from woof.core.kernels import get_kernel
from woof.core.milbrandt2_constants import ck_vector_device
from woof.core.state import DTYPE, DomainState

_CELL_TPB = 64
_COLUMN_TPB = 32
_SHALLOW_KMAX = 64
_KMAX = 256

#: Fewest / most vertical levels the CUDA translation unit accepts.  The
#: floor is 3: ``moist_physics_prep_em``'s surface extrapolation reads two
#: mass levels and the sedimentation flux divergence reads ``k+1`` up to
#: ``ktop+1``, which needs a level above the highest active one.
VERTICAL_LEVEL_BOUNDS = (3, _KMAX)

#: The six mass and six number moments the scheme prognoses, in the order
#: ``woof/core/moist.py`` transports them.
MASS_SPECIES = ("qc", "qr", "qi", "qs", "qg", "qh")
NUMBER_SPECIES = ("nc", "nr", "ni", "ns", "ng", "nh")

def _constants_device() -> cp.ndarray:
    """The read-only FP32 constant vector, uploaded once per process.

    The cache itself lives in :mod:`woof.core.milbrandt2_constants`, with
    the table, because the mixed nest edge into mp=9 needs the same vector
    from a module the RW-WPS preparation wheel stages and this one is not.
    """
    return ck_vector_device()


def _surface_pressure(pressure, z_half, z_at_w0, out):
    """WRF ``p8w(i,1,j)``, module_big_step_utilities_em.F:5566-5574.

    ``z0`` is the surface interface height, ``z1``/``z2`` the lowest two
    mass-level heights, and the weights are the linear-in-z extrapolation
    ``w1 = (z0-z2)/(z1-z2)``, ``w2 = 1-w1`` applied to the FULL pressure.
    """
    z0 = z_at_w0
    z1 = z_half[0]
    z2 = z_half[1]
    w1 = (z0 - z2) / (z1 - z2)
    w2 = DTYPE(1.0) - w1
    cp.add(w1 * pressure[0], w2 * pressure[1], out=out)
    return out


def launch_milbrandt2(
        t, qv, qc, qr, qi, qs, qg, qh, nc, nr, ni, ns, ng, nh,
        w, pressure, psfc, dt: float,
        rainnc, rainncv, snownc, snowncv, graupelnc, graupelncv,
        hailnc, hailncv, sr, zet,
        *, pres, de, ide, dz, idz, gamfact, qsw, qsi,
        qc_in, qr_in, nc_in, nr_in) -> None:
    """Run one WRF Milbrandt-Yau call over an FP32 column batch.

    ``t`` is ABSOLUTE temperature (WRF's ``t2d = th*pii``, :3644), not
    theta: the scheme updates T and the wrapper converts back at :3671.
    Every 3-D argument is ``(nz, ny, nx)`` C-contiguous float32; the
    surface fields are ``(ny, nx)``.  ``t``, the twelve moments and ``qv``
    are updated in place, as are the eight precipitation accumulators,
    ``sr`` and ``zet``.  The keyword arguments are the caller-owned scratch
    volumes the six kernels hand to one another -- they are part of the
    contract because the adapter parks them in named ``DomainState`` slots
    so a run's allocation is budgeted, not per-step.

    ``w`` is WRF's ``WZ``: the vertical velocity on MASS levels, which the
    microphysics driver receives as ``grid%w_2`` indexed ``w(i,k,j)`` over
    ``kts:kte`` (module_microphysics_driver.F:1848).
    """
    shape = t.shape
    if len(shape) != 3:
        raise ValueError(f"Milbrandt-Yau fields must be 3-D, got {shape}")
    nz, ny, nx = shape
    minimum, maximum = VERTICAL_LEVEL_BOUNDS
    if nz > maximum:
        raise ValueError(f"nz={nz} exceeds MY2_KMAX={maximum}")
    if nz < minimum:
        raise ValueError(f"Milbrandt-Yau requires nz >= {minimum}, got {nz}")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")

    volumes = {
        "qv": qv, "qc": qc, "qr": qr, "qi": qi, "qs": qs, "qg": qg,
        "qh": qh, "nc": nc, "nr": nr, "ni": ni, "ns": ns, "ng": ng,
        "nh": nh, "w": w, "pressure": pressure, "zet": zet,
        "pres": pres, "de": de, "ide": ide, "dz": dz, "idz": idz,
        "gamfact": gamfact, "qsw": qsw, "qsi": qsi,
        "qc_in": qc_in, "qr_in": qr_in, "nc_in": nc_in, "nr_in": nr_in,
    }
    for name, value in volumes.items():
        if value.shape != shape:
            raise ValueError(
                f"{name} must have shape {shape}, got {value.shape}")
        if value.dtype != DTYPE:
            raise TypeError(f"{name} must be float32, got {value.dtype}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
    surface_shape = (ny, nx)
    for name, value in (
            ("psfc", psfc), ("rainnc", rainnc), ("rainncv", rainncv),
            ("snownc", snownc), ("snowncv", snowncv),
            ("graupelnc", graupelnc), ("graupelncv", graupelncv),
            ("hailnc", hailnc), ("hailncv", hailncv), ("sr", sr)):
        if value.shape != surface_shape or value.dtype != DTYPE:
            raise ValueError(f"{name} must be float32 {surface_shape}, "
                             f"got {value.dtype} {value.shape}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")

    ck = _constants_device()
    ncell = nz * ny * nx
    ncol = ny * nx
    cell_blocks = (ncell + _CELL_TPB - 1) // _CELL_TPB
    column_blocks = (ncol + _COLUMN_TPB - 1) // _COLUMN_TPB
    dims = (np.int32(nz), np.int32(ny), np.int32(nx))

    prelim = get_kernel("milbrandt2", "milbrandt2_prelim")
    geometry = get_kernel("milbrandt2", "milbrandt2_geometry")
    cold = get_kernel("milbrandt2", "milbrandt2_cold")
    warm = get_kernel("milbrandt2", "milbrandt2_warm")
    sediment = get_kernel(
        "milbrandt2",
        "milbrandt2_sediment_64" if nz <= _SHALLOW_KMAX
        else "milbrandt2_sediment_256")
    diagnostics = get_kernel("milbrandt2", "milbrandt2_diagnostics")

    prelim((cell_blocks,), (_CELL_TPB,),
           (t, qv, qc, qr, qi, qs, qg, qh, nc, nr, ni, ns, ng, nh,
            pressure, psfc, pres, de, ide, gamfact, qsw, qsi,
            qc_in, qr_in, nc_in, nr_in, ck) + dims)
    geometry((cell_blocks,), (_CELL_TPB,),
             (pres, psfc, de, dz, idz) + dims)
    cold((cell_blocks,), (_CELL_TPB,),
         (t, qv, qc, qr, qi, qs, qg, qh, nc, nr, ni, ns, ng, nh,
          w, pres, de, ide, gamfact, qsw, qsi, ck, DTYPE(dt)) + dims)
    warm((cell_blocks,), (_CELL_TPB,),
         (t, qv, qc, qr, qi, nc, nr, ni, w, pres, de, ide, qsw,
          qc_in, qr_in, nc_in, nr_in, ck, DTYPE(dt)) + dims)
    sediment((column_blocks,), (_COLUMN_TPB, 5),
             (t, qv, qc, qr, qi, qs, qg, qh, nc, nr, ni, ns, ng, nh,
              de, ide, dz, idz, gamfact,
              rainnc, rainncv, snownc, snowncv, graupelnc, graupelncv,
              hailnc, hailncv, sr, ck, DTYPE(dt)) + dims, shared_mem=5 * _COLUMN_TPB * 4)
    diagnostics((cell_blocks,), (_CELL_TPB,),
                (t, qv, qc, qr, qi, qs, qg, qh, nc, nr, ni, ns, ng, nh,
                 pres, zet, ck) + dims)


def apply(state: DomainState, cfg: RunConfig, dt: float, *,
          refl_10cm_due: bool = False):
    """Prepare WRF's inputs and apply Milbrandt-Yau to ``state`` in place.

    The scheme owns REFL_10CM natively: the WRF driver binds ``Zet`` to
    ``refl_10cm`` (module_microphysics_driver.F:1878) rather than calling a
    separate radar operator, so on a history step the stash is fed the
    scheme's own dBZ field instead of woof's generic reflectivity
    diagnostic.
    """
    from woof.core.microphysics import (MicrophysicsDiagnostics,
                                         moist_physics_finish,
                                         save_pre_mp_theta)

    nz, ny, nx = state.p.shape
    required = MASS_SPECIES + NUMBER_SPECIES
    missing = [name for name in required
               if getattr(state, name, None) is None]
    if missing:
        raise ValueError("mp_physics=9 state lacks Milbrandt-Yau fields: "
                         + ", ".join(missing))

    thb = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
    phb = state.phb if state.phb.ndim == 3 else state.phb[:, None, None]
    theta = state.scratch((nz, ny, nx), "my2_theta")
    pii = state.scratch((nz, ny, nx), "my2_pii")
    temperature = state.scratch((nz, ny, nx), "my2_t")
    z8w = state.scratch((nz + 1, ny, nx), "my2_z8w")
    zh = state.scratch((nz, ny, nx), "my2_z")
    psfc = state.scratch((ny, nx), "my2_psfc")

    theta[...] = (state.total_theta() if getattr(state, "use_theta_m", 0)
               else thb + state.thp)
    pii[...] = cp.power(state.p / DTYPE(c.P0), DTYPE(c.RCP))
    # WRF's t2d, :3644.  The scheme integrates absolute temperature and the
    # wrapper divides by pii again at :3671.
    temperature[...] = theta * pii
    z8w[...] = (phb + state.php) / DTYPE(c.G)
    zh[...] = 0.5 * (z8w[:nz] + z8w[1:])
    _surface_pressure(state.p, zh, z8w[0], psfc)

    # WRF hands the microphysics driver grid%w_2 indexed kts:kte -- the
    # LOWER full-level slice, not a mass-level average.  This is the same
    # reading the Thompson adapter takes (woof/core/microphysics.py).
    w_mass = state.w[:nz]

    surface = (ny, nx)
    rainnc = state.scratch(surface, "mp_rainnc")
    rainncv = state.scratch(surface, "mp_rainncv")
    snownc = state.scratch(surface, "mp_snownc")
    snowncv = state.scratch(surface, "mp_snowncv")
    graupelnc = state.scratch(surface, "mp_graupelnc")
    graupelncv = state.scratch(surface, "mp_graupelncv")
    hailnc = state.scratch(surface, "mp_hailnc")
    hailncv = state.scratch(surface, "mp_hailncv")
    sr = state.scratch(surface, "mp_sr")
    # Zet is an INOUT dummy of the driver bound straight to refl_10cm, so
    # the scheme writes the persistent slot on every call, history step or
    # not; woof keeps the same slot the generic operator uses.
    zet = state.scratch((nz, ny, nx), "refl_10cm")

    save_pre_mp_theta(state)
    launch_milbrandt2(
        temperature, state.qv, state.qc, state.qr, state.qi, state.qs,
        state.qg, state.qh, state.nc, state.nr, state.ni, state.ns,
        state.ng, state.nh, w_mass, state.p, psfc, dt,
        rainnc, rainncv, snownc, snowncv, graupelnc, graupelncv,
        hailnc, hailncv, sr, zet,
        pres=state.scratch((nz, ny, nx), "my2_pres"),
        de=state.scratch((nz, ny, nx), "my2_de"),
        ide=state.scratch((nz, ny, nx), "my2_ide"),
        dz=state.scratch((nz, ny, nx), "my2_dz"),
        idz=state.scratch((nz, ny, nx), "my2_idz"),
        gamfact=state.scratch((nz, ny, nx), "my2_gamfact"),
        qsw=state.scratch((nz, ny, nx), "my2_qsw"),
        qsi=state.scratch((nz, ny, nx), "my2_qsi"),
        qc_in=state.scratch((nz, ny, nx), "my2_qc_in"),
        qr_in=state.scratch((nz, ny, nx), "my2_qr_in"),
        nc_in=state.scratch((nz, ny, nx), "my2_nc_in"),
        nr_in=state.scratch((nz, ny, nx), "my2_nr_in"),
    )
    if refl_10cm_due:
        # No generic operator call: the scheme already filled the slot, so
        # the history frame gets the SCHEME's dBZ.  Stashing the same array
        # keeps the one-frame handoff contract (refl.py:565-577) intact.
        from woof.core.refl import stash_refl_10cm
        stash_refl_10cm(state, zet)
    # :3671 -- back to potential temperature on the unchanged Exner.
    theta[...] = temperature / pii
    moist_physics_finish(state, cfg, theta, dt)
    return MicrophysicsDiagnostics(
        rainnc=rainnc, rainncv=rainncv, sr=sr,
        snownc=snownc, snowncv=snowncv,
        graupelnc=graupelnc, graupelncv=graupelncv,
        hailnc=hailnc, hailncv=hailncv)


def reflectivity(state: DomainState, *, temperature=None, pressure=None):
    """H_Z(x): the scheme's own 10 cm reflectivity, as a pure diagnostic.

    The Z block of ``mp_milbrandt2mom_main``'s final diagnostics
    (module_mp_milbrandt2mom.F:3400-3466) reads T, pressure, five
    precipitating masses and their five number moments and writes Zet.  It
    updates nothing, so unlike P3's fused ``k_loop_final_diagnostics`` it
    is a real observation operator and is launched here as one
    (``woof/core/kernels/milbrandt2_zet.cu``) -- the same relationship
    ``woof.da.obsop`` has with NSSL's ``radardd02``.

    ``state``'s number moments are per unit mass, the convention a
    ``DomainState`` carries between steps; the kernel converts with the
    scheme's own ``de``.  ``temperature``/``pressure`` are the
    microphysics-time pair when a caller has it; omitting both diagnoses
    temperature from the state on the state's own pressure.

    Returns the state's persistent ``refl_10cm`` scratch slot.
    """

    nz, ny, nx = state.p.shape
    missing = [name for name in ("qr", "qi", "qs", "qg", "qh",
                                 "nr", "ni", "ns", "ng", "nh")
               if getattr(state, name, None) is None]
    if missing:
        raise ValueError(
            "the Milbrandt-Yau reflectivity operator needs the scheme's own "
            "state: missing " + ", ".join(missing))
    if (temperature is None) != (pressure is None):
        raise ValueError("temperature and pressure must be supplied together")
    if temperature is None:
        thb = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
        t = state.scratch((nz, ny, nx), "my2_zet_t")
        t[...] = (state.total_theta() if getattr(state, "use_theta_m", 0)
                  else thb + state.thp) * cp.power(
            state.p / DTYPE(c.P0), DTYPE(c.RCP))
        pres = state.p
    else:
        t = temperature
        pres = pressure
    zet = state.scratch((nz, ny, nx), "refl_10cm")
    ncell = nz * ny * nx
    blocks = (ncell + _CELL_TPB - 1) // _CELL_TPB
    get_kernel("milbrandt2_zet", "milbrandt2_zet")(
        (blocks,), (_CELL_TPB,), (
            t, state.qr, state.qi, state.qs, state.qg, state.qh,
            state.nr, state.ni, state.ns, state.ng, state.nh,
            pres, zet, _constants_device(), np.int32(0),
            np.int32(nz), np.int32(ny), np.int32(nx)))
    return zet
