"""Aerosol-aware Thompson (``mp_physics=28``) fallout and phase cleanup.

Two launchers, both backed by ``woof/core/kernels/thompson_aerosol_sed.cu``:

``launch_aa_cloud_sedimentation``
    WRF's cloud-water fallout with the droplet-NUMBER channel that classic
    Thompson does not have at all (module_mp_thompson.F:3644-3666 and
    :3823-3838).

``launch_aa_final_phase_cleanup``
    The two instantaneous phase transfers at :3943-3966, made
    number-conserving: melted cloud ice hands its number to the droplet
    population and homogeneously frozen cloud water hands its number back.

Neither writes ``state.nc``.  Both write the shared per-kilogram-per-second
accumulator ``ncten``, which WP-04's terminal state kernel applies once with
WRF's clamps (:3972-4021).  That split is the whole accumulator contract: WRF
clamps the droplet number exactly once per call, and four independent clamps
would be a silent physics change no unit test would flag.

WHAT THIS MODULE DELIBERATELY DOES NOT DO
-----------------------------------------
* It does not sediment rain, ice, snow or graupel.  The mp=28 adapter calls
  ``woof.core.thompson``'s launchers for those, byte-for-byte unchanged;
  module_mp_thompson.F:3790-3936 contains no ``is_aerosol_aware`` branch and
  no nc/nwfa/nifa reference.
* It does not sediment aerosol.  ``nwfa`` and ``nifa`` have NO fallout term
  anywhere in module_mp_thompson.F.  Any implementation that adds one is
  wrong.
* It does not substep.  WRF's cloud fallout is a single pass with no
  ``onstep`` factor, unlike every other species (see the kernel header).
* It accumulates nothing at the surface.  Cloud mass and number leaving the
  lowest level are discarded by WRF, so a number budget over this launcher
  does not close.  It closes against the two END FLUXES, which is what
  ``tests/test_thompson_aerosol_sed_gpu.py`` asserts instead.

END-TO-END STATUS
-----------------
These two launchers are exercised inside a complete mp_physics=28 column call
-- eighteen launchers in WRF's driver order -- against the committed
``aero-nc-sed`` (109) and ``aero-reduces-to-classic`` (119) fixtures.  The
driver lives in the test module because
``woof/core/microphysics_aerosol.py::_apply_thompson_aerosol`` (WP-09) does
not exist yet; when it does, the test should call it and the local driver
should be deleted.

MEASURED (RTX 5090, cupy 14.1.1): aero-nc-sed reproduces WRF BIT-EXACTLY in
qv, qc, qi, qs, qg, ni, nwfa, nifa, temperature, all three effective radii and
RAINNC, and to 1.2e-6 in qr/nr/nc.  aero-reduces-to-classic reproduces WRF to
9.5e-7 or better everywhere except one level of qr/nr; see the test module for
that level's localisation.  Both measurements require the shared saturation
fit in ``thompson_aerosol_common.cuh`` to be contraction-pinned; without that
one change all twenty cloud-free levels of aero-nc-sed condense spurious water
and lose up to 5.6 percent of the column CCN, because the FMA-contracted fit is
one ulp low and WRF opens its condensation block on ``ssatw > 1.E-15``
(:3400, with eps declared at :185).
"""

from __future__ import annotations

import numpy as np

from woof.core import thompson as _classic
from woof.core.thompson_aerosol_launch import aerosol_kernel
from woof.core.state import DTYPE
from woof.core.thompson_aerosol_launch import (
    DEFAULT_THREADS,
    SED_MODULE,
    launch_grid,
    validate_fields,
)

#: Column-kernel launch width.  Matches ``woof/core/thompson.py:16`` so the
#: aerosol and classic sedimentation passes tile identically.
_COLUMN_TPB = 32

#: Template instantiations in thompson_aerosol_sed.cu.  These mirror
#: ``THOMPSON_AA_KMAX_SHALLOW`` / ``THOMPSON_AA_KMAX_GENERIC``.
_SHALLOW_KMAX = 64
_KMAX = 256

# The serial host oracle cannot execute a block barrier. Its installer selects
# the retained column kernels; every device launch uses the parallel route.
LEVEL_PARALLEL_FALLOUT = True

#: Advertised vertical extent, matching ``thompson.VERTICAL_LEVEL_BOUNDS``.
VERTICAL_LEVEL_BOUNDS = (2, _KMAX)

#: Diagnostic outputs ``launch_aa_cloud_sedimentation`` can fill, in the order
#: the kernel takes them.  Part of the published API; do not reorder.
DIAGNOSTIC_FIELDS = (
    "mass_velocity",
    "number_velocity",
    "cloud_mass",
    "cloud_number",
)


def _validate_surface_mask(name, value, surface_shape) -> None:
    if value.shape != surface_shape:
        raise ValueError(
            f"{name} must have shape {surface_shape}, got {value.shape}")
    if value.dtype != DTYPE:
        raise TypeError(f"{name} must be float32, got {value.dtype}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")


def launch_aa_cloud_sedimentation(
        qc, cloud_number_entry, cloud_number_tendency,
        temperature, pressure, qv, vertical_velocity, dz,
        dt: float, *, reference_density,
        rain_active_columns=None, cloud_active_columns=None,
        diagnostics=None, qcten=None) -> None:
    """Apply WRF's number-weighted cloud-water fallout.

    Parameters
    ----------
    qc
        Cloud water mixing ratio, read-modify-written in place exactly as the
        frozen mp=8 launcher does.
    cloud_number_entry
        ``nc1d``: the per-kilogram droplet number as it entered the
        microphysics call.  READ ONLY.

        NOT raw ``state.nc``.  module_mp_thompson.F:1844-1846 rewrites the
        caller's own column on the way in::

            else                    ! qc1d(k) .le. R1
               qc1d(k) = 0.0
               nc1d(k) = 0.0

        so the entry droplet number every later block reads is the state array
        ZEROED wherever cloud water was absent at call entry.  Handing in the
        raw array instead gives a non-zero working droplet number in air that
        has no droplets, which stays bounded and never trips a health check.
    cloud_number_tendency
        ``ncten``: the shared per-kilogram-per-second accumulator.  Read for
        the working droplet number and then added to.
    reference_density
        WRF's HELD pre-adjustment density -- the value ``rc`` and ``nc`` are
        both formed on at :3216/:3486, before ``rho`` is refreshed at :3489.
        It is REQUIRED here, unlike in the classic launcher: forming the
        droplet number on the refreshed density instead is an invisible error
        that leaves every bound intact.
    rain_active_columns, cloud_active_columns
        WRF's post-source ``ANY(L_qr)`` and held ``ANY(L_qc)`` column guards,
        with the same meaning and the same ordering constraint as
        ``woof.core.thompson.launch_cloud_sedimentation``.
    diagnostics
        Optional mapping of names in :data:`DIAGNOSTIC_FIELDS` to float32
        arrays shaped like ``qc``.  Supplying any of them selects a separate
        kernel entry point; the physics is identical and the unrequested
        outputs are passed as null pointers.  This exists so the oracle gate
        can pin WRF's intermediate columns rather than only the endpoints, and
        it is not used by the forecast adapter.

        * ``mass_velocity`` -- WRF's ``vtck``.
        * ``number_velocity`` -- WRF's ``vtnck``, which has no mp=8
          counterpart.
        * ``cloud_mass`` -- the working ``rc`` BEFORE the fallout is applied,
          i.e. the value ``sed_c`` is built from.
        * ``cloud_number`` -- the working ``nc`` AFTER the fallout is applied,
          i.e. WRF's :3835 with its floor of 10.
    qcten
        WRF's cloud water accumulator, per kilogram per second.  Given, ``qc``
        is the read-only entry cloud, the working ``rc`` is formed from
        ``qc + qcten*dt`` and the fallout tendency is added to ``qcten``
        (:3832).  Only the production entry points (``cloud_active_columns``
        given, no diagnostics) carry it.
    """
    shape, _ = validate_fields({
        "qc": qc,
        "cloud_number_entry": cloud_number_entry,
        "cloud_number_tendency": cloud_number_tendency,
        "temperature": temperature,
        "pressure": pressure,
        "qv": qv,
        "reference_density": reference_density,
        "vertical_velocity": vertical_velocity,
        "dz": dz,
    })
    if len(shape) != 3:
        raise ValueError(
            f"Thompson aerosol cloud fields must be 3-D, got {shape}")
    nz, ny, nx = shape
    if nz < 2 or nz > _KMAX:
        raise ValueError(
            f"Thompson aerosol cloud sedimentation requires 2 <= nz <= "
            f"{_KMAX}, got {nz}")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")

    if qcten is not None:
        if cloud_active_columns is None or diagnostics:
            raise ValueError(
                "qcten is carried only by the production entry point: pass "
                "cloud_active_columns and no diagnostics")
        validate_fields({"qc": qc, "qcten": qcten})

    surface_shape = (ny, nx)
    if rain_active_columns is not None:
        _validate_surface_mask(
            "rain_active_columns", rain_active_columns, surface_shape)
    if cloud_active_columns is not None:
        if rain_active_columns is None:
            raise ValueError(
                "cloud_active_columns requires rain_active_columns")
        _validate_surface_mask(
            "cloud_active_columns", cloud_active_columns, surface_shape)

    diagnostic_arguments = None
    if diagnostics:
        unknown = set(diagnostics) - set(DIAGNOSTIC_FIELDS)
        if unknown:
            raise ValueError(
                f"unknown sedimentation diagnostics {sorted(unknown)}; "
                f"expected a subset of {list(DIAGNOSTIC_FIELDS)}")
        supplied = {name: diagnostics[name] for name in DIAGNOSTIC_FIELDS
                    if diagnostics.get(name) is not None}
        if supplied:
            validate_fields({"qc": qc, **supplied})
            diagnostic_arguments = tuple(
                diagnostics.get(name) for name in DIAGNOSTIC_FIELDS)

    suffix = "64" if nz <= _SHALLOW_KMAX else "256"
    if diagnostic_arguments is not None:
        kernel_name = f"thompson_aa_cloud_sediment_{suffix}_diagnostic"
    elif cloud_active_columns is not None:
        kernel_name = f"thompson_aa_cloud_sediment_{suffix}_with_masks"
    elif rain_active_columns is not None:
        kernel_name = f"thompson_aa_cloud_sediment_{suffix}_with_rain"
    else:
        kernel_name = f"thompson_aa_cloud_sediment_{suffix}"

    arguments = (qc, cloud_number_entry, cloud_number_tendency,
                 temperature, pressure, qv, reference_density)
    if diagnostic_arguments is not None:
        arguments += (rain_active_columns, cloud_active_columns)
    else:
        if rain_active_columns is not None:
            arguments += (rain_active_columns,)
        if cloud_active_columns is not None:
            arguments += (cloud_active_columns,)
    arguments += (vertical_velocity, dz)
    if diagnostic_arguments is not None:
        arguments += diagnostic_arguments
    if cloud_active_columns is not None and diagnostic_arguments is None:
        arguments += (qcten,)
    arguments += (DTYPE(dt), np.int32(nz), np.int32(ny), np.int32(nx))

    ncol = ny * nx
    # The classic launchers' switch (woof.core.thompson.
    # LEVEL_PARALLEL_FALLOUT): eight columns a block, one thread per level,
    # for the production entry point; the diagnostic ones keep the column
    # kernel.
    if (_classic.LEVEL_PARALLEL_FALLOUT and nz <= _SHALLOW_KMAX
            and cloud_active_columns is not None
            and diagnostic_arguments is None):
        aerosol_kernel(SED_MODULE, "thompson_aa_cloud_sediment_levels_64_with_masks")(
            ((ncol + 7) // 8,), (8, _SHALLOW_KMAX), arguments)
        return
    blocks = (ncol + _COLUMN_TPB - 1) // _COLUMN_TPB
    aerosol_kernel(SED_MODULE, kernel_name)(
        (blocks,), (_COLUMN_TPB,), arguments)


def launch_aa_final_phase_cleanup(
        qc, qi, ni, temperature, cloud_number_entry, ice_number_entry,
        cloud_number_tendency, pressure, qv, dt: float, *,
        qcten=None, qiten=None, niten=None, tten=None, heat_ocp=None,
        heat_lvap=None, theta=None, exner=None) -> None:
    """Melt cloud ice above 0 C and freeze cloud water below HGFR.

    module_mp_thompson.F:3943-3966.  Both transfers move NUMBER as well as
    mass, which classic Thompson does not do:

    * melting credits ``ncten`` with ``ni1d(k)`` -- the ENTRY ice number, not
      the live ``ni``.  ``ice_number_entry`` therefore has to be a per-call
      snapshot taken before any source network runs; handing in ``state.ni``
      would be a plausible-looking, silently wrong droplet source.  Like
      ``cloud_number_entry`` it is the entry array with :1870-1871's zeroing
      applied, i.e. zero wherever ``qi <= R1`` at call entry.
    * freezing debits ``ncten`` by ``nc1d(k) + ncten(k)*DT``, the true running
      per-kilogram droplet number, read after the melt branch and deliberately
      unclamped.

    ``qc``, ``qi``, ``ni`` and ``temperature`` are updated in place; ``nc`` is
    not touched.  With ``qcten`` (WRF's cloud water accumulator) ``qc`` is
    the read-only entry cloud instead: the melt adds ``xri*odt`` to
    ``qcten`` (:3949), the freeze reads ``xrc`` from ``qc + qcten*dt`` and
    subtracts ``xrc*odt`` (:3955, :3962), and the R1 zeroing waits for the
    terminal apply.

    ``tten``, ``heat_ocp``, ``heat_lvap``, ``theta`` and ``exner`` (all five
    or none, with the ice accumulators): WRF's temperature accumulator and
    the ``ocp(k)``/``lvap(k)`` WRF last formed (the rain evaporation's
    export).  Given, the melt and the freeze add their latent heat to
    ``tten`` as WRF writes it (:3953, :3964) and ``temperature`` becomes
    ``t1d + tten*DT`` (:3973), ``t1d = theta*exner``.
    """
    heat = {"tten": tten, "heat_ocp": heat_ocp, "heat_lvap": heat_lvap,
            "theta": theta, "exner": exner}
    heat_given = [name for name, value in heat.items() if value is not None]
    if heat_given and len(heat_given) != 5:
        raise ValueError("tten, heat_ocp, heat_lvap, theta and exner are "
                         f"given together or not at all (got {heat_given})")
    if heat_given and qiten is None:
        raise ValueError("the temperature accumulator rides with the ice "
                         "and cloud ones")
    _, size = validate_fields({
        "qc": qc,
        "qi": qi,
        "ni": ni,
        "temperature": temperature,
        "cloud_number_entry": cloud_number_entry,
        "ice_number_entry": ice_number_entry,
        "cloud_number_tendency": cloud_number_tendency,
        "pressure": pressure,
        "qv": qv,
        **({} if qcten is None else {"qcten": qcten}),
        **({} if qiten is None else {"qiten": qiten, "niten": niten}),
        **({} if not heat_given else heat),
    })
    if (qiten is None) != (niten is None):
        raise ValueError("qiten and niten are given together or not at all")
    if qiten is not None and qcten is None:
        raise ValueError("the ice accumulators ride with the cloud one")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    grid, block = launch_grid(size, DEFAULT_THREADS)
    aerosol_kernel(SED_MODULE, "thompson_aa_final_phase_cleanup")(
        grid, block,
        (qc, qi, ni, temperature, cloud_number_entry, ice_number_entry,
         cloud_number_tendency, pressure, qv, qcten, qiten, niten,
         tten, heat_ocp, heat_lvap, theta, exner,
         DTYPE(dt), np.int32(size)))


def launch_aa_rain_sedimentation_accumulate(
        qr1d, nr1d, qrten, nrten, temperature, pressure, qv, dz,
        rainnc, rainncv, dt: float, *, reference_density,
        accumulate_surface: bool = True,
        export_surface: bool = False) -> None:
    """WRF's rain fallout in tendency form (:3611-3640, :3790-3812).

    ``qr1d``/``nr1d`` are the read-only entry rain; the fallout is added to
    ``qrten``/``nrten`` and nothing is applied (the terminal apply,
    :func:`woof.core.thompson_aerosol_state.launch_terminal_rain_ice`, does
    that once, with the size bounds).  ``reference_density`` is the rain
    evaporation's level-wise export (zero: no L_qr; negative: the :3568
    rewrite on that density; positive: the :3193 TAU+1 density).  The surface
    bookkeeping is the classic launcher's, unless ``export_surface``: then
    the column's pptrain alone goes into ``rainncv`` and ``rainnc`` is not
    touched, for :func:`launch_aa_surface_precipitation`.
    """
    shape, _ = validate_fields({
        "qr1d": qr1d, "nr1d": nr1d, "qrten": qrten, "nrten": nrten,
        "temperature": temperature, "pressure": pressure, "qv": qv,
        "reference_density": reference_density, "dz": dz,
    })
    nz, ny, nx = shape
    if nz < 2 or nz > _KMAX:
        raise ValueError(f"rain fallout requires 2 <= nz <= {_KMAX}, got {nz}")
    _validate_surface_mask("rainnc", rainnc, (ny, nx))
    _validate_surface_mask("rainncv", rainncv, (ny, nx))
    _column_launch("thompson_aa_rain_sediment_accumulate", nz, ny, nx,
                   (qr1d, nr1d, qrten, nrten, temperature, pressure, qv,
                    reference_density, dz, rainnc, rainncv,
                    np.int32(2 if export_surface
                             else 1 if accumulate_surface else 0), DTYPE(dt),
                    np.int32(nz), np.int32(ny), np.int32(nx)))


def launch_aa_ice_sedimentation_accumulate(
        qi1d, ni1d, qiten, niten, temperature, pressure, qv, dz,
        rainnc, rainncv, snownc, snowncv, dt: float, *, reference_density,
        rain_active_columns, export_surface: bool = False) -> None:
    """WRF's ice fallout in tendency form (:3664-3698, :3838-3870).

    The working pair is :3226-3233's on the held :3193 density
    (``reference_density``); rhof is refreshed from the current density
    only in columns ``rain_active_columns`` marks (WRF's ANY(L_qr)).  The
    surface bookkeeping is the classic launcher's, unless ``export_surface``:
    then the column's pptice alone goes into ``snowncv`` and nothing else is
    written, for :func:`launch_aa_surface_precipitation`.
    """
    shape, _ = validate_fields({
        "qi1d": qi1d, "ni1d": ni1d, "qiten": qiten, "niten": niten,
        "temperature": temperature, "pressure": pressure, "qv": qv,
        "reference_density": reference_density, "dz": dz,
    })
    nz, ny, nx = shape
    if nz < 2 or nz > _KMAX:
        raise ValueError(f"ice fallout requires 2 <= nz <= {_KMAX}, got {nz}")
    for name, value in (("rainnc", rainnc), ("rainncv", rainncv),
                        ("snownc", snownc), ("snowncv", snowncv),
                        ("rain_active_columns", rain_active_columns)):
        _validate_surface_mask(name, value, (ny, nx))
    _column_launch("thompson_aa_ice_sediment_accumulate", nz, ny, nx,
                   (qi1d, ni1d, qiten, niten, temperature, pressure, qv,
                    reference_density, rain_active_columns, dz, rainnc,
                    rainncv, snownc, snowncv,
                    np.int32(1 if export_surface else 0), DTYPE(dt),
                    np.int32(nz), np.int32(ny), np.int32(nx)))


def launch_aa_snow_sedimentation(
        qs, temperature, pressure, qv, dz, snow_precipitation, dt: float, *,
        reference_density, reference_temperature, snow_melt_marker,
        velocity_boost, qr1d, nr1d, qrten, nrten, rain_density,
        qsten=None) -> None:
    """WRF v4.6.1's snow fallout (:3257-3262, :3313-3353, :3699-3733,
    :3871-3902) and its terminal zero (:4054-4055).

    ``qs`` is the post-source snow the networks leave in place; the fallout
    tendency times DT is added to it.  ``reference_density`` and
    ``reference_temperature`` are the :3193 density and :3188 temperature.
    The melting blend reads the rain fall-speed pass's own vtrk and rr, from
    the entry rain, its tendency so far and ``rain_density`` (the rain
    evaporation's export, which also carries ANY(L_qr) for rhof).  The
    column's pptsnow goes into ``snow_precipitation``.

    ``qsten`` (keyword): WRF's snow accumulator as the source networks left
    it.  Given, ``qs`` is the entry snow qs1d, the fallout forms its working
    snow as qs1d + qsten*DT, adds its own tendency to ``qsten`` and applies
    the sum once (:4054); ``None`` keeps the in-place form above.
    """
    if qsten is not None:
        validate_fields({"qs": qs, "qsten": qsten})
    shape, _ = validate_fields({
        "qs": qs, "temperature": temperature, "pressure": pressure,
        "qv": qv, "dz": dz, "reference_density": reference_density,
        "reference_temperature": reference_temperature,
        "snow_melt_marker": snow_melt_marker,
        "velocity_boost": velocity_boost, "qr1d": qr1d, "nr1d": nr1d,
        "qrten": qrten, "nrten": nrten, "rain_density": rain_density,
    })
    nz, ny, nx = shape
    if nz < 2 or nz > _KMAX:
        raise ValueError(f"snow fallout requires 2 <= nz <= {_KMAX}, got {nz}")
    _validate_surface_mask("snow_precipitation", snow_precipitation, (ny, nx))
    _column_launch("thompson_aa_snow_sediment", nz, ny, nx,
                   (qs, snow_melt_marker, qr1d, nr1d, qrten, nrten,
                    rain_density, temperature, pressure, qv,
                    reference_density, reference_temperature, velocity_boost,
                    dz, snow_precipitation, qsten, DTYPE(dt),
                    np.int32(nz), np.int32(ny), np.int32(nx)))


def launch_aa_graupel_sedimentation(
        qg, graupel_number, temperature, pressure, qv, dz,
        graupel_precipitation, dt: float, *, reference_density,
        active_columns, rain_density, qgten=None, ngten=None) -> None:
    """WRF v4.6.1's classic graupel fallout (:3283-3303, :3370-3376,
    :3740-3773, :3903-3937).

    ``qg`` and ``graupel_number`` (the private ng1d) are the post-source
    values in place; the fallout tendencies times DT are added to them, and
    :func:`woof.core.thompson_aerosol_state.launch_aa_graupel_number_finalize`
    applies :4058-4077 after the phase cleanup.  ``active_columns`` is
    WRF's ANY(L_qg); ``rain_density`` carries ANY(L_qr) for rhof.  The
    column's pptgraul goes into ``graupel_precipitation``.

    ``qgten``/``ngten`` (keywords, both or neither): WRF's graupel and
    graupel-number accumulators as the source networks left them.  Given,
    ``qg`` and ``graupel_number`` are the entry qg1d and ng1d, the fallout
    forms its working graupel as qg1d + qgten*DT, adds its own tendencies to
    the accumulators and applies the sums (:4058-4059); in a column with no
    fallout the sources' tendencies are applied alone.
    """
    if (qgten is None) != (ngten is None):
        raise ValueError("qgten and ngten are given together or not at all")
    if qgten is not None:
        validate_fields({"qg": qg, "qgten": qgten, "ngten": ngten})
    shape, _ = validate_fields({
        "qg": qg, "graupel_number": graupel_number,
        "temperature": temperature, "pressure": pressure, "qv": qv,
        "dz": dz, "reference_density": reference_density,
        "rain_density": rain_density,
    })
    nz, ny, nx = shape
    if nz < 2 or nz > _KMAX:
        raise ValueError(
            f"graupel fallout requires 2 <= nz <= {_KMAX}, got {nz}")
    for name, value in (("active_columns", active_columns),
                        ("graupel_precipitation", graupel_precipitation)):
        _validate_surface_mask(name, value, (ny, nx))
    _column_launch("thompson_aa_graupel_sediment", nz, ny, nx,
                   (qg, graupel_number, rain_density, temperature, pressure,
                    qv, reference_density, dz, active_columns,
                    graupel_precipitation, qgten, ngten, DTYPE(dt),
                    np.int32(nz), np.int32(ny), np.int32(nx)))


def launch_aa_surface_precipitation(rainnc, rainncv, snownc, snowncv,
                                    graupelnc, graupelncv, sr) -> None:
    """mp_gt_driver :1294-1308 in the driver's order of addition.

    On entry ``sr`` holds pptrain, ``rainncv`` pptsnow, ``snowncv`` pptice
    and ``graupelncv`` pptgraul (the four fallout passes' exports); on exit
    all seven hold WRF's surface fields, SR included.
    """
    surface = {"rainnc": rainnc, "rainncv": rainncv, "snownc": snownc,
               "snowncv": snowncv, "graupelnc": graupelnc,
               "graupelncv": graupelncv, "sr": sr}
    shape = rainnc.shape
    for name, value in surface.items():
        _validate_surface_mask(name, value, shape)
    ncol = int(np.prod(shape))
    grid, block = launch_grid(ncol)
    aerosol_kernel(SED_MODULE, "thompson_aa_surface_precipitation")(
        grid, block, (rainnc, rainncv, snownc, snowncv, graupelnc,
                      graupelncv, sr, np.int32(ncol)))


__all__ = [
    "DIAGNOSTIC_FIELDS",
    "VERTICAL_LEVEL_BOUNDS",
    "launch_aa_cloud_sedimentation",
    "launch_aa_graupel_sedimentation",
    "launch_aa_ice_sedimentation_accumulate",
    "launch_aa_rain_sedimentation_accumulate",
    "launch_aa_snow_sedimentation",
    "launch_aa_surface_precipitation",
    "launch_aa_final_phase_cleanup",
]


# ---------------------------------------------------------------------------
# The operational WRF 3.9 fork's ice and graupel fallout (thompson_version =
# "wrf_39_noaa").  thompson_aerosol_sed.cu carries them under
# THOMPSON_AA_WRF39; see the comment above thompson_aa_wrf39_ice_sediment_impl.
# ---------------------------------------------------------------------------

def _require_wrf39(what: str) -> None:
    from woof.core.thompson_aerosol_launch import active_thompson_version
    version = active_thompson_version()
    if version != "wrf_39_noaa":
        raise RuntimeError(
            f"{what} is the WRF 3.9 fork's kernel; the active Thompson "
            f"generation is {version!r}")


def _column_launch(name_stem: str, nz: int, ny: int, nx: int, args) -> None:
    if nz < 2 or nz > _KMAX:
        raise ValueError(f"fallout needs 2 <= nz <= {_KMAX}, got {nz}")
    if LEVEL_PARALLEL_FALLOUT and nz <= _SHALLOW_KMAX and name_stem in (
            "thompson_aa_snow_sediment", "thompson_aa_graupel_sediment"):
        aerosol_kernel(SED_MODULE, name_stem + "_levels_64")(
            (ny * nx,), (64,), args)
        return
    name = name_stem + ("_64" if nz <= _SHALLOW_KMAX else "_256")
    blocks = (ny * nx + _COLUMN_TPB - 1) // _COLUMN_TPB
    aerosol_kernel(SED_MODULE, name)((blocks,), (_COLUMN_TPB,), args)


def launch_wrf39_ice_sedimentation(
        qi, ni, temperature, pressure, qv, dz,
        rainnc, rainncv, snownc, snowncv, dt: float, *,
        reference_density) -> None:
    """The fork's cloud-ice fallout: av_i = 1847.5, the 499.D3 ceiling."""
    _require_wrf39("launch_wrf39_ice_sedimentation")
    shape, _ = validate_fields({
        "qi": qi, "ni": ni, "temperature": temperature,
        "pressure": pressure, "qv": qv, "dz": dz,
        "reference_density": reference_density})
    nz, ny, nx = shape
    for name, value in (("rainnc", rainnc), ("rainncv", rainncv),
                        ("snownc", snownc), ("snowncv", snowncv)):
        _validate_surface_mask(name, value, (ny, nx))
    _column_launch("thompson_aa_wrf39_ice_sediment", nz, ny, nx, (
        qi, ni, temperature, pressure, qv, reference_density, dz,
        rainnc, rainncv, snownc, snowncv,
        DTYPE(dt), np.int32(nz), np.int32(ny), np.int32(nx)))


def launch_wrf39_graupel_sedimentation(
        qg, graupel_intercept, temperature, pressure, qv, dz,
        rainnc, rainncv, graupelnc, graupelncv, dt: float, *,
        reference_density, melt_rain_qr, melt_rain_nr, melt_rain_density,
        active_columns) -> None:
    """The fork's graupel fallout from its post-source column intercept,
    floored at the rain fall speed above 0 C.  Accumulates the surface
    totals, as the classic graupel pass does on the coupled adapter."""
    _require_wrf39("launch_wrf39_graupel_sedimentation")
    shape, _ = validate_fields({
        "qg": qg, "graupel_intercept": graupel_intercept,
        "temperature": temperature, "pressure": pressure, "qv": qv,
        "dz": dz, "reference_density": reference_density,
        "melt_rain_qr": melt_rain_qr, "melt_rain_nr": melt_rain_nr,
        "melt_rain_density": melt_rain_density})
    nz, ny, nx = shape
    for name, value in (("rainnc", rainnc), ("rainncv", rainncv),
                        ("graupelnc", graupelnc), ("graupelncv", graupelncv),
                        ("active_columns", active_columns)):
        _validate_surface_mask(name, value, (ny, nx))
    _column_launch("thompson_aa_wrf39_graupel_sediment", nz, ny, nx, (
        qg, graupel_intercept, temperature, pressure, qv, reference_density,
        melt_rain_qr, melt_rain_nr, melt_rain_density, dz,
        rainnc, rainncv, graupelnc, graupelncv, active_columns,
        DTYPE(dt), np.int32(nz), np.int32(ny), np.int32(nx)))


def launch_wrf39_snow_sedimentation(
        qs, temperature, pressure, qv, dz,
        rainnc, rainncv, snownc, snowncv, dt: float, *,
        reference_density, reference_temperature, snow_melt_marker,
        melt_rain_qr, melt_rain_nr, melt_rain_density, velocity_boost,
        singular_fall: bool) -> None:
    """The fork's snow fallout: surface snow above R1*10, and melting snow
    falling by the rain-share blend (``singular_fall`` False, the default
    snow fall) or by the fork's own singular expression (True)."""
    _require_wrf39("launch_wrf39_snow_sedimentation")
    shape, _ = validate_fields({
        "qs": qs, "temperature": temperature, "pressure": pressure,
        "qv": qv, "dz": dz, "reference_density": reference_density,
        "reference_temperature": reference_temperature,
        "snow_melt_marker": snow_melt_marker, "melt_rain_qr": melt_rain_qr,
        "melt_rain_nr": melt_rain_nr, "melt_rain_density": melt_rain_density,
        "velocity_boost": velocity_boost})
    nz, ny, nx = shape
    for name, value in (("rainnc", rainnc), ("rainncv", rainncv),
                        ("snownc", snownc), ("snowncv", snowncv)):
        _validate_surface_mask(name, value, (ny, nx))
    _column_launch("thompson_aa_wrf39_snow_sediment", nz, ny, nx, (
        qs, snow_melt_marker, melt_rain_qr, melt_rain_nr, temperature,
        pressure, qv, reference_density, reference_temperature,
        velocity_boost, melt_rain_density, dz,
        rainnc, rainncv, snownc, snowncv, np.int32(bool(singular_fall)),
        DTYPE(dt), np.int32(nz), np.int32(ny), np.int32(nx)))


def launch_wrf39_rain_sedimentation(
        qr, nr, temperature, pressure, qv, dz, rainnc, rainncv, dt: float,
        *, reference_density) -> None:
    """The fork's rain fallout: the v4.6.1 rain-presence pass with surface
    rain counted above R1*10 (fork :3556, audit T21).  Accumulates the
    surface totals, as the classic rain pass does on the coupled adapter;
    ``reference_density`` carries L_qr and the :3568 rewrite."""
    _require_wrf39("launch_wrf39_rain_sedimentation")
    shape, _ = validate_fields({
        "qr": qr, "nr": nr, "temperature": temperature,
        "pressure": pressure, "qv": qv, "dz": dz,
        "reference_density": reference_density})
    nz, ny, nx = shape
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    for name, value in (("rainnc", rainnc), ("rainncv", rainncv)):
        _validate_surface_mask(name, value, (ny, nx))
    _column_launch("thompson_aa_wrf39_rain_sediment", nz, ny, nx, (
        qr, nr, temperature, pressure, qv, reference_density, dz,
        rainnc, rainncv,
        DTYPE(dt), np.int32(nz), np.int32(ny), np.int32(nx)))


def launch_wrf39_warm_snow_boost(entry_warm_mask, velocity_boost) -> None:
    """fork :2151: vts_boost = 1.5 on the levels the sources found at or
    above 0 C, for the fork's singular snow fall only."""
    _require_wrf39("launch_wrf39_warm_snow_boost")
    _, size = validate_fields({"entry_warm_mask": entry_warm_mask,
                               "velocity_boost": velocity_boost})
    grid, block = launch_grid(size)
    aerosol_kernel(SED_MODULE, "thompson_aa_wrf39_warm_snow_boost")(
        grid, block, (entry_warm_mask, velocity_boost, np.int32(size)))


# ---------------------------------------------------------------------------
# The hail-aware graupel fallout (mp_physics = 38).  thompson_aerosol_sed.cu
# carries it under THOMPSON_AA_HAIL; see the comment above
# thompson_aa_hail_graupel_sediment_impl.
# ---------------------------------------------------------------------------

def _require_hail(what: str) -> None:
    from woof.core.thompson_aerosol_launch import active_hail_aware
    if not active_hail_aware():
        raise RuntimeError(
            f"{what} is the hail-aware (mp_physics=38) kernel; the active "
            "call is not hail aware")


def launch_aa_hail_graupel_sedimentation(
        qg1d, ng1d, qb1d, qgten, ngten, qbten, temperature, pressure, qv,
        dz, rainnc, rainncv, graupelnc, graupelncv, dt: float, *,
        reference_density, rain_active_columns, visco_temperature) -> None:
    """WRF's hail-aware graupel fallout in tendency form.

    :3266-3282 (the TAU+1 graupel mass, number and volume on the :3193
    density ``reference_density``), :3373-3377, the density-dependent fall
    speed of :3744-3781 and the substepped divergence of :3902-3939, with
    WRF's volume-mixing-ratio flux and operation order.  ``qg1d``/``ng1d``/``qb1d`` are the entry state as the
    entry kernel rewrote it, read-only; the fallout is added to
    ``qgten``/``ngten``/``qbten``.  ``rain_active_columns`` is WRF's
    ANY(L_qr) (rhof from the current density), ``visco_temperature`` the
    temperature WRF's visco(k) was last formed from (:3201, or :3510 where
    rain evaporated).  Surface graupel joins RAINNC/RAINNCV and
    GRAUPELNC/GRAUPELNCV.
    """
    _require_hail("launch_aa_hail_graupel_sedimentation")
    shape, _ = validate_fields({
        "qg1d": qg1d, "ng1d": ng1d, "qb1d": qb1d, "qgten": qgten,
        "ngten": ngten, "qbten": qbten, "temperature": temperature,
        "pressure": pressure, "qv": qv, "dz": dz,
        "reference_density": reference_density,
        "visco_temperature": visco_temperature})
    nz, ny, nx = shape
    for name, value in (("rainnc", rainnc), ("rainncv", rainncv),
                        ("graupelnc", graupelnc), ("graupelncv", graupelncv),
                        ("rain_active_columns", rain_active_columns)):
        _validate_surface_mask(name, value, (ny, nx))
    _column_launch("thompson_aa_hail_graupel_sediment", nz, ny, nx, (
        qg1d, ng1d, qb1d, qgten, ngten, qbten, temperature, pressure, qv,
        reference_density, rain_active_columns, visco_temperature, dz,
        rainnc, rainncv, graupelnc, graupelncv,
        DTYPE(dt), np.int32(nz), np.int32(ny), np.int32(nx)))
