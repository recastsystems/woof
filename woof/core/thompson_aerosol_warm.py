"""Launchers for the aerosol-aware (``mp_physics=28``) warm source network.

This module owns three launches:

``launch_aerosol_warm_source_network``
    The ambient-warm half of WRF's source phase, with the droplet
    distribution driven by prognostic ``nc`` instead of the constant
    ``Nt_c``.  It writes hydrometeor mass/number state exactly as
    ``woof.core.thompson.launch_warm_frozen_source_network`` does, and
    writes the mp=28 droplet/aerosol tendencies **only** into the three
    shared scratch accumulators.

``launch_ncten_balance``
    WRF's cloud water mass/number balance (``module_mp_thompson.F``
    2996-3019).  It is a separate launch on purpose: WRF applies it ONCE
    per column after all ``ncten`` sources.  The adapter must call it
    between the warm network and the saturation adjustment, and must never
    call it from inside a network launcher.

``probe_warm_rates``
    Non-mutating per-cell readback of every individual WP-07 rate.  It
    exists so ``tests/test_thompson_aerosol_warm_gpu.py`` can gate each rate
    against a Fortran oracle rather than only observing their summed effect.

``probe_warm_frozen_rates``
    The same, for WRF's ALWAYS-RUN frozen collection block
    (``module_mp_thompson.F``:2402-2471).  ``iiwarm`` is a PARAMETER
    ``.false.`` (:59), so snow and graupel collect cloud water and scavenge
    aerosol at ambient-warm levels too -- in a MELTING LAYER -- and six of
    those rates (pnc_scw, pnc_gcw, pna_sca, pnd_scd, pna_gca, pnd_gcd) are
    new in mp=28 with no mp=8 counterpart to inherit validation from.  The
    oracle is ``tools/thompson_wrf461_oracle/probe_warm_frozen_aero.F90``,
    committed and regenerable via ``build_probe_warm_frozen.sh``.

``probe_frozen_constants``
    Reads back the REAL(4) graupel exponents this translation unit derives,
    so they can be compared against the digits that same Fortran program
    prints rather than against a second transcription.

Nothing here imports from :mod:`woof.core.thompson`; that module is
read-only for this port.  Validation helpers come from
:mod:`woof.core.thompson_aerosol_launch`.
"""

from __future__ import annotations

import numpy as np

from woof.core.thompson_aerosol_launch import (
    aerosol_kernel, hail_graupel_arguments)
from woof.core.state import DTYPE
from woof.core.thompson_aerosol_launch import (
    WARM_MODULE,
    launch_grid,
    add_optional_fields,
    validate_fields,
    validate_fp64_fortran_table,
)

#: WRF's ordered rain/snow collision table names (module_mp_thompson.F
#: qr_acr_qsV2 cache), matching ``woof.core.thompson_runtime``'s
#: ``DeviceClassicColdSourceTables.rain_snow_tables`` order exactly.
RAIN_SNOW_TABLE_NAMES = (
    "tcs_racs1", "tmr_racs1", "tcs_racs2", "tmr_racs2",
    "tcr_sacr1", "tms_sacr1", "tcr_sacr2", "tms_sacr2",
    "tnr_racs1", "tnr_racs2", "tnr_sacr1", "tnr_sacr2",
)

#: WRF's ordered rain/graupel collision table names (qr_acr_qg_V4 cache).
RAIN_GRAUPEL_TABLE_NAMES = (
    "tcg_racg", "tmr_racg", "tcr_gacr", "tnr_racg", "tnr_gacr",
)

RAIN_SNOW_TABLE_SHAPE = (37, 9, 37, 37)
RAIN_GRAUPEL_TABLE_SHAPE = (37, 37, 1, 37, 37)
#: The operational WRF 3.9 fork's shapes (thompson_version = "wrf_39_noaa",
#: woof.core.thompson_contract.FORK_GENERATED_TABLE_FILES).
WRF39_RAIN_SNOW_TABLE_SHAPE = (28, 9, 37, 37)
WRF39_RAIN_GRAUPEL_TABLE_SHAPE = (28, 28, 37, 37)
#: The hail-aware scheme's nine-slab shape (mp_physics=38,
#: qr_acr_qg_mp38V1.dat, module_mp_thompson.F:4117-4125, :4192-4273).
HAIL_RAIN_GRAUPEL_TABLE_SHAPE = (37, 37, 9, 37, 37)


def _rain_snow_shape():
    from woof.core.thompson_aerosol_launch import active_thompson_version
    return (WRF39_RAIN_SNOW_TABLE_SHAPE
            if active_thompson_version() == "wrf_39_noaa"
            else RAIN_SNOW_TABLE_SHAPE)


def _rain_graupel_shape():
    from woof.core.thompson_aerosol_launch import (
        active_hail_aware, active_thompson_version)
    if active_hail_aware():
        return HAIL_RAIN_GRAUPEL_TABLE_SHAPE
    return (WRF39_RAIN_GRAUPEL_TABLE_SHAPE
            if active_thompson_version() == "wrf_39_noaa"
            else RAIN_GRAUPEL_TABLE_SHAPE)
EFFICIENCY_TABLE_SHAPE = (100, 100)


def _arrays_overlap(left, right) -> bool:
    """Return whether two contiguous device arrays share any storage.

    Transcribed from ``woof/core/thompson.py:22-48``; that module is
    read-only for this port and ``_arrays_overlap`` is a private name there.
    """
    if left is right:
        return True

    def interval(value):
        interface = getattr(value, "__cuda_array_interface__", None)
        address_space = "cuda"
        if interface is None:
            interface = getattr(value, "__array_interface__", None)
            address_space = "host"
        if interface is None:
            return None
        pointer = int(interface["data"][0] or 0)
        nbytes = int(value.nbytes)
        return address_space, pointer, pointer + nbytes

    left_interval = interval(left)
    right_interval = interval(right)
    if left_interval is None or right_interval is None:
        return False
    left_space, left_start, left_end = left_interval
    right_space, right_start, right_end = right_interval
    return (left_space == right_space
            and left_start < right_end
            and right_start < left_end)


def _validate_dt(dt: float) -> None:
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")


def _validate_collision_tables(rain_snow_tables, rain_graupel_tables):
    rain_snow_values = tuple(rain_snow_tables)
    rain_graupel_values = tuple(rain_graupel_tables)
    if len(rain_snow_values) != len(RAIN_SNOW_TABLE_NAMES):
        raise ValueError(
            "rain_snow_tables must contain "
            f"{len(RAIN_SNOW_TABLE_NAMES)} arrays, "
            f"got {len(rain_snow_values)}")
    if len(rain_graupel_values) != len(RAIN_GRAUPEL_TABLE_NAMES):
        raise ValueError(
            "rain_graupel_tables must contain "
            f"{len(RAIN_GRAUPEL_TABLE_NAMES)} arrays, "
            f"got {len(rain_graupel_values)}")
    for name, table in zip(
            RAIN_SNOW_TABLE_NAMES, rain_snow_values, strict=True):
        validate_fp64_fortran_table(name, table, _rain_snow_shape())
    for name, table in zip(
            RAIN_GRAUPEL_TABLE_NAMES, rain_graupel_values, strict=True):
        validate_fp64_fortran_table(
            name, table, _rain_graupel_shape())
    return rain_snow_values, rain_graupel_values


def launch_aerosol_warm_source_network(
        qc, qr, nr, qs, qg, graupel_number_shadow,
        graupel_melt_marker, snow_melt_marker,
        temperature, pressure, qv,
        nc_entry, nwfa_entry, nifa_entry,
        ncten, nwfaten, nifaten,
        rain_cloud_efficiency, snow_cloud_efficiency,
        rain_snow_tables, rain_graupel_tables, dt: float, *,
        qcten=None, qrten=None, nrten=None, qvten=None,
        tten=None, qsten=None, qgten=None, ngten=None, graupel_volume=None, qbten=None) -> None:
    """Apply WRF-ordered warm-level sources with a prognostic droplet number.

    ``nc_entry``/``nwfa_entry``/``nifa_entry`` are the FROZEN per-kilogram
    entry state for the whole mp=28 call and are never written.  The droplet
    and aerosol tendencies are ADDED to ``ncten``/``nwfaten``/``nifaten``
    (per kilogram per second), which a terminal state kernel applies once
    with WRF's clamps.  Hydrometeor mass, rain number, graupel number,
    vapour and temperature are updated in place exactly as the mp=8 warm
    network updates them.

    ``graupel_melt_marker`` must arrive carrying the held ``T >= 273.15 K``
    entry mask; the kernel consumes it and overwrites it with WRF's held
    ``prr_gml > 0`` decision.  ``snow_melt_marker`` receives the independent
    ``prr_sml > 0`` decision.

    This launcher does NOT run the ncten balance limiter.  Call
    :func:`launch_ncten_balance` once, after this and after the cold
    network, and before the saturation adjustment.

    ``qcten`` (keyword, optional) is WRF's per-kilogram-per-second cloud
    water accumulator.  Given, ``qc`` is the read-only entry cloud and the
    stage's cloud tendency is ADDED to ``qcten``, to be applied once with
    every other cloud tendency (:3975); the production adapter passes it.
    Left ``None``, the tendency is applied to ``qc`` in place.

    ``qvten``/``tten`` (keywords, both or neither) are WRF's vapour and
    temperature accumulators.  Given, the sources are added to them in
    WRF's form (:2982, :3174-3179), ``qv`` stays the entry vapour and
    ``temperature`` becomes ``t1d + DT*tten``.
    """
    fields = {
        "qc": qc,
        "qr": qr,
        "nr": nr,
        "qs": qs,
        "qg": qg,
        "graupel_number_shadow": graupel_number_shadow,
        "graupel_melt_marker": graupel_melt_marker,
        "snow_melt_marker": snow_melt_marker,
        "temperature": temperature,
        "pressure": pressure,
        "qv": qv,
        "nc_entry": nc_entry,
        "nwfa_entry": nwfa_entry,
        "nifa_entry": nifa_entry,
        "ncten": ncten,
        "nwfaten": nwfaten,
        "nifaten": nifaten,
    }
    if qcten is not None:
        fields["qcten"] = qcten
    add_optional_fields(fields, qrten=qrten, nrten=nrten)
    add_optional_fields(fields, qvten=qvten, tten=tten)
    add_optional_fields(fields, qsten=qsten, qgten=qgten, ngten=ngten)
    hail_arguments = hail_graupel_arguments(
        fields, graupel_volume=graupel_volume, qgten=qgten, ngten=ngten,
        qbten=qbten)
    _, size = validate_fields(fields)
    if qcten is not None and _arrays_overlap(qc, qcten):
        raise ValueError("qcten must not alias qc; the entry cloud is "
                         "read-only while the accumulator carries it")
    if _arrays_overlap(graupel_melt_marker, snow_melt_marker):
        raise ValueError(
            "snow_melt_marker must not alias graupel_melt_marker")
    # The accumulator contract is the whole point of this package: an
    # accumulator aliasing its own entry state would make the "read-only
    # entry state" guarantee silently false.
    for entry_name, entry in (("nc_entry", nc_entry),
                              ("nwfa_entry", nwfa_entry),
                              ("nifa_entry", nifa_entry)):
        for accum_name, accum in (("ncten", ncten),
                                  ("nwfaten", nwfaten),
                                  ("nifaten", nifaten)):
            if _arrays_overlap(entry, accum):
                raise ValueError(
                    f"{accum_name} must not alias {entry_name}; mp=28 entry "
                    "state is read-only for the whole call")
    for name, accum in (("nwfaten", nwfaten), ("nifaten", nifaten)):
        if _arrays_overlap(ncten, accum):
            raise ValueError(f"{name} must not alias ncten")
    if _arrays_overlap(nwfaten, nifaten):
        raise ValueError("nifaten must not alias nwfaten")
    validate_fp64_fortran_table(
        "rain_cloud_efficiency", rain_cloud_efficiency,
        EFFICIENCY_TABLE_SHAPE)
    validate_fp64_fortran_table(
        "snow_cloud_efficiency", snow_cloud_efficiency,
        EFFICIENCY_TABLE_SHAPE)
    rain_snow_values, rain_graupel_values = _validate_collision_tables(
        rain_snow_tables, rain_graupel_tables)
    _validate_dt(dt)

    # Smaller blocks admit more resident warps without changing cell arithmetic.
    grid, block = launch_grid(size, threads=64)
    aerosol_kernel(WARM_MODULE, "thompson_aa_warm_source_network")(
        grid, block,
        (qc, qr, nr, qs, qg, graupel_number_shadow,
         graupel_melt_marker, snow_melt_marker,
         temperature, pressure, qv,
         nc_entry, nwfa_entry, nifa_entry,
         ncten, nwfaten, nifaten,
         rain_cloud_efficiency, snow_cloud_efficiency,
         *rain_snow_values, *rain_graupel_values,
         qcten, qrten, nrten, qvten, tten, qsten, qgten, ngten,
         graupel_volume, qbten,
         DTYPE(dt), np.int32(size)))


def launch_aerosol_warm_source_network_from_owner(
        qc, qr, nr, qs, qg, graupel_number_shadow,
        graupel_melt_marker, snow_melt_marker,
        temperature, pressure, qv,
        nc_entry, nwfa_entry, nifa_entry,
        ncten, nwfaten, nifaten,
        table_owner, dt: float, *, qcten=None, qrten=None,
        nrten=None, qvten=None, tten=None, qsten=None, qgten=None,
        ngten=None, graupel_volume=None, qbten=None) -> None:
    """Launch the warm network from one verified classic table owner.

    mp=28 reuses the four classic Thompson caches unchanged; only
    ``CCN_ACTIVATE.BIN`` is new, and the warm network never reads it.
    """
    from woof.core.thompson_runtime import DeviceClassicTableSet

    if (not isinstance(table_owner, DeviceClassicTableSet)
            or not table_owner.roundtrip_verified):
        raise TypeError(
            "table_owner must be a verified DeviceClassicTableSet")
    tables = table_owner.cold_source_tables
    launch_aerosol_warm_source_network(
        qc, qr, nr, qs, qg, graupel_number_shadow,
        graupel_melt_marker, snow_melt_marker,
        temperature, pressure, qv,
        nc_entry, nwfa_entry, nifa_entry,
        ncten, nwfaten, nifaten,
        tables.rain_cloud_efficiency, table_owner.t_Efsw,
        tables.rain_snow_tables, tables.rain_graupel_tables, dt,
        qcten=qcten, qrten=qrten, nrten=nrten, qvten=qvten, tten=tten,
        qsten=qsten, qgten=qgten, ngten=ngten,
        graupel_volume=graupel_volume, qbten=qbten)


def launch_ncten_balance(
        qc_entry, qc_after, nc_entry, density, ncten, dt: float) -> None:
    """Apply WRF's cloud water mass/number balance ONCE.

    ``module_mp_thompson.F:2996-3019``.  Keeps the mass-weighted mean droplet
    size between 1 and 100 microns and the total below ``Nt_c_max``, by
    OVERWRITING ``ncten`` wherever a clamp fires -- the limiter backs the
    tendency out against ``nc1d*rho``, it does not add to it.

    Sequencing is part of the contract.  WRF runs this once per column after
    every ``ncten`` source has been accumulated (autoconversion, accretion,
    Bigg freezing, snow riming, graupel riming), and before the saturation
    adjustment's ``pnc_wcd``.  Calling it from inside both the warm and the
    cold network double-applies it: the result stays finite and plausible and
    is wrong, which no per-kernel unit test would catch.

    ``qc_entry`` is the frozen entry cloud mixing ratio, ``qc_after`` the
    current one (WRF's ``qc1d + qcten*dtsave``), and ``density`` the ENTRY
    air density (``module_mp_thompson.F:1802``) -- not a density rediagnosed
    from the mutated temperature and vapour.
    """
    _, size = validate_fields({
        "qc_entry": qc_entry,
        "qc_after": qc_after,
        "nc_entry": nc_entry,
        "density": density,
        "ncten": ncten,
    })
    if _arrays_overlap(ncten, nc_entry):
        raise ValueError(
            "ncten must not alias nc_entry; mp=28 entry state is read-only "
            "for the whole call")
    if _arrays_overlap(ncten, qc_entry) or _arrays_overlap(ncten, qc_after):
        raise ValueError("ncten must not alias the cloud water fields")
    if _arrays_overlap(qc_entry, qc_after):
        raise ValueError(
            "qc_after must not alias qc_entry; the limiter needs both the "
            "entry and the post-source cloud mass")
    _validate_dt(dt)

    grid, block = launch_grid(size)
    aerosol_kernel(WARM_MODULE, "thompson_aa_ncten_balance")(
        grid, block,
        (qc_entry, qc_after, nc_entry, density, ncten,
         DTYPE(dt), np.int32(size)))


def probe_warm_rates(pressure, temperature, qv, qc, nc_entry, qr, nr_entry,
                     nwfa_entry, nifa_entry, rain_cloud_efficiency,
                     dt: float):
    """Read back every WP-07 rate per cell without mutating any state.

    Returns a dict keyed by WRF's own rate names.  The distribution
    quantities (``nc_m3``, ``nu_c``, ``lamc``, ``mvd_c``, ``xDc``,
    ``nr_m3``, ``lamr``, ``mvd_r``, ``N0_r``) come first because a
    disagreement there explains every rate downstream of it.

    ``lamc``/``lamr``/``N0_r`` and every ``p*`` rate are float64 because WRF
    declares them DOUBLE PRECISION.
    """
    import cupy as cp

    shape, size = validate_fields({
        "pressure": pressure,
        "temperature": temperature,
        "qv": qv,
        "qc": qc,
        "nc_entry": nc_entry,
        "qr": qr,
        "nr_entry": nr_entry,
        "nwfa_entry": nwfa_entry,
        "nifa_entry": nifa_entry,
    })
    validate_fp64_fortran_table(
        "rain_cloud_efficiency", rain_cloud_efficiency,
        EFFICIENCY_TABLE_SHAPE)
    _validate_dt(dt)

    float_names = ("nc_m3", "mvd_c", "xDc", "nwfa_m3", "nifa_m3",
                   "nr_m3", "mvd_r")
    double_names = ("lamc", "lamr", "N0_r", "prr_wau", "pnr_wau",
                    "pnc_wau", "prr_rcw", "pnc_rcw", "pnr_rcr",
                    "pna_rca", "pnd_rcd")
    out = {name: cp.empty(shape, dtype=DTYPE) for name in float_names}
    out["nu_c"] = cp.empty(shape, dtype=cp.int32)
    for name in double_names:
        out[name] = cp.empty(shape, dtype=cp.float64)

    grid, block = launch_grid(size)
    aerosol_kernel(WARM_MODULE, "thompson_aa_probe_warm_rates")(
        grid, block,
        (pressure, temperature, qv, qc, nc_entry, qr, nr_entry,
         nwfa_entry, nifa_entry, rain_cloud_efficiency,
         out["nc_m3"], out["nu_c"], out["lamc"], out["mvd_c"],
         out["xDc"], out["nwfa_m3"], out["nifa_m3"], out["nr_m3"],
         out["lamr"], out["mvd_r"], out["N0_r"],
         out["prr_wau"], out["pnr_wau"], out["pnc_wau"],
         out["prr_rcw"], out["pnc_rcw"], out["pnr_rcr"],
         out["pna_rca"], out["pnd_rcd"],
         DTYPE(dt), np.int32(size)))
    return out


#: Order of the doubles ``thompson_aa_probe_frozen_constants`` writes.  These
#: are WRF's REAL(4) graupel exponents and prefactors; the Fortran oracle
#: prints the same quantities as ``WP07F_*`` so the comparison is against
#: WRF's own digits, never a second transcription.
FROZEN_CONSTANT_NAMES = (
    "bv_g", "cge6", "cge9", "cge11",
    "cgg6", "cgg9", "cgg11", "mvdg_num",
    "am_g", "ogg3", "t1_qs_qc", "t1_qg_qc",
)


def probe_frozen_constants():
    """Read back the REAL(4) graupel exponents this translation unit uses.

    Returns a plain dict of Python floats keyed by
    :data:`FROZEN_CONSTANT_NAMES`.  ``cge(9,idx_bg1)`` is 3.8899998664855957,
    NOT the decimal 3.89: WRF builds it by REAL(4) arithmetic from a REAL
    ``bv_g`` array slot (``module_mp_thompson.F``:149-150, :463-464, :759),
    and ``ilamg**cge(9)`` moves by up to 1.5e-6 relative between the two --
    against a 2e-6 end-to-end fixture gate.
    """
    import cupy as cp

    out = cp.empty(len(FROZEN_CONSTANT_NAMES), dtype=cp.float64)
    aerosol_kernel(WARM_MODULE, "thompson_aa_probe_frozen_constants")(
        (1,), (32,), (out,))
    values = cp.asnumpy(out)
    return dict(zip(FROZEN_CONSTANT_NAMES,
                    (float(v) for v in values), strict=True))


#: Fields ``probe_warm_frozen_rates`` returns as float32 (WRF REAL) and as
#: float64 (WRF DOUBLE PRECISION), in the order the kernel writes them.
_FROZEN_FLOAT_NAMES = (
    "rho", "rhof", "visco", "twet", "nc_m3", "mvd_c",
    "nwfa_m3", "nifa_m3", "xDs", "smoe",
)
_FROZEN_DOUBLE_HEAD = ("ilamg", "N0_g")
_FROZEN_FLOAT_TAIL = ("xDg", "vtg", "stoke_g", "Ef_sw", "Ef_gw")
_FROZEN_RATE_NAMES = (
    "prs_scw", "pnc_scw", "prg_gcw", "pnc_gcw",
    "pna_sca", "pnd_scd", "pna_gca", "pnd_gcd",
)


def probe_warm_frozen_rates(
        pressure, temperature, qv, qc, nc_entry, qs, qg, ng_entry,
        nwfa_entry, nifa_entry, rain_cloud_efficiency,
        snow_cloud_efficiency, dt: float):
    """Read back WRF's always-run frozen collection block, per cell.

    ``module_mp_thompson.F``:2402-2471, evaluated with no temperature guard
    because ``iiwarm`` is a PARAMETER ``.false.`` (:59).  Six of the eight
    returned rates -- ``pnc_scw``, ``pnc_gcw``, ``pna_sca``, ``pnd_scd``,
    ``pna_gca``, ``pnd_gcd`` -- are new in mp=28 and have no mp=8 counterpart
    to inherit validation from; the two mass companions ``prs_scw`` and
    ``prg_gcw`` come back with them because WRF diagnoses all eight from one
    snow and one graupel distribution.

    The kernel calls the SAME device function
    (``thompson_aa_frozen_collect_rates``) that
    :func:`launch_aerosol_warm_source_network` calls, so a disagreement here
    is by construction a disagreement in the network.

    Nothing is mutated.  ``ng_entry`` is graupel number per kilogram, WRF's
    ``ng1d``; the intermediates ``twet``, ``xDs``, ``smoe``, ``ilamg``,
    ``N0_g``, ``xDg``, ``vtg``, ``stoke_g``, ``Ef_sw`` and ``Ef_gw`` are
    returned so a failure localizes to one WRF line.
    """
    import cupy as cp

    shape, size = validate_fields({
        "pressure": pressure,
        "temperature": temperature,
        "qv": qv,
        "qc": qc,
        "nc_entry": nc_entry,
        "qs": qs,
        "qg": qg,
        "ng_entry": ng_entry,
        "nwfa_entry": nwfa_entry,
        "nifa_entry": nifa_entry,
    })
    validate_fp64_fortran_table(
        "rain_cloud_efficiency", rain_cloud_efficiency,
        EFFICIENCY_TABLE_SHAPE)
    validate_fp64_fortran_table(
        "snow_cloud_efficiency", snow_cloud_efficiency,
        EFFICIENCY_TABLE_SHAPE)
    _validate_dt(dt)

    out = {}
    for name in _FROZEN_FLOAT_NAMES + _FROZEN_FLOAT_TAIL:
        out[name] = cp.empty(shape, dtype=DTYPE)
    for name in _FROZEN_DOUBLE_HEAD + _FROZEN_RATE_NAMES:
        out[name] = cp.empty(shape, dtype=cp.float64)

    ordered = (
        [out[name] for name in _FROZEN_FLOAT_NAMES]
        + [out[name] for name in _FROZEN_DOUBLE_HEAD]
        + [out[name] for name in _FROZEN_FLOAT_TAIL]
        + [out[name] for name in _FROZEN_RATE_NAMES])

    grid, block = launch_grid(size)
    aerosol_kernel(WARM_MODULE, "thompson_aa_probe_warm_frozen_rates")(
        grid, block,
        (pressure, temperature, qv, qc, nc_entry, qs, qg, ng_entry,
         nwfa_entry, nifa_entry,
         rain_cloud_efficiency, snow_cloud_efficiency,
         *ordered, DTYPE(dt), np.int32(size)))
    return out


__all__ = [
    "EFFICIENCY_TABLE_SHAPE",
    "FROZEN_CONSTANT_NAMES",
    "RAIN_GRAUPEL_TABLE_NAMES",
    "RAIN_GRAUPEL_TABLE_SHAPE",
    "RAIN_SNOW_TABLE_NAMES",
    "RAIN_SNOW_TABLE_SHAPE",
    "launch_aerosol_warm_source_network",
    "launch_aerosol_warm_source_network_from_owner",
    "launch_ncten_balance",
    "probe_frozen_constants",
    "probe_warm_frozen_rates",
    "probe_warm_rates",
]
