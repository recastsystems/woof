"""WP-06 launcher for the aerosol-aware Thompson cold network (mp=28).

One kernel, ``thompson_aa_cold_network`` in
``woof/core/kernels/thompson_aerosol_cold.cu``, covering every sub-freezing
source of WRF v4.6.1's ``module_mp_thompson.F`` with ``is_aerosol_aware``
true: iceDeMott deposition nucleation (which *replaces* Cooper), a live
``idx_IN`` into the freezeH2O tables, a live droplet bin ``idx_n``, Koop
homogeneous haze freezing, the three cold cloud-number sinks, and the four
frozen aerosol wet-scavenging rates.

Importing this module compiles nothing and does not make ``mp_physics=28``
selectable.

Ownership boundaries this module honours
----------------------------------------
* ``woof/core/thompson.py`` is read-only.  :func:`validate_fields` and the
  grid arithmetic come from :mod:`woof.core.thompson_aerosol_launch`, which
  is a transcription of the classic launcher's private helpers, not an alias.
* ``state.nc``/``state.nwfa``/``state.nifa`` are **entry state**.  This
  launcher takes them as read-only inputs and writes exclusively into the
  three per-kilogram scratch accumulators ``ncten``/``nwfaten``/``nifaten``
  that WP-04's terminal kernel applies once with WRF's clamps
  (module_mp_thompson.F:3972-4021).

Where the evidence for this kernel lives
----------------------------------------
``tests/test_thompson_aerosol_cold_gpu.py`` carries three classes of gate.
The wave-4 additions are the ones that made previously invisible things
observable, and each names the WRF line it enforces:

* ``test_production_kernel_uses_the_working_stage_nu_c`` -- drives THIS
  launcher (not the readback probe) into the ``rc > 3.44e-2 kg m^-3``
  regime where module_mp_thompson.F:1832's nu_c and :2170's disagree, and
  compares its own ``qr``/``nr``/``ncten`` against WRF.  ``nr`` is the only
  discriminator there; ``ncten`` is pinned to the ``nc*odts`` cap at :2192
  and ``prr_wau`` to ``rc*odts`` at :2190.
* ``test_two_gamma_snow_number_and_not_smo0_decides_the_koop_gate`` --
  puts two cells either side of the ``ns <= 999e3`` Koop gate at :2635 at a
  snow content where WRF's explicit two-gamma ``ns`` (:2081-2088) and the
  zeroth power-law moment ``smo0`` give OPPOSITE answers.
* ``test_cold_network_reproduces_wrfs_own_ice_koop_tendency`` and
  ``test_ice_koop_is_quantised_and_one_pressure_ulp_moves_it_one_quantum``
  -- together they attribute ``aero-ice-koop``'s remaining adapter-level
  G3 residual to one 2^-24 quantum of ``prob_h`` moved by the adapter
  harness's 1-ulp pressure reconstruction, not to this kernel.
"""

from __future__ import annotations

import numpy as np

from woof.core.thompson_aerosol_launch import aerosol_kernel
from woof.core.thompson_aerosol_launch import (
    hail_graupel_arguments,
    COLD_MODULE,
    launch_grid,
    add_optional_fields,
    validate_fields,
    validate_fp64_fortran_table,
)


#: Kernel entry point.  One kernel, no flag matrix: mp=28's production path
#: always carries cloud water and always includes the complete cold-rain
#: group, so mp=8's include_cold_rain / include_cold_cloud /
#: include_snow_rime_conversion / track_graupel_number switches have no
#: reachable "off" state here and were dropped.
COLD_NETWORK_KERNEL = "thompson_aa_cold_network"

#: Readback probe for the droplet-distribution staging and the four
#: ``:2157-2234`` number/aerosol rates.  Those quantities have no end-to-end
#: observable in the committed fixtures -- every state where the two nu_c
#: stages disagree clamps ``mvd_c`` onto ``D0c`` or ``D0r``, and ``Dc_g`` and
#: ``pnr_wau`` only switch on above ``rc = 0.01e-3`` -- so a defect there is
#: invisible to a state comparison.  This is how the wrong-stage ``nu_c``
#: survived wave 2, and it is why the probe exists.
COLD_WARM_LOOP_PROBE_KERNEL = "thompson_aa_probe_cold_warm_loop"

#: The 21 classic cold-rain coefficient records, in the exact order the
#: kernel's parameter list expects.  Reordering these silently swaps table
#: axes, which is why the names are pinned here rather than left to the
#: caller.
RAIN_SNOW_TABLE_NAMES = (
    "tcs_racs1", "tmr_racs1", "tcs_racs2", "tmr_racs2",
    "tcr_sacr1", "tms_sacr1", "tcr_sacr2", "tms_sacr2",
    "tnr_racs1", "tnr_racs2", "tnr_sacr1", "tnr_sacr2",
)
RAIN_GRAUPEL_TABLE_NAMES = (
    "tcg_racg", "tmr_racg", "tcr_gacr", "tnr_racg", "tnr_gacr",
)
RAIN_FREEZING_TABLE_NAMES = (
    "rain_to_ice_mass", "rain_to_ice_number",
    "rain_to_graupel_mass", "rain_to_graupel_number",
)
CLOUD_FREEZING_TABLE_NAMES = ("cloud_to_ice_mass", "cloud_to_ice_number")

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
#: ``(ntb_r, ntb_r1, 45, ntb_IN)``.  The last axis is ``idx_IN``; mp=8 only
#: ever reads slice 27 of it.
RAIN_FREEZING_TABLE_SHAPE = (37, 37, 45, 55)
#: ``(ntb_c, nbc, 45, ntb_IN)``.  Axis 1 is ``idx_n`` (the droplet bin, 65 in
#: mp=8) and axis 3 is ``idx_IN``; mp=28 makes both live.
CLOUD_FREEZING_TABLE_SHAPE = (37, 100, 45, 55)

ICE_PARTITION_SHAPE = (64, 55)
RAIN_CLOUD_EFFICIENCY_SHAPE = (100, 100)


def launch_aa_cold_network(
        qi, ni, qs, qg, qr, nr, qc, temperature, pressure, qv,
        nc_entry, nwfa_entry, nifa_entry,
        ncten, nwfaten, nifaten,
        graupel_number_shadow, snow_velocity_boost,
        ice_deposition_partition, ice_to_snow_mass, ice_to_snow_number,
        rain_snow_tables, rain_graupel_tables, rain_freezing_tables,
        rain_cloud_efficiency, cloud_freezing_tables,
        dt: float, *, qcten=None, qrten=None, nrten=None, qiten=None,
        niten=None, snow_cloud_efficiency=None, qvten=None,
        tten=None, qsten=None, qgten=None, ngten=None, graupel_volume=None, qbten=None) -> None:
    """Apply the complete aerosol-aware sub-freezing source group.

    Every array is float32, C-contiguous and of one common shape.  The
    coefficient tables are float64 Fortran-ordered device arrays, exactly as
    :mod:`woof.core.thompson_runtime` uploads them.

    ``nc_entry``/``nwfa_entry``/``nifa_entry`` are the *per-kilogram* entry
    state arrays and are never written.  ``ncten``/``nwfaten``/``nifaten``
    are per-kilogram-per-second accumulators; this kernel adds to them and
    nothing else applies them.

    ``graupel_number_shadow`` carries the classic wrapper's untransported
    ``ng1d``; ``snow_velocity_boost`` is WRF's ``vts_boost`` and is reset to
    1.0 for **every** cell, including warm and hydrometeor-free ones,
    because the later column sedimentation kernel consumes the whole field.

    ``qcten`` (keyword, optional) is WRF's per-kilogram-per-second cloud
    water accumulator.  Given, ``qc`` is the read-only entry cloud and the
    stage's cloud tendency is ADDED to ``qcten``, to be applied once with
    every other cloud tendency (:3975); the production adapter passes it.
    Left ``None``, the tendency is applied to ``qc`` in place.

    ``snow_cloud_efficiency`` (keyword) is WRF's ``t_Efsw`` (100, 100), the
    snow-collects-cloud-water efficiency table (:2407).  The v4.6.1
    generation reads it and refuses to run without it: the breakage it
    prevents is snow riming computed from no table at all.  The WRF 3.9
    fork generation does not read it.

    ``qvten``/``tten`` (keywords, both or neither) are WRF's vapour and
    temperature accumulators.  Given, the sources are added to them in
    WRF's form (:2982, :3164-3172), ``qv`` stays the entry vapour and
    ``temperature`` becomes ``t1d + DT*tten``.

    ``qsten``/``qgten``/``ngten`` (keywords, all three or none) are WRF's
    snow, graupel and graupel-number accumulators.  Given, the sources are
    added to them and ``qs``, ``qg`` and the graupel number stay the entry
    state; the snow and graupel fallout add their own tendencies and apply
    the sums once (:4054-4059).
    Under :func:`woof.core.thompson_aerosol_launch.hail_aware_scope`
    (mp_physics=38) ``graupel_number_shadow`` is WRF's prognostic ng1d,
    ``graupel_volume`` the entry-rewritten qb1d, both read-only, and the
    graupel sources go to ``qgten``/``ngten``/``qbten`` (all four keywords
    required, with the rain/ice accumulators).

    ``snow_cloud_efficiency`` (keyword) is WRF's ``t_Efsw`` (100, 100), the
    snow-collects-cloud-water efficiency table (:2407).  The v4.6.1
    generation reads it and refuses to run without it: the breakage it
    prevents is snow riming computed from no table at all.  The WRF 3.9
    fork generation does not read it.

    ``qvten``/``tten`` (keywords, both or neither) are WRF's vapour and
    temperature accumulators.  Given, the sources are added to them in
    WRF's form (:2982, :3164-3172), ``qv`` stays the entry vapour and
    ``temperature`` becomes ``t1d + DT*tten``.

    ``qsten``/``qgten``/``ngten`` (keywords, all three or none) are WRF's
    snow, graupel and graupel-number accumulators.  Given, the sources are
    added to them and ``qs``, ``qg`` and the graupel number stay the entry
    state; the snow and graupel fallout add their own tendencies and apply
    the sums once (:4054-4059).
    """
    from woof.core.thompson_aerosol_launch import active_thompson_version
    if snow_cloud_efficiency is None:
        if active_thompson_version() != "wrf_39_noaa":
            raise ValueError(
                "the v4.6.1 cold network needs snow_cloud_efficiency "
                "(t_Efsw); without it snow cannot rime")
        snow_cloud_efficiency = rain_cloud_efficiency
    fields = {
        "qi": qi,
        "ni": ni,
        "qs": qs,
        "qg": qg,
        "qr": qr,
        "nr": nr,
        "qc": qc,
        "temperature": temperature,
        "pressure": pressure,
        "qv": qv,
        "nc_entry": nc_entry,
        "nwfa_entry": nwfa_entry,
        "nifa_entry": nifa_entry,
        "ncten": ncten,
        "nwfaten": nwfaten,
        "nifaten": nifaten,
        "graupel_number_shadow": graupel_number_shadow,
        "snow_velocity_boost": snow_velocity_boost,
    }
    if qcten is not None:
        fields["qcten"] = qcten
    # With rain and ice accumulators, qr/nr/qi/ni remain read-only.
    add_optional_fields(fields, qrten=qrten, nrten=nrten,
                        qiten=qiten, niten=niten)
    add_optional_fields(fields, qvten=qvten, tten=tten)
    add_optional_fields(fields, qsten=qsten, qgten=qgten, ngten=ngten)
    hail_arguments = hail_graupel_arguments(
        fields, graupel_volume=graupel_volume, qgten=qgten, ngten=ngten,
        qbten=qbten)
    shape, size = validate_fields(fields)
    del shape

    validate_fp64_fortran_table(
        "ice_deposition_partition", ice_deposition_partition,
        ICE_PARTITION_SHAPE)
    validate_fp64_fortran_table(
        "ice_to_snow_mass", ice_to_snow_mass, ICE_PARTITION_SHAPE)
    validate_fp64_fortran_table(
        "ice_to_snow_number", ice_to_snow_number, ICE_PARTITION_SHAPE)
    validate_fp64_fortran_table(
        "rain_cloud_efficiency", rain_cloud_efficiency,
        RAIN_CLOUD_EFFICIENCY_SHAPE)
    validate_fp64_fortran_table(
        "snow_cloud_efficiency", snow_cloud_efficiency,
        RAIN_CLOUD_EFFICIENCY_SHAPE)

    groups = (
        ("rain_snow_tables", rain_snow_tables, RAIN_SNOW_TABLE_NAMES,
         _rain_snow_shape()),
        ("rain_graupel_tables", rain_graupel_tables,
         RAIN_GRAUPEL_TABLE_NAMES, _rain_graupel_shape()),
        ("rain_freezing_tables", rain_freezing_tables,
         RAIN_FREEZING_TABLE_NAMES, RAIN_FREEZING_TABLE_SHAPE),
        ("cloud_freezing_tables", cloud_freezing_tables,
         CLOUD_FREEZING_TABLE_NAMES, CLOUD_FREEZING_TABLE_SHAPE),
    )
    resolved: dict[str, tuple] = {}
    for label, supplied, names, table_shape in groups:
        try:
            values = tuple(supplied)
        except TypeError as exc:
            raise TypeError(f"{label} must be iterable") from exc
        if len(values) != len(names):
            raise ValueError(
                f"{label} must contain {len(names)} arrays, "
                f"got {len(values)}")
        for name, table in zip(names, values, strict=True):
            validate_fp64_fortran_table(name, table, table_shape)
        resolved[label] = values

    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")

    # Use the measured block width; each thread still evaluates one unchanged cell.
    grid, block = launch_grid(size, threads=32)
    aerosol_kernel(COLD_MODULE, COLD_NETWORK_KERNEL)(
        grid, block,
        (qi, ni, qs, qg, qr, nr, qc, temperature, pressure, qv,
         nc_entry, nwfa_entry, nifa_entry,
         ncten, nwfaten, nifaten,
         graupel_number_shadow, snow_velocity_boost,
         ice_deposition_partition, ice_to_snow_mass, ice_to_snow_number,
         *resolved["rain_snow_tables"],
         *resolved["rain_graupel_tables"],
         *resolved["rain_freezing_tables"],
         rain_cloud_efficiency,
         *resolved["cloud_freezing_tables"],
         snow_cloud_efficiency,
         qcten, qrten, nrten, qiten, niten, qvten, tten,
         qsten, qgten, ngten,
         graupel_volume, qbten,
         np.float32(dt), np.int32(size)))


def launch_aa_cold_network_from_owner(
        qi, ni, qs, qg, qr, nr, qc, temperature, pressure, qv,
        nc_entry, nwfa_entry, nifa_entry,
        ncten, nwfaten, nifaten,
        graupel_number_shadow, snow_velocity_boost,
        classic_table_owner, dt: float, *, qcten=None, qrten=None,
        nrten=None, qiten=None, niten=None, qvten=None, tten=None,
        qsten=None, qgten=None, ngten=None, graupel_volume=None, qbten=None) -> None:
    """Launch the cold network from one verified classic table owner.

    Every coefficient this kernel needs already exists in the mp=8 table
    set: the freezeH2O records ``tpi_qcfz``/``tni_qcfz``/``tpi_qrfz``/
    ``tpg_qrfz``/``tni_qrfz``/``tnr_qrfz`` carry all 55 ``idx_IN`` slices and
    all 100 droplet bins, mp=8 simply never indexed anything but slice 27 and
    bin 65.  mp=28 therefore introduces **no new cold-network asset**; the
    only new table in the whole port is ``CCN_ACTIVATE.BIN``, which belongs
    to the saturation adjustment (WP-05), not here.

    ``classic_table_owner`` is a
    :class:`woof.core.thompson_runtime.DeviceClassicTableSet`.
    """
    from woof.core.thompson_runtime import DeviceClassicTableSet

    if (not isinstance(classic_table_owner, DeviceClassicTableSet)
            or not classic_table_owner.roundtrip_verified):
        raise TypeError(
            "classic_table_owner must be a verified DeviceClassicTableSet")
    tables = classic_table_owner.cold_source_tables
    launch_aa_cold_network(
        qi, ni, qs, qg, qr, nr, qc, temperature, pressure, qv,
        nc_entry, nwfa_entry, nifa_entry,
        ncten, nwfaten, nifaten,
        graupel_number_shadow, snow_velocity_boost,
        tables.ice_deposition_partition,
        tables.ice_to_snow_mass,
        tables.ice_to_snow_number,
        tables.rain_snow_tables,
        tables.rain_graupel_tables,
        tables.rain_freezing_tables,
        tables.rain_cloud_efficiency,
        tables.cloud_freezing_tables,
        dt,
        qcten=qcten, qrten=qrten, nrten=nrten, qiten=qiten, niten=niten,
        snow_cloud_efficiency=classic_table_owner.t_Efsw,
        qvten=qvten, tten=tten, qsten=qsten, qgten=qgten, ngten=ngten,
        graupel_volume=graupel_volume, qbten=qbten,
    )


def probe_cold_warm_loop(
        qc, nc_entry, qr, nr, nwfa_entry, nifa_entry,
        temperature, pressure, qv, rain_cloud_efficiency, dt: float):
    """Read back the cold kernel's droplet staging and its :2157-2234 rates.

    Returns a dict of device arrays:

    ``nu_c_entry``      int32   -- module_mp_thompson.F:1832, the PRE-
                                   rediagnosis shape parameter.  Diagnostic
                                   only; nothing after :1838 may consume it.
    ``nu_c_working``    int32   -- :2170, recomputed from the rediagnosed nc.
                                   THIS is what every rate below uses.
    ``nc_m3``           float32 -- :1840, the rediagnosed droplet number.
    ``mvd_c``           float32 -- :2174-2175, clamped to [D0c, D0r].
    ``mvd_r``           float32 -- :2149.
    ``pnc_wau``         float64 -- :2192-2193
    ``pnc_rcw``         float64 -- :2205-2207
    ``pna_rca``         float64 -- :2213-2216
    ``pnd_rcd``         float64 -- :2218-2221
    ``prr_wau``         float64 -- :2189-2190
    ``pnr_wau``         float64 -- :2191

    The probe calls the same shared helpers and reproduces the same
    expressions as ``thompson_aa_cold_network``; the test suite additionally
    gates its four rates against the production kernel's own accumulator
    output on a shared column so the two cannot drift.
    """
    import cupy as cp

    _, size = validate_fields({
        "qc": qc,
        "nc_entry": nc_entry,
        "qr": qr,
        "nr": nr,
        "nwfa_entry": nwfa_entry,
        "nifa_entry": nifa_entry,
        "temperature": temperature,
        "pressure": pressure,
        "qv": qv,
    })
    validate_fp64_fortran_table(
        "rain_cloud_efficiency", rain_cloud_efficiency,
        RAIN_CLOUD_EFFICIENCY_SHAPE)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")

    ints = {name: cp.empty(qc.shape, dtype=cp.int32)
            for name in ("nu_c_entry", "nu_c_working")}
    floats = {name: cp.empty(qc.shape, dtype=cp.float32)
              for name in ("nc_m3", "mvd_c", "mvd_r")}
    doubles = {name: cp.empty(qc.shape, dtype=cp.float64)
               for name in ("pnc_wau", "pnc_rcw", "pna_rca", "pnd_rcd",
                            "prr_wau", "pnr_wau")}

    grid, block = launch_grid(size)
    aerosol_kernel(COLD_MODULE, COLD_WARM_LOOP_PROBE_KERNEL)(
        grid, block,
        (qc, nc_entry, qr, nr, nwfa_entry, nifa_entry,
         temperature, pressure, qv, rain_cloud_efficiency,
         ints["nu_c_entry"], ints["nu_c_working"],
         floats["nc_m3"], floats["mvd_c"], floats["mvd_r"],
         doubles["pnc_wau"], doubles["pnc_rcw"],
         doubles["pna_rca"], doubles["pnd_rcd"],
         doubles["prr_wau"], doubles["pnr_wau"],
         np.float32(dt), np.int32(size)))
    return {**ints, **floats, **doubles}


__all__ = [
    "CLOUD_FREEZING_TABLE_NAMES",
    "CLOUD_FREEZING_TABLE_SHAPE",
    "COLD_NETWORK_KERNEL",
    "COLD_WARM_LOOP_PROBE_KERNEL",
    "RAIN_FREEZING_TABLE_NAMES",
    "RAIN_FREEZING_TABLE_SHAPE",
    "RAIN_GRAUPEL_TABLE_NAMES",
    "RAIN_GRAUPEL_TABLE_SHAPE",
    "RAIN_SNOW_TABLE_NAMES",
    "RAIN_SNOW_TABLE_SHAPE",
    "launch_aa_cold_network",
    "launch_aa_cold_network_from_owner",
    "probe_cold_warm_loop",
]
