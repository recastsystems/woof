"""WRF v4.6.1 Dudhia shortwave radiation (``ra_sw_physics=1``).

The production column routine is a direct transcription of
``phys/module_ra_sw.F:SWRAD/SWPARA`` from the pinned WRF v4.6.1 source.
It operates on either NumPy or CuPy arrays.  Columns are laid out
top-to-bottom, while woof's state remains bottom-to-top; the adapter owns
that single packing boundary and leaves every layer calculation on device.

The supported WRF path is the meteorological (non-WRF-Chem), no-eclipse,
no-slope-radiation path used by the historical-downscaling configuration.
The clear-air scattering control retains WRF's ``swrad_scat=1`` default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from woof.core import constants as c
from woof.core.mynn_radiation import (
    merge_mynn_bl_clouds,
    mynn_bl_cloud_active,
    wrf_itimestep,
)


# module_ra_sw.F:298-308.  The Fortran DATA statement fills the first
# dimension fastest, hence the transpose from water-path columns below.
_ALBTAB = np.asarray([
    [0.0, 0.0, 0.0, 0.0],
    [69.0, 58.0, 40.0, 15.0],
    [90.0, 80.0, 70.0, 60.0],
    [94.0, 90.0, 82.0, 78.0],
    [96.0, 92.0, 85.0, 80.0],
], dtype=np.float32).T
_ABSTAB = np.asarray([
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 2.5, 4.0, 5.0],
    [0.0, 2.6, 7.0, 10.0],
    [0.0, 3.3, 10.0, 14.0],
    [0.0, 3.7, 10.0, 15.0],
], dtype=np.float32).T


def _array_namespace(*values):
    """Return CuPy only when at least one input is a CUDA array."""
    if any(hasattr(value, "__cuda_array_interface__") for value in values):
        import cupy as cp
        return cp
    return np


def wrf_solar_geometry(valid_time: datetime, latitude_deg, longitude_deg,
                       *, hour_offset_seconds: float = 0.0, generation="wrf_480",
                       start_time=None, elapsed_seconds=0.0):
    """Return the selected WRF ``(coszen, solcon)`` for a radiation call.

    This is the standard ``radconst``/``calc_coszen`` path in
    ``module_radiation_driver.F:3469-3541``.  As in WRF, only the hour
    angle receives the half-radiation-interval offset; declination and the
    equation of time use the unshifted call-time Julian day.
    """
    if not isinstance(valid_time, datetime):
        raise TypeError("valid_time must be a datetime")
    if not np.isfinite(hour_offset_seconds):
        raise ValueError("hour_offset_seconds must be finite")
    from woof.core.solar import solar_generation, geometry_at
    solar_generation(generation)
    if generation == "wrf_480":
        import cupy as cp
        ds, cz, _ha = geometry_at(valid_time, latitude_deg, longitude_deg,
            hour_offset_seconds=hour_offset_seconds, generation=generation,
            start_time=start_time, elapsed_seconds=elapsed_seconds)
        return (cz if hasattr(latitude_deg, "__cuda_array_interface__") else cp.asnumpy(cz)), float(ds[1].get())
    xp = _array_namespace(latitude_deg, longitude_deg)
    lat = xp.asarray(latitude_deg)
    lon = xp.asarray(longitude_deg)
    if lat.shape != lon.shape:
        raise ValueError("radiation latitude/longitude shapes must match")

    hour = (valid_time.hour + valid_time.minute / 60.0
            + valid_time.second / 3600.0
            + valid_time.microsecond / 3.6e9)
    julian = valid_time.timetuple().tm_yday - 1.0 + hour / 24.0
    degrad = np.pi / 180.0
    dpd = 360.0 / 365.0
    solar_longitude = dpd * (
        julian - 80.0 if julian >= 80.0 else julian + 285.0)
    declination = np.arcsin(
        np.sin(23.5 * degrad) * np.sin(solar_longitude * degrad))
    da = 2.0 * np.pi * (julian - 1.0) / 365.0
    equation = 229.18 * (
        0.000075 + 0.001868 * np.cos(da) - 0.032077 * np.sin(da)
        - 0.014615 * np.cos(2.0 * da) - 0.04089 * np.sin(2.0 * da))
    solar_minutes = (60.0 * (hour + hour_offset_seconds / 3600.0)
                     + equation + 4.0 * lon)
    hour_angle = xp.deg2rad(solar_minutes / 4.0 - 180.0)
    latitude = xp.deg2rad(lat)
    cosine_zenith = (
        xp.sin(latitude) * np.sin(declination)
        + xp.cos(latitude) * np.cos(declination) * xp.cos(hour_angle))
    cosine_zenith = xp.clip(cosine_zenith, -1.0, 1.0)

    # module_radiation_driver.F:3504-3509 (Paltridge & Platt orbit).
    orbit = 2.0 * np.pi * julian / 365.0
    eccentricity = (
        1.000110 + 0.034221 * np.cos(orbit) + 0.001280 * np.sin(orbit)
        + 0.000719 * np.cos(2.0 * orbit)
        + 0.000077 * np.sin(2.0 * orbit))
    return cosine_zenith, 1370.0 * eccentricity


def _wrf_cssca(swrad_scat: float):
    """``swinit``: ``cssca = swrad_scat * 1.e-5`` in REAL (float32)."""
    return np.float32(np.float32(swrad_scat) * np.float32(1.0e-5))


def _device_columns(arrays, mu, surface_albedo, exner, *, solcon, cssca,
                    icloud):
    """Launch the dudhia_sw_column kernel on top-down (ncol, nlay) columns."""
    import cupy as cp
    from woof.core.kernels import load_module

    f32 = [cp.ascontiguousarray(value, dtype=cp.float32) for value in arrays]
    ncol, nlay = f32[0].shape
    mu = cp.ascontiguousarray(mu, dtype=cp.float32)
    surface_albedo = cp.ascontiguousarray(surface_albedo, dtype=cp.float32)
    heating = cp.empty((ncol, nlay), dtype=cp.float32)
    gsw = cp.empty((ncol,), dtype=cp.float32)
    swdown = cp.empty((ncol,), dtype=cp.float32)
    use_exner = exner is not None
    pi3d = (cp.ascontiguousarray(exner, dtype=cp.float32) if use_exner
            else heating)
    if ncol == 0:
        return heating, swdown, gsw
    kernel = load_module("dudhia_sw").get_function("dudhia_sw_column")
    block = 128
    kernel(((ncol + block - 1) // block,), (block,),
           (*f32, pi3d, mu, surface_albedo, heating, gsw, swdown,
            np.float32(solcon), cssca, np.int32(icloud),
            np.int32(1 if use_exner else 0), np.int32(ncol), np.int32(nlay),
            np.int64(nlay), np.int64(1), np.int64(0)))
    return heating, swdown, gsw


def dudhia_shortwave_columns(
        temperature, pressure, qv, qc, qr, qi, qs, qg, dz,
        cosine_zenith, albedo, *, solcon: float,
        exner=None, icloud: int = 1, swrad_scat: float = 1.0):
    """Evaluate Dudhia SW for top-to-bottom atmospheric columns.

    Parameters with a vertical coordinate have shape ``(ncol,nlay)``;
    ``cosine_zenith`` and ``albedo`` have shape ``(ncol,)``.  The returned
    tuple is ``(theta_heating, swdown, gsw)`` where ``theta_heating`` is in
    K s-1, ``swdown`` is the radiation driver's incident surface shortwave
    ``GSW/(1-ALBEDO)`` and ``gsw`` the net (surface-absorbed) flux.
    Supplying ``exner`` applies SWRAD's temperature-to-potential-temperature
    conversion; omitting it returns temperature heating.

    CUDA inputs run the ``dudhia_sw`` kernel in float32 (WRF's REAL; the
    cosine of the zenith angle is rounded to float32 first, as WRF's COSZEN
    is a REAL), graded bitwise against WRF v4.6.1 by
    tests/test_dudhia_sw_wrf461_parity.py.  NumPy inputs run a host mirror
    of the same expressions in the input dtype with NumPy's own power and
    log10, which is not graded bitwise.
    """
    xp = _array_namespace(
        temperature, pressure, qv, qc, qr, qi, qs, qg, dz,
        cosine_zenith, albedo, exner)
    arrays = [xp.asarray(value) for value in
              (temperature, pressure, qv, qc, qr, qi, qs, qg, dz)]
    shape = arrays[0].shape
    if len(shape) != 2 or any(value.shape != shape for value in arrays[1:]):
        raise ValueError("Dudhia layer inputs must share shape (ncol,nlay)")
    ncol, nlay = shape
    mu = xp.asarray(cosine_zenith)
    surface_albedo = xp.asarray(albedo)
    if mu.shape != (ncol,) or surface_albedo.shape != (ncol,):
        raise ValueError("Dudhia surface inputs must have shape (ncol,)")
    if exner is not None and xp.asarray(exner).shape != shape:
        raise ValueError("Dudhia exner must match the layer shape")
    if icloud not in (0, 1, 2, 3):
        raise ValueError("icloud must be 0, 1, 2 or 3")
    if (not np.isfinite(solcon) or solcon <= 0.0
            or not np.isfinite(swrad_scat) or swrad_scat < 0.0):
        raise ValueError("solcon must be positive and swrad_scat non-negative")

    dtype = arrays[0].dtype
    if dtype.kind != "f":
        raise TypeError("Dudhia atmospheric inputs must be floating point")
    if xp is not np:
        return _device_columns(
            arrays, mu, surface_albedo,
            None if exner is None else xp.asarray(exner),
            solcon=solcon, cssca=_wrf_cssca(swrad_scat), icloud=icloud)

    def real(value):
        return dtype.type(value)
    mu = mu.astype(dtype)
    surface_albedo = surface_albedo.astype(dtype)
    t, p, qv_a, qc_a, qr_a, qi_a, qs_a, qg_a, dz_a = arrays
    rho = p / (real(c.RD) * t)
    vapor_path = rho * xp.maximum(qv_a, real(0.0)) * dz_a * real(1000.0)
    air_path = rho * dz_a
    if icloud:
        cloud_path = rho * real(1000.0) * dz_a * (
            xp.maximum(qc_a, real(0.0))
            + real(0.1) * xp.maximum(qi_a, real(0.0))
            + real(0.05) * xp.maximum(qr_a, real(0.0))
            + real(0.02) * xp.maximum(qs_a, real(0.0))
            + real(0.05) * xp.maximum(qg_a, real(0.0)))
    else:
        cloud_path = xp.zeros_like(t)

    daylight = mu > real(1.0e-9)
    mu_safe = xp.where(daylight, mu, real(1.0))
    soltop = real(solcon)
    sdown = xp.where(daylight, soltop * mu_safe, real(0.0))
    top_flux = sdown.copy()
    heating = xp.zeros_like(t)

    ww = xp.zeros((ncol,), dtype=dtype)
    uv = xp.zeros_like(ww)
    oldalb = xp.zeros_like(ww)
    oldabc = xp.zeros_like(ww)
    totabs = xp.zeros_like(ww)
    dsca = xp.zeros_like(ww)
    dabs = xp.zeros_like(ww)
    dscld = xp.zeros_like(ww)
    cssca = (real(_wrf_cssca(swrad_scat)) if dtype == np.float32
             else real(swrad_scat * 1.0e-5))
    albtab = xp.asarray(_ALBTAB, dtype=dtype)
    abstab = xp.asarray(_ABSTAB, dtype=dtype)
    xmuval = xp.asarray([0.0, 0.2, 0.5, 1.0], dtype=dtype)

    # SWPARA's top-down recurrence, vectorized over columns, with the same
    # expression forms the device kernel (kernels/dudhia_sw.cu) evaluates.
    for k in range(nlay):
        ww = ww + cloud_path[:, k]
        uv = uv + vapor_path[:, k]
        wgm = ww / mu_safe
        ugcm = uv * real(0.0001) / mu_safe
        oldabs = totabs
        totabs = (real(2.9) * ugcm
                  / ((real(1.0) + real(141.5) * ugcm) ** real(0.635)
                     + real(5.925) * ugcm))
        xsca = cssca * air_path[:, k] / mu_safe
        safe_sdown = xp.where(daylight, sdown, real(1.0))
        xabs = (totabs - oldabs) * (top_flux - dscld - dsca) / safe_sdown
        xabs = xp.maximum(xabs, real(0.0))

        alw = xp.minimum(xp.log10(wgm + real(1.0)), real(3.999))
        # The last of XMUVAL(1..3) that XMU exceeds (one-based IIL).
        iil = xp.where(mu_safe > real(0.5), 3,
                       xp.where(mu_safe > real(0.2), 2, 1)).astype(xp.int32)
        iu = iil + 1
        xi = ((mu_safe - xmuval[iil - 1])
              / (xmuval[iu - 1] - xmuval[iil - 1]) + iil.astype(dtype))
        jjl = alw.astype(xp.int32) + 1
        ju = jjl + 1
        yj = alw + real(1.0)
        wa = xi - iil.astype(dtype)
        wb = iu.astype(dtype) - xi
        wc = yj - jjl.astype(dtype)
        wd = ju.astype(dtype) - yj

        def interp(table):
            return (table[iu - 1, ju - 1] * wa * wc
                    + table[iil - 1, ju - 1] * wb * wc
                    + table[iu - 1, jjl - 1] * wa * wd
                    + table[iil - 1, jjl - 1] * wb * wd)

        alba = interp(albtab)
        absc = interp(abstab)
        clear_available = top_flux - dsca - dabs
        xalb = xp.maximum(
            (alba - oldalb) * clear_available / safe_sdown, real(0.0))
        xabsc = xp.maximum(
            (absc - oldabc) * clear_available / safe_sdown, real(0.0))
        dscld = dscld + (xalb + xabsc) * sdown * real(0.01)
        dsca = dsca + xsca * sdown
        dabs = dabs + xabs * sdown
        oldalb = alba
        oldabc = absc

        trans = (real(100.0) - xalb - xabsc
                 - xabs * real(100.0) - xsca * real(100.0))
        over = trans < real(1.0)
        attenuation = (xalb + xabsc + xabs * real(100.0)
                       + xsca * real(100.0))
        ff = real(99.0) / xp.maximum(attenuation, real(1.0e-30))
        xabsc = xp.where(over, xabsc * ff, xabsc)
        xabs = xp.where(over, xabs * ff, xabs)
        trans = xp.where(over, real(1.0), trans)
        heating[:, k] = xp.where(
            daylight,
            sdown * (xabsc + xabs * real(100.0)) * real(0.01)
            / (rho[:, k] * real(c.CP) * dz_a[:, k]),
            real(0.0))
        sdown = xp.where(
            daylight,
            xp.maximum(real(1.0e-9), sdown * trans * real(0.01)),
            real(0.0))

    if exner is not None:
        heating = heating / xp.asarray(exner)
    # GSW=(1.-ALBEDO)*SDOWN(kte+1); the driver's SWDOWN=GSW/(1.-ALBEDO)
    # (module_radiation_driver.F:2877).
    gsw = (real(1.0) - surface_albedo) * sdown
    swdown = gsw / (real(1.0) - surface_albedo)
    return heating, swdown, gsw


@dataclass
class DudhiaShortwaveRadiation:
    """GPU-resident adapter for WRF ``ra_sw_physics=1``."""

    start_time: datetime
    latitude_deg: object
    longitude_deg: object
    swrad_scat: float = 1.0
    icloud: int = 1
    update_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        import cupy as cp

        if not isinstance(self.start_time, datetime):
            raise TypeError("radiation_start_time must be a datetime")
        self.latitude_deg = cp.ascontiguousarray(
            cp.asarray(self.latitude_deg, dtype=cp.float32))
        self.longitude_deg = cp.ascontiguousarray(
            cp.asarray(self.longitude_deg, dtype=cp.float32))
        if self.latitude_deg.shape != self.longitude_deg.shape:
            raise ValueError("radiation latitude/longitude shapes must match")
        if (not np.isfinite(self.swrad_scat) or self.swrad_scat < 0.0):
            raise ValueError("swrad_scat must be finite and non-negative")
        if self.icloud not in (0, 1, 2, 3):
            raise ValueError("icloud must be 0, 1, 2 or 3")

    @staticmethod
    def _top_down(array):
        import cupy as cp
        nz, ny, nx = array.shape
        return cp.ascontiguousarray(
            array.transpose(1, 2, 0).reshape(ny * nx, nz)[:, ::-1])

    @staticmethod
    def _state_field(state, name, fallback):
        value = getattr(state, name, None)
        return fallback if value is None else value

    def __call__(self, *, atmosphere, fields, state, cfg):
        import cupy as cp

        temperature = atmosphere["temperature"]
        nz, ny, nx = temperature.shape
        if self.latitude_deg.shape != (ny, nx):
            raise ValueError("radiation latitude/longitude must match state grid")
        zero = cp.zeros_like(temperature)
        qc = self._state_field(state, "qc", atmosphere["qc"])
        qr = self._state_field(state, "qr", zero)
        qi = self._state_field(state, "qi", atmosphere["qi"])
        qs = self._state_field(state, "qs", zero)
        qg = self._state_field(state, "qg", zero)
        if int(getattr(cfg, "cu_physics", 0)) == 10:
            # module_radiation_driver.F:1510-1557 precedes every spectrum.
            from woof.core.rrtmgp import cal_cldfra1, cal_cldfra2
            from woof.core.rrtmg_legacy import legacy_cloud_fraction_flags
            from woof.core.kfcup_radiation import merge_kfcup_clouds
            level = lambda value: value.reshape(nz, -1)
            f_qi, f_qs = legacy_cloud_fraction_flags(int(cfg.mp_physics))
            liquid = level(qc).copy()
            flags = None
            if self.icloud == 1:
                fraction, flags = cal_cldfra1(level(atmosphere["qv"]), liquid,
                    level(qi), level(qs), level(temperature), level(atmosphere["pressure"]),
                    f_qc=True, f_qi=f_qi, f_qs=f_qs,
                    mp_physics=int(cfg.mp_physics), return_flags=True,
                    f_ice_phy=level(state.f_ice_phy) if int(cfg.mp_physics)==95 else None)
            elif self.icloud == 2:
                fraction = cal_cldfra2(liquid, level(qi), f_qc=True, f_qi=f_qi)
            else:
                fraction = cp.zeros_like(liquid)
            liquid, _ = merge_kfcup_clouds(liquid, fraction, flags,
                fields, cfg, level_major=True)
            qc = liquid.reshape(nz, ny, nx)
        qc_cols = self._top_down(qc)
        qi_cols = self._top_down(qi)
        active_bl = self.icloud not in (2, 3) and mynn_bl_cloud_active(
            getattr(cfg, "bl_pbl_physics", 0), getattr(cfg, "icloud_bl", 0))
        qc_bl = self._top_down(fields["qc_bl"]) if active_bl else None
        qi_bl = self._top_down(fields["qi_bl"]) if active_bl else None
        cldfra_bl = (
            self._top_down(fields["cldfra_bl"]) if active_bl else None)
        qc_cols, qi_cols, _ = merge_mynn_bl_clouds(
            qc_cols, qi_cols, None, qc_bl=qc_bl, qi_bl=qi_bl,
            cldfra_bl=cldfra_bl,
            bl_pbl_physics=getattr(cfg, "bl_pbl_physics", 0),
            icloud_bl=(getattr(cfg, "icloud_bl", 0) if self.icloud not in (2, 3) else 0),
            itimestep=(wrf_itimestep(state.elapsed_seconds, cfg.dt)
                       if active_bl else 1),
            bl_mynn_version=getattr(cfg, "bl_mynn_version", "wrf_461"),
            temperature=self._top_down(temperature) if active_bl else None,
        )

        from woof.core.column_cumulus import merge_column_cumulus_clouds
        qc_cols, qi_cols, _ = merge_column_cumulus_clouds(
            qc_cols, qi_cols, None, fields=fields, cfg=cfg,
            columns=self._top_down)

        qs_cols=self._top_down(qs)
        feedback=atmosphere.get("cumulus_feedback")
        if feedback is not None and self.icloud:
            from woof.core.cumulus_feedback import merge_cumulus_feedback
            packed={name:self._top_down(value) for name,value in feedback.items()}
            qc_cols,qi_cols,qs_cols,_,_=merge_cumulus_feedback(qc_cols,qi_cols,qs_cols,cp.zeros_like(qc_cols),packed,aerosol=cfg.aercu_opt>0,
                radii={target:self._top_down(getattr(state,source)) for target,source in (("effc","efcg"),("effi","efig"),("effs","efsg"))} if cfg.aercu_opt>0 else None)

        from woof.core.physics import (
            RadiationResult, _physics_period_seconds)
        valid_time = (self.start_time
                      + timedelta(seconds=float(state.elapsed_seconds)))
        radt_minutes = cfg.radt if cfg.radt > 0.0 else cfg.radt_minutes
        interval = _physics_period_seconds(radt_minutes, cfg)
        from woof.core.solar import solar_period_seconds
        interval = solar_period_seconds(cfg, interval)
        mu, solcon = wrf_solar_geometry(
            valid_time, self.latitude_deg, self.longitude_deg,
            hour_offset_seconds=0.5 * interval, generation=getattr(cfg, "solar_generation", "wrf_480"),
            start_time=self.start_time, elapsed_seconds=state.elapsed_seconds)
        heating_top, swdown, gsw = dudhia_shortwave_columns(
            self._top_down(temperature),
            self._top_down(atmosphere["pressure"]),
            self._top_down(atmosphere["qv"]), qc_cols,
            self._top_down(qr), qi_cols, qs_cols,
            self._top_down(qg), self._top_down(atmosphere["dz"]),
            mu.reshape(-1), cp.asarray(fields["albedo"]).reshape(-1),
            solcon=solcon, exner=self._top_down(atmosphere["exner"]),
            icloud=self.icloud, swrad_scat=self.swrad_scat)
        heating = cp.ascontiguousarray(
            heating_top[:, ::-1].reshape(ny, nx, nz).transpose(2, 0, 1))
        result = RadiationResult(
            rthratenlw=cp.zeros_like(heating), rthratensw=heating,
            swdown=swdown.reshape(ny, nx), glw=fields["glw"],
            gsw=gsw.reshape(ny, nx),
            coszen=cp.asarray(mu, dtype=cp.float32).reshape(ny, nx))
        self.update_count += 1
        return result

    @property
    def restart_identity(self) -> dict[str, object]:
        """Trajectory-changing setup bound into tree restart identity."""
        return {
            "algorithm": "wrf-v4.6.1-dudhia-shortwave",
            "wrf_source": "phys/module_ra_sw.F:SWRAD/SWPARA",
            "swrad_scat": float(self.swrad_scat),
            "icloud": int(self.icloud),
            "supported_path": "no-chem,no-eclipse,no-slope",
        }


__all__ = ["DudhiaShortwaveRadiation", "dudhia_shortwave_columns",
           "wrf_solar_geometry"]
