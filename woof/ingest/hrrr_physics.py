"""Shared prepared-real physics initialization for GPU forecast runners.

The historical public entry point remains :func:`initialize_hrrr_physics`,
but the actual physics setup consumes the source-neutral Noah surface contract
stored by ``gpuwm-prepared-real-cache-v1``.  GFS and ERA5 cache restores use
that same path instead of re-decoding source-specific soil fields.
"""

from __future__ import annotations
from woof.surface_ocean import ocean_input_fields

from collections.abc import Mapping
from types import MappingProxyType, SimpleNamespace

import numpy as np

from woof.config import soil_layer_count
from woof.core.noah import noah_initial_snow_albedo
from woof.ingest.hrrr_surface import surface_fields_to_device
from woof.ingest.vegetation import initial_vegetation_fraction


_CANONICAL_SURFACE_FIELDS = frozenset({
    "TSK", "TSLB", "SMOIS", "SH2O", "TMN", "SEAICE", "XLAND",
    "LANDMASK", "SNOW", "SNOWH",
})

#: The native soil pair a met snapshot carries when nothing has solved
#: the surface yet.  Named here because the prepared-cache writer DROPS
#: it precisely when a solved surface rides along
#: (:func:`woof.ingest.prepared_cache._prepared_met_names`), so these
#: two names are the whole difference between the two roads below.
_NATIVE_SOIL_FIELDS = ("SOILT", "SOILW")


def _validate_prepared_surface(surface, cfg) -> Mapping[str, object]:
    try:
        fields = surface.fields
    except AttributeError as exc:
        raise TypeError(
            "prepared physics surface must expose a fields mapping") from exc
    if not isinstance(fields, Mapping):
        raise TypeError("prepared physics surface fields must be a mapping")
    if set(fields) != _CANONICAL_SURFACE_FIELDS:
        raise ValueError(
            "prepared physics surface inventory differs from the canonical "
            f"Noah contract: expected {sorted(_CANONICAL_SURFACE_FIELDS)}, "
            f"got {sorted(fields)}")
    horizontal_shape = (int(cfg.ny), int(cfg.nx))
    # The prepared cache stores whatever geometry the run that wrote it used,
    # so the shape gate is the SELECTED scheme's layer count, not a literal.
    # A four-layer cache restored under a nine-layer scheme is refused here
    # by shape rather than broadcast into place downstream.
    soil_shape = (soil_layer_count(cfg), *horizontal_shape)
    host_fields = {}
    for name in sorted(_CANONICAL_SURFACE_FIELDS):
        value = fields[name]
        if hasattr(value, "get"):
            value = value.get()
        array = np.asarray(value)
        host_fields[name] = array
        expected = soil_shape if name in {"TSLB", "SMOIS", "SH2O"} \
            else horizontal_shape
        if array.shape != expected:
            raise ValueError(
                f"prepared surface {name} has shape {array.shape}, "
                f"expected {expected}")
        if (array.dtype.hasobject
                or not np.issubdtype(array.dtype, np.number)
                or not np.isfinite(array).all()):
            raise ValueError(
                f"prepared surface {name} must be finite numeric data")
    if not np.array_equal(
            host_fields["XLAND"],
            np.where(host_fields["LANDMASK"] >= 0.5, 1.0, 2.0)):
        raise ValueError(
            "prepared surface XLAND differs from its canonical LANDMASK")
    return fields


_PBLH_LOGGED: set[str] = set()


def _seeded_initial_pblh(landuse, met):
    """The initial PBLH: the source's own, when it publishes one.

    An HRRR f00 is a cycled state.  Its wrfinput carries the background
    forecast's PBL height, so NOAA's f00 products (the GUST message and
    every PBL-depth product) see a boundary layer, and the first
    surface-layer call reads that depth for its convective velocity.  A
    cold start seeds zero, as real.exe does, and the initial history then
    wrote PBLH = 0 at every point: the f00 gust was the bare 10 m wind
    (2026-10-21 06Z full CONUS: WOOF f00 gust 2.6 m/s weaker than the same
    wrf-rust gust on NOAA's own f00 state, whose HPBL averaged 347 m).

    A source that publishes a PBL height (the mapped ``PBLH`` met field:
    HRRR wrfnat HPBL today) seeds it.  Every other source keeps stock WRF's
    zero, and the run says so once.
    """
    from dataclasses import replace

    fields = getattr(met, "fields", None) or {}
    value = fields.get("PBLH")
    if value is None:
        message = ("physics init: PBLH starts at 0 everywhere (stock WRF "
                   "cold start): the initial source publishes no PBL height")
        if message not in _PBLH_LOGGED:
            _PBLH_LOGGED.add(message)
            print(message, flush=True)
        return landuse
    host = value.get() if hasattr(value, "get") else value
    array = np.asarray(host)
    expected = np.shape(landuse.pblh)
    if array.shape != expected:
        raise ValueError(
            f"source PBLH has shape {array.shape}, expected {expected}; a "
            "PBL height on another grid would seed the wrong columns")
    if (array.dtype.hasobject or not np.issubdtype(array.dtype, np.number)
            or not np.isfinite(array).all()):
        raise ValueError("source PBLH must be finite numeric data")
    if array.min() < -1.0 or array.max() > 10_000.0:
        raise ValueError(
            "source PBLH is outside 0..10000 m: observed "
            f"{float(array.min()):.6g}..{float(array.max()):.6g}")
    message = ("physics init: PBLH seeded from the initial source's own PBL "
               "height (as a cycled HRRR wrfinput carries it)")
    if message not in _PBLH_LOGGED:
        _PBLH_LOGGED.add(message)
        print(message, flush=True)
    return replace(landuse, pblh=np.ascontiguousarray(
        np.maximum(array, 0.0), dtype=np.float32))


def _validate_prepared_near_surface(result, met, cfg) -> Mapping[str, np.ndarray]:
    """Validate exact, physical near-surface arrays before GPU allocation."""

    try:
        met_fields = met.fields
    except AttributeError as exc:
        raise TypeError(
            "prepared physics meteorology must expose a fields mapping") from exc
    if not isinstance(met_fields, Mapping):
        raise TypeError("prepared physics meteorology fields must be a mapping")
    ny, nx = int(cfg.ny), int(cfg.nx)
    specs = {
        "surface_pressure": (result.surface_pressure, (ny, nx), 1_000.0, 120_000.0),
        "surface_qv": (result.surface_qv, (ny, nx), 0.0, 0.2),
        "T2": (met_fields["T2"], (ny, nx), 100.0, 400.0),
        "SKINTEMP": (met_fields["SKINTEMP"], (ny, nx), 100.0, 400.0),
        "U10": (met_fields["U10"], (ny, nx + 1), -250.0, 250.0),
        "V10": (met_fields["V10"], (ny + 1, nx), -250.0, 250.0),
        "LANDSEA": (met_fields["LANDSEA"], (ny, nx), 0.0, 1.0),
    }
    validated = {}
    for name, (value, expected_shape, lower, upper) in specs.items():
        if hasattr(value, "get"):
            value = value.get()
        array = np.asarray(value)
        if array.shape != expected_shape:
            raise ValueError(
                f"prepared near-surface {name} has shape {array.shape}, "
                f"expected {expected_shape}")
        if (array.dtype.hasobject
                or not np.issubdtype(array.dtype, np.number)
                or not np.isfinite(array).all()):
            raise ValueError(
                f"prepared near-surface {name} must be finite numeric data")
        if array.min() < lower or array.max() > upper:
            # Name the numbers.  This refusal used to say only which field
            # and which range, and a field report ("surface_qv is outside
            # the physical range 0.0..0.2") cost a full re-run of a
            # two-stage preparation to learn that the excursion was two
            # cells at -1.8e-05 rather than, say, a unit error.  The
            # observed extremes and the offending-cell count are already
            # in hand here and decide which of those it is on sight.
            outside = int(np.count_nonzero(
                (array < lower) | (array > upper)))
            raise ValueError(
                f"prepared near-surface {name} is outside the physical "
                f"range {lower}..{upper}: {outside} of {array.size} "
                f"cell(s) are out, observed range "
                f"{float(array.min()):.6g}..{float(array.max()):.6g}")
        validated[name] = array
    if not np.isin(validated["LANDSEA"], (0.0, 1.0)).all():
        raise ValueError("prepared near-surface LANDSEA must be exactly binary")
    return MappingProxyType(validated)


def initialize_prepared_physics(
        result, cfg, met, surface, static, landuse_attrs, grid, valid_time, *,
        center_lat=None, constant_glw_wm2=None, fractional_seaice=True,
        isoilwater=14, p_top=None, column_chunk=None,
        trace_gas_overrides=None, ozone_parent=None, cam_ozone=None,
        simulation_start_time=None):
    """Attach physics to one restored source-neutral prepared initial state.

    ``surface`` is the canonical Noah inventory persisted with direct GFS and
    ERA5 caches.  Every field is validated before device allocation.  The
    actual land-use, diagnostics, physics-driver, and near-surface setup is the
    same implementation used by the native HRRR runner.

    ``valid_time`` dates this domain's analysis and land climatologies.
    ``simulation_start_time`` is the radiation calendar's origin when a
    child starts later: model elapsed seconds are measured from the root
    experiment, so adding them to the child's start would count its delay
    twice. Omission retains the ordinary single-epoch behavior.

    ``constant_glw_wm2`` is the DECLARED constant downward longwave, for a
    suite that runs a land-surface scheme with ``ra_lw_physics = 0``.  The
    caller reads it off the experiment's acknowledgements
    (:func:`woof.runtime.declared_constant_glw`); ``None`` -- the normal
    answer -- lets the attached longwave scheme own the field, and makes
    :func:`~woof.core.physics.initialize_physics` refuse an undeclared
    suite that has no longwave scheme at all.
    """

    import cupy as cp

    from woof.core.diagnostics import update_diagnostics
    from woof.core.landuse import (initialize_landuse,
                                    prepared_sea_ice_cells,
                                    usemonalb_landuse_inputs)
    from woof.core.physics import initialize_physics
    from woof.ingest.ruc_mosaic import ruc_mosaic_physics_inputs
    from woof.ingest.lake_physics import lake_physics_inputs
    from woof.static.build import monthly_interp_to_date

    fields = _validate_prepared_surface(surface, cfg)
    near_surface_host = _validate_prepared_near_surface(result, met, cfg)
    required_attrs = {"MMINLU", "ISWATER", "ISLAKE", "ISICE"}
    admitted_attrs = (required_attrs | {"ISURBAN"}
                      if getattr(cfg, "sf_surface_mosaic", 0) == 1 else required_attrs)
    if not isinstance(landuse_attrs, Mapping) \
            or not required_attrs <= set(landuse_attrs) <= admitted_attrs:
        raise ValueError(
            "prepared physics land-use identity must contain exactly "
            f"{sorted(required_attrs)}")
    state = result.state
    update_diagnostics(state, cfg.hypsometric_opt)
    landuse = initialize_landuse(
        static["LU_INDEX"], soil_type=static["SCT_DOM"],
        urban_legend=int(getattr(cfg, "sf_urban_physics", 0)) > 0,
        landmask=static["LANDMASK"], snow=fields["SNOW"],
        xice=fields["SEAICE"], valid_time=valid_time,
        cen_lat=float(
            getattr(grid, "cen_lat", grid.ref_lat)
            if center_lat is None else center_lat),
        mminlu=str(landuse_attrs["MMINLU"]),
        iswater=int(landuse_attrs["ISWATER"]),
        islake=int(landuse_attrs["ISLAKE"]),
        isice=int(landuse_attrs["ISICE"]), fractional_seaice=fractional_seaice,
        isoilwater=int(isoilwater),
        # real.exe's landmask/soil-category reconciliation decides a
        # disagreeing column from its soil temperature, then its SST.
        soil_temperature=fields["TSLB"], sst=fields.get("SST"),
        # The cells the solved surface made sea ice (XICE or real.exe's
        # seaice_threshold), so the ice category lands where its column is.
        sea_ice_cells=prepared_sea_ice_cells(
            fields, getattr(met, "fields", None), static["LANDMASK"]),
        **usemonalb_landuse_inputs(cfg, static, valid_time))
    landuse = _seeded_initial_pblh(landuse, met)
    vegfra = initial_vegetation_fraction(met, static, valid_time)
    from woof.core.landuse import surface_leaf_area
    lai = surface_leaf_area(cfg, static["LAI12M"], valid_time)
    lat, lon = grid.latlon_mass()
    from woof.core.radiation_composition import make_radiation
    radiation_origin = (valid_time if simulation_start_time is None
                        else simulation_start_time)
    radiation = make_radiation(
        cfg, radiation_origin, lat, lon, p_top=p_top, column_chunk=column_chunk,
        trace_gas_overrides=trace_gas_overrides, ozone_parent=ozone_parent)
    driver = initialize_physics(
        state, cfg, landuse=landuse, radiation=radiation, tsk=fields["TSK"],
        # The coupled hydro exchange binds to this exact parent grid (2.9.3
        # improve-1: the prepared GFS/ERA5 route passed none, so a default-on
        # hydro forecast stopped with "needs the actual forecast ProjectedGrid").
        hydro_parent_geometry=grid,
        irrigation=static.get("IRRIGATION"), irr_ph_field=static.get("IRR_PH_FIELD"),
        soil_temperature=fields["TSLB"],
        soil_moisture=fields["SMOIS"],
        liquid_moisture=fields["SH2O"],
        ivgtyp=static["LU_INDEX"], isltyp=static["SCT_DOM"],
        vegfra=vegfra, tmn=fields["TMN"], xice=fields["SEAICE"],
        snow=fields["SNOW"], snow_depth=fields["SNOWH"],
        sst=fields.get("SST", fields["TSK"]),
        glw=constant_glw_wm2,
        radiation_start_time=radiation_origin, radiation_latitude=lat,
        radiation_longitude=lon,
        landuse_dataset=str(landuse_attrs["MMINLU"]),
        # The prepared statics carry the sub-grid orographic statistics when
        # the preparation ran with topo_wind / gwd_opt on; read only then.
        terrain_drag_static=static,
        **lake_physics_inputs(cfg, static),
        **ocean_input_fields(cfg, fields),
        **ruc_mosaic_physics_inputs(
            cfg, static, landuse_attrs=landuse_attrs, xice=fields["SEAICE"],
            fractional_seaice=fractional_seaice),
        **({"cam_ozone": cam_ozone} if cam_ozone is not None else {}))
    from woof.core.landuse import surface_snow_albedo
    driver.fields["snoalb"][...] = cp.asarray(
        surface_snow_albedo(cfg, static, driver.noah_params),
        dtype=cp.float32)
    driver.fields["lai"][...] = cp.asarray(lai, dtype=cp.float32)
    driver.fields["shdmin"][...] = cp.asarray(
        100.0 * static["GREENFRAC"].min(axis=0), dtype=cp.float32)
    driver.fields["shdmax"][...] = cp.asarray(
        100.0 * static["GREENFRAC"].max(axis=0), dtype=cp.float32)
    driver.fields["psfc"][...] = cp.asarray(
        near_surface_host["surface_pressure"], dtype=cp.float32)
    near_surface = surface_fields_to_device(
        SimpleNamespace(fields=near_surface_host), cp)
    driver.fields["t2"][...] = near_surface["T2"]
    driver.fields["q2"][...] = cp.asarray(
        near_surface_host["surface_qv"], dtype=cp.float32)
    driver.fields["th2"][...] = (
        driver.fields["t2"]
        * (cp.float32(100000.0) / driver.fields["psfc"])
        ** cp.float32(287.0 / 1004.0))
    driver.fields["u10"][...] = 0.5 * (
        near_surface["U10"][:, :-1] + near_surface["U10"][:, 1:])
    driver.fields["v10"][...] = 0.5 * (
        near_surface["V10"][:-1] + near_surface["V10"][1:])
    from woof.core.myjsfc import seed_eta_family_10m_winds
    seed_eta_family_10m_winds(driver.fields, cfg.sf_sfclay_physics)
    if getattr(cfg, "sf_surface_mosaic", 0) == 1:
        from woof.core.noah_mosaic_door import attach_noah_mosaic_to_driver
        attach_noah_mosaic_to_driver(
            driver, cfg, landusef=static.get("LANDUSEF"), processed=False,
            landuse_attrs=landuse_attrs, landmask=static["LANDMASK"],
            fractional_seaice=fractional_seaice)
    _seed_initial_cldfra(driver, state, cfg, met)
    return driver


#: UPP's RAP/HRRR cloud-layer tops (Pa): low cloud is below 642 hPa,
#: middle 642-350 hPa, high above 350 hPa -- the layers the source's
#: LCDC/MCDC/HCDC describe.
_CLOUD_LAYER_TOPS_PA = (64200.0, 35000.0)


def _seed_initial_cldfra(driver, state, cfg, met=None) -> None:
    """The initial CLDFRA: the radiation's own cloud fraction of the start.

    CLDFRA is the cloud fraction the radiation radiated through, written by
    each due radiation call and held between calls; it is output only (no
    physics reads ``driver.cldfra``).  Before the first call it was zero,
    as WRF's t=0 frame is, so every f00 cloud product was blank: the
    2026-10-21 06Z full-CONUS f00 column-maximum cloud fraction was 0 at
    every point (the plot floor) against NOAA's 54 % mean, and low, middle
    and high cloud cover drew nothing.

    Seeded here with the SAME function the radiation applies on its first
    call (icloud=1: WRF's cal_cldfra1 Xu-Randall fraction; icloud=2: the
    binary cal_cldfra2), on the initial hydrometeors, temperature and
    pressure (``_prepare_atmosphere``, the radiation's own inputs), with the
    microphysics' F_QI/F_QS pair.  The MYNN subgrid merge adds nothing at
    a start: CLDFRA_BL is zero until the PBL scheme first runs.  The first
    radiation call overwrites the field as before.

    Where the source carries its own low/middle/high cloud cover (HRRR
    wrfnat LCDC/MCDC/HCDC; it publishes no 3-D cloud fraction), each layer
    of each column is brought up to the source's cover: levels with
    resolved cloud take max(own, cover); a layer with cover but no resolved
    cloud puts it on its most humid level.  So the column's cover in each
    layer is the source's (HRRR's cover is mostly MYNN subgrid cloud,
    which the start state cannot hold: 2026-10-21 06Z, resolved clouds
    alone gave a column-maximum mean of 6 % against NOAA's 54 %).
    """
    cldfra = getattr(driver, "cldfra", None)
    icloud = int(getattr(cfg, "icloud", 1))
    if cldfra is None or icloud not in (1, 2):
        return
    import cupy as cp

    from woof.core.physics import _prepare_atmosphere
    from woof.core.rrtmg_legacy import legacy_cloud_fraction_flags
    from woof.core.rrtmgp import cal_cldfra1, cal_cldfra2

    atmosphere = _prepare_atmosphere(state)
    nz, ny, nx = atmosphere["pressure"].shape
    ncol = ny * nx
    zeros = cp.zeros((nz, ncol), dtype=cp.float32)

    def grid(value):
        return cp.ascontiguousarray(
            cp.asarray(value, dtype=cp.float32)).reshape(nz, ncol)

    def species(name):
        value = getattr(state, name, None)
        return zeros if value is None else grid(value)

    mp_physics = int(cfg.mp_physics)
    f_qi, f_qs = ((True, True) if mp_physics == 7
                  else legacy_cloud_fraction_flags(mp_physics))
    if icloud == 2:
        values = cal_cldfra2(species("qc"), species("qi"), f_qc=True, f_qi=f_qi)
    else:
        values = cal_cldfra1(
            species("qv"), species("qc"), species("qi"), species("qs"),
            grid(atmosphere["temperature"]), grid(atmosphere["pressure"]),
            f_qc=True, f_qi=f_qi, f_qs=f_qs, mp_physics=mp_physics,
            f_ice_phy=species("f_ice_phy") if mp_physics == 95 else None)
    values = cp.asarray(values, dtype=cp.float32)
    fields = getattr(met, "fields", None) or {}
    covers = [fields.get(name) for name in ("LCDC", "MCDC", "HCDC")]
    if all(cover is not None for cover in covers):
        pressure = grid(atmosphere["pressure"])
        temperature = grid(atmosphere["temperature"])
        es = cp.float32(611.2) * cp.exp(cp.float32(17.67) * (temperature - cp.float32(273.15))
                                        / (temperature - cp.float32(29.65)))
        humidity = species("qv") / (cp.float32(0.622) * es / cp.maximum(pressure - es, cp.float32(1.0)))
        low_top, mid_top = _CLOUD_LAYER_TOPS_PA
        layers = (pressure >= low_top,
                  (pressure < low_top) & (pressure >= mid_top),
                  pressure < mid_top)
        for inside, cover in zip(layers, covers):
            host = cover.get() if hasattr(cover, "get") else cover
            cover = cp.clip(cp.asarray(host, dtype=cp.float32).reshape(ncol)
                            / cp.float32(100.0), 0.0, 1.0)
            resolved = inside & (values > 0)
            values = cp.where(resolved, cp.maximum(values, cover[None, :]), values)
            # Layers with cover and no resolved cloud: the most humid level.
            empty = ~resolved.any(axis=0) & (cover > 0) & inside.any(axis=0)
            level = cp.argmax(cp.where(inside, humidity, cp.float32(-1.0)), axis=0)
            columns = cp.nonzero(empty)[0]
            values[level[columns], columns] = cp.maximum(
                values[level[columns], columns], cover[columns])
    cldfra[...] = values.astype(cldfra.dtype).reshape(cldfra.shape)


def resolve_prepared_noah_surface(met, cfg, static, *, surface=None, soil_mesh=None):
    """The canonical Noah surface this state initializes its land model from.

    Two roads arrive here holding the same forecast, and they hold
    DIFFERENT halves of the soil statement.

    The FRESH road has just decoded the source: its met snapshot carries
    the native ``SOILT``/``SOILW`` node column and nothing has solved a
    surface yet, so the surface is derived here
    (``preprocess_land_surface_soil`` -> ``canonical_noah_surface``).

    The RESTORE road holds a prepared cache, and the solved surface is
    exactly what that cache persists -- ``surface/TSLB``, ``SMOIS``,
    ``SH2O`` and the rest of the canonical Noah inventory.  Because the
    surface is stored, the writer deliberately DROPS the native pair from
    the met contract beside it
    (:func:`woof.ingest.prepared_cache._prepared_met_names`: the pair is
    required only when ``surface is None``).  So a restore has the
    answer and not the ingredients.

    Re-deriving on the restore road is therefore not a slow path, it is a
    crash: the soil router, finding no native pair and no mapped or GFS
    markers, falls through to its ERA5-named arm and dies on ``missing
    soil input field(s): ['ST000007', ...]`` -- four frames deep, on a
    cache that was complete.  MEASURED 2026-08-18 on the Linux shakeout's
    prepared-cache restore arm, where the solved soil was sitting in
    ``surface/`` the whole time.

    ``surface`` is therefore consumed when the caller has one and derived
    only when it does not.  A state holding NEITHER is refused here by
    name rather than downstream by KeyError.
    """

    from types import MappingProxyType, SimpleNamespace

    if surface is not None:
        return surface

    from woof.ingest.ruc_soil import preprocess_land_surface_soil
    from woof.native_wrf_contract import canonical_noah_surface

    try:
        fields = met.fields
    except AttributeError as exc:
        raise TypeError(
            "prepared physics meteorology must expose a fields mapping"
        ) from exc
    absent = [name for name in _NATIVE_SOIL_FIELDS if name not in fields]
    if absent:
        raise ValueError(
            "this prepared state carries no solved Noah surface and no "
            f"native soil to derive one from ({', '.join(absent)} absent "
            "from its met fields), so the land model has no soil state to "
            "start from and the forecast would run on fabricated ground.\n"
            "  A prepared cache written by this release stores the solved "
            "surface (surface/TSLB, SMOIS, SH2O and the rest of the "
            "canonical Noah inventory) and drops the native SOILT/SOILW "
            "pair, so a restore must pass restore_prepared_cache(...)"
            ".surface through to this call rather than re-deriving.\n"
            "  A cache carrying neither is incomplete and must be prepared "
            "again; nothing here can reconstruct soil that was never "
            "written.")
    from woof.core.landuse import (
        ruc_fractional_seaice as _ruc_fractional_seaice,
        seaice_threshold_of as _seaice_threshold_of)
    soil = preprocess_land_surface_soil(
        fields, sf_surface_physics=int(cfg.sf_surface_physics),
        # real.exe's adjust_for_seaice_pre/post keep the fraction under
        # fractional_seaice = 1 (threshold 0.02) and snap to 0/1 at 0.5
        # otherwise (module_soil_pre.F:216-219, :337-343, :392-393 of the HRRR
        # v4.1.21 fork).
        fractional_seaice=_ruc_fractional_seaice(cfg),
        seaice_threshold=_seaice_threshold_of(cfg),
        # The RESOLVED count, the same number :func:`_soil_shape` above
        # allocates from.  Left off, RUC's soil ingest took its own
        # nine-level default, so a six-level config prepared a nine-level
        # surface and met a six-level allocation -- a shape error four
        # frames down instead of a forecast, which is the difference
        # between "six levels is unverified" and "six levels is
        # unreachable on this route".
        num_soil_layers=soil_layer_count(cfg),
        soil_type=static["SCT_DOM"],
        deep_soil_temperature=static["SOILTEMP"], soil_mesh=soil_mesh,
        water_temperature=getattr(met, "water_temperature", None),
        water_temperature_policy=(getattr(met, "water_temperature_receipt", None)
                                  or {}).get("policy"))
    return SimpleNamespace(fields=MappingProxyType(
        canonical_noah_surface(soil)),
        soil_texture_downscale=getattr(soil, "soil_texture_downscale", None),
        soil_temperature_repair=getattr(soil, "soil_temperature_repair", None))


_AUTO_SOIL_MESH = object()


def initialize_hrrr_physics(
        result, cfg, met, static, attrs, grid, valid_time, *,
        constant_glw_wm2=None, surface=None, soil_mesh=_AUTO_SOIL_MESH,
        p_top=None, column_chunk=None, trace_gas_overrides=None, ozone_parent=None, cam_ozone=None):
    """Initialize physics, accepting either host or device ingestion fields.

    ``surface`` is the solved Noah surface when the caller already holds
    one -- what :func:`woof.ingest.prepared_cache.restore_prepared_cache`
    hands back.  Omitted, the surface is derived from the met snapshot's
    native soil, which is what a freshly decoded state carries.  See
    :func:`resolve_prepared_noah_surface` for why the two roads differ.
    Omitting ``soil_mesh`` keeps automatic planning; explicitly passing
    ``None`` preserves the caller's decision to disable soil downscaling.
    """

    if soil_mesh is _AUTO_SOIL_MESH:
        soil_mesh = None
        if surface is None:
            from woof.ingest.hrrr import hrrr_source_grid
            from woof.ingest.soil_downscale import soil_mesh_plan_from_case
            soil_mesh = soil_mesh_plan_from_case(
                None, grid, source_grid=hrrr_source_grid())
    surface = resolve_prepared_noah_surface(
        met, cfg, static, surface=surface, soil_mesh=soil_mesh)
    landuse_attrs = {
        name: attrs[name] for name in ("MMINLU", "ISWATER", "ISLAKE", "ISICE")
    }
    return initialize_prepared_physics(
        result, cfg, met, surface, static, landuse_attrs, grid, valid_time,
        center_lat=attrs["CEN_LAT"], constant_glw_wm2=constant_glw_wm2,
        p_top=p_top, column_chunk=column_chunk, trace_gas_overrides=trace_gas_overrides,
        ozone_parent=ozone_parent,
        **({"cam_ozone": cam_ozone} if cam_ozone is not None else {}))


__all__ = ["initialize_hrrr_physics", "initialize_prepared_physics",
           "resolve_prepared_noah_surface", "initial_vegetation_fraction"]
