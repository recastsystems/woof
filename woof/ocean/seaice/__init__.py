"""GPU thermodynamic sea ice with a prescribed ocean heat boundary.

Original implementation of the zero-heat-capacity ice/snow column equations.
No Icepack code is included. See README.md for equations and scope.
"""
from pathlib import Path
import numpy as np
from woof.core.device_cache import cuda_cache

STATE = ("ice_volume", "ice_snow_mass", "ice_skin", "ice_ocean_mask",
         "ice_rain", "ice_snowfall", "ice_growth", "ice_melt",
         "ice_energy", "ice_energy_error", "ice_ocean_exchange", "ice_ocean_heat", "ice_thickness_change",
         "ice_concentration_change", "ice_ocean_owed", "ice_ocean_coupled")
OUTPUT = {"ice_thickness_change": ("ICETHICKCHANGE", "m"),
          "ice_concentration_change": ("ICECONCCHANGE", "1"),
          "ice_volume": ("ICEVOLUME", "m"),
          "ice_snow_mass": ("ICESNOWMASS", "kg m-2"),
          "ice_skin": ("ICETSK", "K"),
          "ice_growth": ("ICEGROWTH", "m"),
          "ice_melt": ("ICEMELT", "m"),
          "ice_energy": ("ICEENERGY", "J m-2"),
          "ice_energy_error": ("ICEENERGYERROR", "J m-2"),
          "ice_ocean_exchange": ("ICEOCEANEXCHANGE", "J m-2"),
          "ice_thickness": ("ICETHICK", "m"),
          "ice_snow_depth": ("ICESNOWDEPTH", "m")}

@cuda_cache(maxsize=8)
def _module():
    import cupy as cp
    from woof.certify.kernel_manifest import record_module
    source = Path(__file__).with_name("thermodynamics.cu").read_text(encoding="utf-8")
    options = ("-std=c++17",)
    mod = cp.RawModule(code=source, options=options)
    record_module("seaice_thermodynamics", source=source, options=options, module=mod)
    return mod

def validate(cfg):
    from woof.config import validate_seaice_parameters
    validate_seaice_parameters(cfg)

def _launch(name, fields, names, scalars=()):
    shape = fields["tsk"].shape
    arrays = []
    for key in names:
        a = fields[key]
        if a.shape != shape or a.dtype != np.float32 or not a.flags.c_contiguous:
            raise ValueError(f"sea ice {key} requires contiguous float32 {shape}")
        arrays.append(a)
    n = fields["tsk"].size
    _module().get_function(name)(((n+127)//128,), (128,),
        (*arrays, *(np.float32(x) for x in scalars), np.int32(n)))

def initialize(fields, cfg):
    """Activate by default only on a domain containing ocean ice.

    All state is in the existing restart field dictionary. A checkpoint restore
    overwrites this cold seed before any integration. Thickness is a declared
    prior unless supplied in fields['ice_thickness']; it is never called an
    observed thickness. Lake columns retain their existing lake owner.
    """
    import cupy as cp
    validate(cfg)
    from woof.core.noah_fractional_seaice import prescribed
    if prescribed(cfg):
        return
    if not cfg.seaice_prognostic or not cfg.sf_surface_physics:
        return
    mask = ((fields["xice"] > 0) & (fields["lakemask"] < .5))
    if not bool(cp.any(mask)):
        return
    for key in STATE:
        fields[key] = cp.zeros_like(fields["tsk"])
    for key, prior in (("ice_thickness", cfg.seaice_initial_thickness),
                       ("ice_snow_depth", 0)):
        if key not in fields:
            fields[key] = cp.full_like(fields["tsk"], prior)
    invalid = (~cp.isfinite(fields["ice_thickness"]) | (fields["ice_thickness"] <= 0)) & mask
    if bool(cp.any(invalid)):
        raise ValueError("sea ice thickness must be finite and positive over initial ocean ice")
    _launch("ice_initialize", fields,
        ("xice", "xland", "lakemask", "snow", "tsk", "sst", "ice_thickness",
         "ice_snow_depth", "ice_volume", "ice_snow_mass", "ice_skin",
         "ice_ocean_mask"))

COLD_SEEDED = frozenset(STATE) | {"ice_thickness", "ice_snow_depth"}


def prime_like(fields, domain_fields):
    """Give a tile buffer the ice carriers its domain holds; returns their names.

    :func:`initialize` activates the model only on a domain containing ocean
    ice, so the carriers are a property of the DOMAIN's data.  A tile buffer
    is built on neutral geography with no ice, so it never activated, and a
    streamed domain whose state holds them was refused at attach ("missing
    on tile ['fields/ice_concentration_change', ...]"; X12: the 5 km CONUS
    WRF-input door, whose namelist import leaves ``seaice_prognostic`` on,
    with ocean ice in the domain).  Zeros are enough: every carrier is
    gathered from the store before the buffer serves a tile, and every kernel
    here is column-local, so a tile steps its columns exactly as the domain.
    """
    import cupy as cp
    if "ice_volume" not in domain_fields or "ice_volume" in fields:
        return ()
    primed = []
    for key in sorted(COLD_SEEDED):
        if key in domain_fields and key not in fields:
            fields[key] = cp.zeros_like(fields["tsk"])
            primed.append(f"fields/{key}")
    return tuple(primed)


def cold_seed_in_place(fields, cfg, absent):
    """Seed an existing ice column in place, as :func:`initialize` seeds it.

    For a warm start from a WRF restart, which carries no state of this
    model: ``absent`` names the carriers the restart could not fill.  The
    arrays are the driver's own (bound into its checkpoint manifest), so
    they are written in place, never replaced.
    """
    if "ice_volume" not in fields:
        return
    for key in STATE:
        if key in absent:
            fields[key][...] = 0
    if "ice_thickness" in absent:
        fields["ice_thickness"][...] = cfg.seaice_initial_thickness
    if "ice_snow_depth" in absent:
        fields["ice_snow_depth"][...] = 0
    _launch("ice_initialize", fields,
        ("xice", "xland", "lakemask", "snow", "tsk", "sst", "ice_thickness",
         "ice_snow_depth", "ice_volume", "ice_snow_mass", "ice_skin",
         "ice_ocean_mask"))


def capture_precipitation(driver, atmosphere):
    f = driver.fields
    if "ice_volume" not in f:
        return
    from woof.core.physics import microphysics_scheme_sr_available
    ratio = driver.microphysics.sr if microphysics_scheme_sr_available(driver.mp_physics) else f["sr"]
    forcing = dict(f, ice_pending=driver._pending_rainbl,
                   ice_air=atmosphere["temperature"][0], ice_ratio=ratio)
    _launch("ice_precipitation", forcing,
        ("rainbl", "ice_pending", "ice_air", "ice_ratio", "ice_rain", "ice_snowfall"))

def step(driver, atmosphere, cfg):
    """After the LSM and before its energy ledger and PBL consume fluxes."""
    f = driver.fields
    if "ice_volume" not in f:
        return
    forcing = dict(f, ice_air=atmosphere["temperature"][0],
                   ice_qair=atmosphere["qv"][0])
    _launch("ice_step", forcing,
        ("xice", "ice_volume", "ice_snow_mass", "ice_skin", "ice_ocean_mask",
         "ice_rain", "ice_snowfall", "ice_growth", "ice_melt", "ice_energy",
         "ice_energy_error", "ice_ocean_exchange", "ice_ocean_heat", "ice_ocean_coupled",
         "ice_thickness_change", "ice_concentration_change", "ice_thickness", "ice_snow_depth",
         "tsk", "sst", "ice_air", "ice_qair", "psfc", "chs", "swdown", "glw",
         "hfx", "lh", "qfx", "qsfc", "albedo", "emiss", "snow", "snowh", "znt"),
        (driver.bldt_seconds, cfg.seaice_ocean_heat_flux,
         cfg.seaice_new_ice_thickness))

def history(fields):
    return {external: fields[key] for key, (external, _) in OUTPUT.items()
            if key in fields}

def blend_after_ocean(fields):
    if "ice_volume" in fields:
        _launch("ice_blend_ocean", fields, ("ice_ocean_mask", "xice", "ice_skin", "sst", "tsk"))


_FREEZE_COLUMN = ("ice_ocean_mask", "xice", "ice_volume", "ice_skin",
                  "ice_thickness", "ice_snow_mass", "ice_snow_depth", "ice_growth",
                  "ice_energy", "ice_energy_error", "ice_ocean_exchange",
                  "ice_thickness_change", "ice_concentration_change", "snow", "snowh",
                  "albedo", "znt", "sst")


def _freeze_pwp(fields, new_ice_thickness, water_capacity, dt):
    """Apply the ice step's ocean heat to the 3D PWP mixed layer and freeze it.

    The 3D PWP owner (sf_ocean_physics=2, the multi-day default) carries no
    OML ``tml``/``hml`` slab: its mixed layer is the level-major ``om_tmp``
    profile over the layer bottoms ``om_depth`` down to ``om_ml``. Reading the
    OML slab here was KeyError 'tml' at the first surface step of every run of
    48 h or longer with ice on any water the sea-ice model owns.

    Where the ice step set the surface fluxes (``ice_ocean_coupled``: ice
    present before or after it), ``ice_ocean_heat`` (W m-2, into the ocean)
    replaces the atmospheric surface heat, as on the OML route, so the
    ocean-to-ice heat flux leaves the PWP column; the PWP step skips its
    surface heat there. Ice-free water keeps the atmosphere's own fluxes.
    """
    shape = fields["tsk"].shape
    levels = fields["om_tmp"].shape[0]
    arrays = []
    for key in _FREEZE_COLUMN:
        a = fields[key]
        if a.shape != shape or a.dtype != np.float32 or not a.flags.c_contiguous:
            raise ValueError(f"sea ice {key} requires contiguous float32 {shape}")
        arrays.append(a)
    for key, want in (("om_tmp", (levels, *shape)), ("om_depth", (levels, *shape)),
                      ("om_ml", shape), ("_pwp_land", shape),
                      ("ice_ocean_heat", shape), ("ice_ocean_coupled", shape),
                      ("ice_ocean_owed", shape)):
        a = fields[key]
        if a.shape != want or a.dtype != np.float32 or not a.flags.c_contiguous:
            raise ValueError(f"3D PWP {key} requires contiguous float32 {want}")
        arrays.append(a)
    n = fields["tsk"].size
    _module().get_function("ice_ocean_freeze_pwp")(((n+127)//128,), (128,),
        (*arrays, np.float32(new_ice_thickness), np.float32(water_capacity),
         np.float32(dt), np.int32(levels), np.int32(n)))


def seed_pwp_profile(fields):
    """Raise a cold-started 3D PWP profile to Tf where the ice model owns water.

    ``fields`` is the ocean's own view (``tsk`` is the water SST under ice).
    Called by the PWP initialiser only on a cold start and only when this
    model is active; a checkpoint restore brings back the seeded profile.
    """
    if "ice_volume" not in fields or "om_tmp" not in fields:
        return
    shape = fields["tsk"].shape
    levels = fields["om_tmp"].shape[0]
    arrays = []
    for key, want in (("ice_ocean_mask", shape), ("_pwp_land", shape),
                      ("om_tmp", (levels, *shape)), ("om_tini", (levels, *shape)),
                      ("tsk", shape)):
        a = fields[key]
        if a.shape != want or a.dtype != np.float32 or not a.flags.c_contiguous:
            raise ValueError(f"3D PWP {key} requires contiguous float32 {want}")
        arrays.append(a)
    n = fields["tsk"].size
    _module().get_function("ice_seed_pwp_profile")(((n+127)//128,), (128,),
        (*arrays, np.int32(levels), np.int32(n)))


def reconcile_ocean(fields, cfg, *, dt=None):
    """After the ocean step: freeze supercooled water into the ice.

    ``dt`` (s) is the surface step the ocean just took. The 3D PWP route needs
    it: there the ice step's ``ice_ocean_heat`` rate is applied here.
    """
    from woof.surface_ocean import ocean_option
    constants = ocean_option(cfg.sf_ocean_physics)["constants"]
    if "ice_volume" in fields:
        capacity = constants["rhowater"]*constants["cwater"]
        if "om_tmp" in fields:
            if dt is None or not dt > 0:
                raise ValueError("reconcile_ocean needs the surface step dt on the 3D PWP route: "
                                 "without it the ice-ocean heat flux would never leave the ocean column")
            _freeze_pwp(fields, cfg.seaice_new_ice_thickness, capacity, dt)
        else:
            _launch("ice_ocean_freeze", fields, (*_FREEZE_COLUMN, "tml", "hml"),
                    (cfg.seaice_new_ice_thickness, capacity))
        blend_after_ocean(fields)
