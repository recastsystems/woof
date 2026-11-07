"""Named microphysics schemes and their capabilities (CuPy-free).

A scheme WOOF adds that WRF does not number is selected by NAME (TOML
``mp_physics = "tempo"``) and carries a WOOF-side numeric id outside WRF's
``mp_physics`` range, so a WRF namelist number never means something
different in WOOF.  WOOF-side ids start at 900 (the ``bl_pbl_physics = 900``
SASE precedent).  Everything else about a scheme is a capability row here:
the allocator, transport, mass loading, reflectivity, surface coupling and
radiation consult these rows instead of a literal set of numbers, so the
next named scheme is a new row, not a new code path.

Existing WRF-numbered schemes still use their own (older) literal sets; the
capability functions below answer for the named schemes and return None for
any id they do not own, so callers write ``mp in OLD_SET or capability(mp)``
and an existing option's behaviour is unchanged by construction.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MicrophysicsScheme:
    name: str
    mp_id: int
    label: str
    #: mass species beyond qv/qc/qr (allocated, transported, mass-loaded)
    ice_mass_species: tuple[str, ...]
    #: number and volume moments (allocated, transported, positive-definite)
    moment_species: tuple[str, ...]
    #: aerosol number tracers (allocated, transported)
    aerosol_species: tuple[str, ...]
    #: 2-D surface emission fields
    surface_fields: tuple[str, ...]
    #: the scheme computes its own 10 cm reflectivity
    native_reflectivity: bool
    #: the scheme returns a frozen precipitation fraction (SR) for the LSM
    sr_available: bool
    #: effective-radius family for cloud optics (rrtmgp/rrtmg_legacy)
    cloud_optics_family: str
    #: the scheme is aerosol-aware (prognostic droplet number from CCN)
    aerosol_aware: bool
    #: restart algorithm identity string
    algorithm_identity: str
    #: a state field no other scheme allocates (presence-based consumers)
    discriminator: str
    diagnostic_species: tuple[str, ...] = ()


TEMPO = MicrophysicsScheme(
    name="tempo",
    mp_id=900,
    label="TEMPO (NCAR/TEMPO 9adf9ef, tempo_v3.0.3, RRFS v2), aerosol- and hail-aware",
    ice_mass_species=("qi", "qs", "qg"),
    moment_species=("nc", "nr", "ni", "ng", "volg"),
    aerosol_species=("nwfa", "nifa"),
    surface_fields=("nwfa2d",),
    native_reflectivity=True,
    sr_available=True,
    cloud_optics_family="thompson",
    aerosol_aware=True,
    algorithm_identity="tempo-9adf9ef-gpu-f64-v1",
    discriminator="volg",
)

CAMMGMP = MicrophysicsScheme(
    name="cammgmp", mp_id=11, label="CAM 5.1 Morrison-Gettelman (WRF CAMMGMP)",
    ice_mass_species=("qi", "qs"), moment_species=("nc", "ni", "nr", "ns"),
    aerosol_species=(), surface_fields=(), native_reflectivity=False,
    sr_available=True, cloud_optics_family="cam-mg", aerosol_aware=True,
    algorithm_identity="cammgmp-wrf471-stock-cody-v1", discriminator="cldfra_mp_all",
    diagnostic_species=("qndrop", "cldfra_old_mp", "cldfra_mp", "cldfra_mp_all",
        "lradius", "iradius", "cldfrai", "cldfral", "cldfra_conv", "wsedl3d",
        "rh_old_mp", "lcd_old_mp"))

NAMED_SCHEMES: dict[str, MicrophysicsScheme] = {s.name: s for s in (TEMPO, CAMMGMP)}
SCHEMES_BY_ID: dict[int, MicrophysicsScheme] = {s.mp_id: s for s in NAMED_SCHEMES.values()}
NAMED_MP_IDS: tuple[int, ...] = tuple(sorted(SCHEMES_BY_ID))


def resolve_mp_physics(value):
    """``"tempo"`` -> 900; integers and numeric strings pass through as int."""
    if isinstance(value, bool):
        raise ValueError(f"mp_physics must be a scheme number or name, got {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        key = value.strip().lower()
        if key in NAMED_SCHEMES:
            return NAMED_SCHEMES[key].mp_id
        try:
            return int(key)
        except ValueError:
            raise ValueError(
                f"mp_physics = {value!r} is not a scheme name WOOF knows; named schemes: "
                + ", ".join(sorted(NAMED_SCHEMES))) from None
    return value


def scheme(mp) -> MicrophysicsScheme | None:
    try:
        return SCHEMES_BY_ID.get(int(mp))
    except (TypeError, ValueError):
        return None


def scheme_for_state(state) -> MicrophysicsScheme | None:
    """The named scheme whose discriminator field the state allocated."""
    for s in NAMED_SCHEMES.values():
        if getattr(state, s.discriminator, None) is not None:
            return s
    return None


def mass_loading_species(mp) -> tuple[str, ...] | None:
    """Condensate masses in the acoustic mass loading (cq), beyond qc/qr."""
    s = scheme(mp)
    return None if s is None else s.ice_mass_species

# WRF Registry.EM_COMMON package rows carrying a qs mass species.
PORTED_WRF_SNOW_SELECTORS = (7, 95)
