"""Table-admitted single-moment column microphysics.

A bulk scheme whose WRF driver call is "one column at a time: sediment,
then the process network, on th/qv and a fixed set of mass species, with
WRF's standard rain/snow/graupel surface returns" is admitted to WOOF as
ONE ROW of :data:`MP_COLUMN_SCHEMES`, not as an adapter module.  The row
names the scheme's CUDA unit and entry, the moist species it integrates,
the RunConfig switches that select its variants (handed to the kernel as
int32 in row order), the host constant table it reads, and its
reflectivity formulation.  :func:`apply` is the same for every row: WRF's
``moist_physics_prep_em`` inputs (module_big_step_utilities_em.F:5476-
5547), the column launch, the output-due reflectivity on the post-call
temperature, and ``moist_physics_finish``.

The kernel ABI every row implements (``kernels/gsfcgce.cu`` is the first):

    entry(th, <species...>, rho, pii, p, z, dz8w, ht,
          rainnc, rainncv, snownc, snowncv, graupelnc, graupelncv, sr,
          float* sediment_vt, const float* table, float dt, float grav, float rhowater,
          float rhosnow, int <switches...>, int nz, int ny, int nx)

with ``th`` full potential temperature updated in place, ``rho`` dry density
(kg m-3), ``p`` full pressure (Pa), ``z``/``dz8w`` mass-level height and
layer depth (m), ``ht`` terrain height (m), and the surface fields WRF's
microphysics driver passes (accumulated mm, per-call mm, frozen fraction).

Admitting a further single-moment scheme of this shape is a row here, a
kernel, a constant table, and its registry/config rows -- the arbitrary
acceptance test.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from importlib import import_module

import numpy as np

from woof.core import constants as c
from woof.core.device_cache import cuda_cache

_COLUMN_TPB = 64


@dataclass(frozen=True)
class ColumnScheme:
    """One table-admitted scheme."""

    mp_physics: int
    name: str
    registry_package: str
    #: CUDA translation unit and ``extern "C"`` entry of the column kernel.
    unit: str
    entry: str
    #: Moist species the kernel updates, in kernel argument order.
    species: tuple[str, ...]
    #: RunConfig fields passed to the kernel as int32, in order.
    switches: tuple[str, ...]
    #: ``"module:function"`` returning the float32 constant table for the
    #: switch values (called with them as keyword arguments it accepts).
    table: str
    #: ``"module:function"`` returning the reflectivity parameters
    #: (fixed-intercept exponential PSDs) for the switch values.
    radar: str
    #: Reflectivity unit and entry, and which species feed it per switch
    #: state: ``refl_species(switch_values) -> (use_qs, use_qg)``.
    refl_unit: str
    refl_entry: str
    refl_species: str
    #: The algorithm identity a restart binds (checkpoint_identity).
    restart_identity: str
    #: ``(minimum, maximum)`` levels: the kernel reads the second level's
    #: density, and the reflectivity unit's REFL_KMAX ceiling is 256.
    vertical_level_bounds: tuple[int, int] = (2, 256)


MP_COLUMN_SCHEMES: dict[int, ColumnScheme] = {
    97: ColumnScheme(
        mp_physics=97,
        name="Goddard GCE three-ice",
        registry_package="gsfcgcescheme",
        unit="gsfcgce",
        entry="gsfcgce_column",
        species=("qv", "qc", "qr", "qi", "qs", "qg"),
        switches=("gsfcgce_hail", "gsfcgce_2ice"),
        table="woof.core.gsfcgce_constants:table",
        radar="woof.core.gsfcgce_constants:radar_parameters",
        refl_unit="gsfcgce_refl",
        refl_entry="gsfcgce_refl_column",
        refl_species="woof.core.gsfcgce_constants:refl_species",
        restart_identity="gsfcgce-three-ice-wrf-v4.6.1-v1",
        vertical_level_bounds=(2, 256),
    ),
}


def table_schemes_carrying(*species: str) -> tuple[int, ...]:
    """The rows whose integrated species include every one of ``species``."""
    return tuple(sorted(mp for mp, row in MP_COLUMN_SCHEMES.items()
                        if set(species) <= set(row.species)))


#: Every row.  Each implements the column ABI above, so each returns WRF's
#: seven surface fields (rainnc/rainncv/snownc/snowncv/graupelnc/
#: graupelncv/sr), stages its own REFL_10CM, and computes no effective
#: radii (the ABI has no radius output, so WRF's has_reqc/i/s are 0 for a
#: row by construction).  The hand-written membership tests that ask one
#: of those three questions add this tuple instead of a new literal.
TABLE_SCHEMES = tuple(sorted(MP_COLUMN_SCHEMES))

#: The rows that carry WRF's six-mass moist package
#: (``moist:qv,qc,qr,qi,qs,qg``), read off each row's species.  State
#: allocation, transport, nest forcing and every consumer keyed on "the
#: scheme has qi, qs and qg" add this tuple, so a row's hydrometeor set --
#: not a literal -- decides them.
SIX_MASS_TABLE_SCHEMES = table_schemes_carrying("qi", "qs", "qg")


def _resolve(path: str):
    module, name = path.split(":")
    return getattr(import_module(module), name)


def switch_values(row: ColumnScheme, cfg) -> tuple[int, ...]:
    """The row's switches read from ``cfg`` (int, row order)."""
    return tuple(int(getattr(cfg, name)) for name in row.switches)


def _switch_kwargs(row: ColumnScheme, values) -> dict[str, int]:
    return dict(zip(row.switches, (int(v) for v in values)))


def _call(path: str, row: ColumnScheme, values):
    """Call a row's provider with the switches its signature names."""
    fn = _resolve(path)
    accepted = inspect.signature(fn).parameters
    kwargs = _switch_kwargs(row, values)
    return fn(**{k: v for k, v in kwargs.items() if k in accepted})


def host_table(row: ColumnScheme, values) -> np.ndarray:
    """The float32 constant table for one switch state (host copy)."""
    return np.asarray(_call(row.table, row, values), dtype=np.float32)


@cuda_cache(maxsize=None, ready=True)
def device_table(mp_physics: int, values: tuple[int, ...]):
    import cupy as cp
    return cp.asarray(host_table(MP_COLUMN_SCHEMES[mp_physics], values))


def launch_column(row: ColumnScheme, th, species: dict, rho, pii, p, z,
                  dz8w, ht, rainnc, rainncv, snownc, snowncv, graupelnc,
                  graupelncv, sr, dt: float, values: tuple[int, ...], *,
                  sediment_vt=None) -> None:
    """One launch of ``row``'s column kernel over contiguous float32 fields."""
    import cupy as cp
    from woof.core.kernels import get_kernel
    from woof.core.state import DTYPE

    nz, ny, nx = th.shape
    if sediment_vt is None:
        sediment_vt = cp.empty_like(th)
    if nz < 2:
        raise ValueError(f"{row.name} needs at least 2 levels (it reads "
                         f"the second level's density), got nz={nz}")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    vol = {"th": th, **{s: species[s] for s in row.species}, "rho": rho,
           "pii": pii, "p": p, "z": z, "dz8w": dz8w,
           "sediment_vt": sediment_vt}
    for name, a in vol.items():
        if a.shape != (nz, ny, nx) or a.dtype != DTYPE \
                or not a.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous float32 "
                             f"{(nz, ny, nx)}, got {a.dtype} {a.shape}")
    surf = {"ht": ht, "rainnc": rainnc, "rainncv": rainncv,
            "snownc": snownc, "snowncv": snowncv, "graupelnc": graupelnc,
            "graupelncv": graupelncv, "sr": sr}
    for name, a in surf.items():
        if a.shape != (ny, nx) or a.dtype != DTYPE \
                or not a.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous float32 "
                             f"{(ny, nx)}, got {a.dtype} {a.shape}")
    table = device_table(row.mp_physics, tuple(int(v) for v in values))
    kernel = get_kernel(row.unit, row.entry)
    blocks = (ny * nx + _COLUMN_TPB - 1) // _COLUMN_TPB
    kernel((blocks,), (_COLUMN_TPB,),
           (th, *(species[s] for s in row.species), rho, pii, p, z, dz8w,
            ht, rainnc, rainncv, snownc, snowncv, graupelnc, graupelncv, sr,
            sediment_vt, table, DTYPE(dt), DTYPE(c.G), DTYPE(c.RHOWATER), DTYPE(100.0),
            *(np.int32(v) for v in values), np.int32(nz), np.int32(ny),
            np.int32(nx)))
    del cp


def launch_refl(row: ColumnScheme, th, pii, p, qv, qr, qs, qg, refl,
                values: tuple[int, ...]) -> None:
    """``row``'s REFL_10CM (dBZ) from the post-call column."""
    from woof.core.kernels import get_kernel_int_defines
    from woof.core.refl import _device_tables, kernel_capacity, radar_init

    nz, ny, nx = th.shape
    rp = _call(row.radar, row, values)
    use_qs, use_qg = _call(row.refl_species, row, values)
    if row.mp_physics == 97:
        from woof.core.gsfcgce_constants import native_radar_materials
        rc = native_radar_materials()
    else:
        rc = radar_init()
    kernel = get_kernel_int_defines(row.refl_unit, row.refl_entry,
                                    (("REFL_KMAX", kernel_capacity(nz)),))
    blocks = (ny * nx + _COLUMN_TPB - 1) // _COLUMN_TPB
    f32 = np.float32
    kernel((blocks,), (_COLUMN_TPB,),
           (th, pii, p, qv, qr, qs, qg, _device_tables(),
            np.float64(rc.k_w), np.float64(rc.m_w_0.real),
            np.float64(rc.m_w_0.imag), np.float64(rc.m_i_0.real),
            np.float64(rc.m_i_0.imag), np.float64(rp["pi5"]),
            f32(rp["xam_r"]), f32(rp["xam_s"]), f32(rp["xam_g"]),
            f32(rp["xcg3"]), f32(rp["xcg4"]), f32(rp["xcre1"]),
            f32(rp["xcre4"]), f32(rp["xocms"]), f32(rp["xocmg"]),
            f32(rp["n0g"]), np.int32(use_qs), np.int32(use_qg), refl,
            np.int32(nz), np.int32(ny), np.int32(nx)))


def apply(state, cfg, dt: float, *, refl_10cm_due: bool = False):
    """Prepare WRF's microphysics inputs, run the row for ``cfg.mp_physics``,
    and finish with WRF's retained microphysics heating."""
    import cupy as cp

    from woof.core.microphysics import (MicrophysicsDiagnostics,
                                         moist_physics_finish,
                                         save_pre_mp_theta)
    from woof.core.state import DTYPE

    row = MP_COLUMN_SCHEMES[int(cfg.mp_physics)]
    missing = [s for s in row.species if getattr(state, s, None) is None]
    if missing:
        raise ValueError(f"mp_physics={row.mp_physics} ({row.name}) state "
                         "lacks " + ", ".join(missing))
    values = switch_values(row, cfg)
    nz, ny, nx = state.p.shape
    thb = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
    phb = state.phb if state.phb.ndim == 3 else state.phb[:, None, None]
    th = state.scratch((nz, ny, nx), "mp_th")
    rho = state.scratch((nz, ny, nx), "mp_rho")
    pii = state.scratch((nz, ny, nx), "mp_pii")
    zh = state.scratch((nz, ny, nx), "mp_z")
    dz8w = state.scratch((nz, ny, nx), "mp_dz8w")
    z8w = state.scratch((nz + 1, ny, nx), "mp_z8w")
    th[...] = (state.total_theta() if getattr(state, "use_theta_m", 0)
                   else thb + state.thp)
    rho[...] = 1.0 / state.alt                   # dry density 1/(al+alb)
    pii[...] = cp.power(state.p / DTYPE(c.P0), DTYPE(c.RCP))
    z8w[...] = (phb + state.php) / DTYPE(c.G)
    zh[...] = 0.5 * (z8w[:nz] + z8w[1:])
    dz8w[...] = z8w[1:] - z8w[:nz]

    surface = (ny, nx)
    rainnc = state.scratch(surface, "mp_rainnc")
    rainncv = state.scratch(surface, "mp_rainncv")
    snownc = state.scratch(surface, "mp_snownc")
    snowncv = state.scratch(surface, "mp_snowncv")
    graupelnc = state.scratch(surface, "mp_graupelnc")
    graupelncv = state.scratch(surface, "mp_graupelncv")
    sr = state.scratch(surface, "mp_sr")
    save_pre_mp_theta(state)                     # WRF moist_physics_prep_em
    launch_column(row, th, {s: getattr(state, s) for s in row.species},
                  rho, pii, state.p, zh, dz8w, state.ht, rainnc, rainncv,
                  snownc, snowncv, graupelnc, graupelncv, sr, dt, values,
                  sediment_vt=state.scratch((nz, ny, nx), "mp_column_vt"))
    if refl_10cm_due:
        from woof.core.refl import stash_refl_10cm
        refl = state.scratch((nz, ny, nx), "refl_10cm")
        launch_refl(row, th, pii, state.p, state.qv, state.qr, state.qs,
                    state.qg, refl, values)
        stash_refl_10cm(state, refl)
    moist_physics_finish(state, cfg, th, dt)
    return MicrophysicsDiagnostics(
        rainnc=rainnc, rainncv=rainncv, sr=sr, snownc=snownc,
        snowncv=snowncv, graupelnc=graupelnc, graupelncv=graupelncv)


def reflectivity(state, cfg, *, temperature=None, pressure=None):
    """H_Z(x) for a table-admitted scheme, between steps, state untouched.

    The scheme's own REFL_10CM routine on the state's current columns:
    dry theta from the state's prognostic carrier, Exner from its full
    pressure, and the row's species -- the same call the forecast step makes on its post-call
    column.  ``temperature``/``pressure`` are refused rather than half
    honoured: the routine forms its temperature from theta and Exner.
    Device states only; the result is the state's ``refl_10cm`` scratch
    slot (copy it to keep it).
    """
    import cupy as cp

    from woof.core.state import DTYPE

    if temperature is not None or pressure is not None:
        raise ValueError(
            "a table-admitted scheme's reflectivity forms its temperature "
            "from the state's full theta and Exner, as the step does; it "
            "takes no temperature/pressure pair")
    if not hasattr(state.p, "get"):
        raise NotImplementedError(
            f"mp_physics={int(cfg.mp_physics)} reflectivity runs on the "
            "device only: its routine is the CUDA unit the forecast step "
            "launches, and no host mirror of it exists to run instead")
    row = MP_COLUMN_SCHEMES[int(cfg.mp_physics)]
    nz, ny, nx = state.p.shape
    thb = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
    th = cp.empty((nz, ny, nx), dtype=DTYPE)
    th[...] = (state.total_theta() if getattr(state, "use_theta_m", 0)
                   else thb + state.thp)
    pii = cp.power(state.p / DTYPE(c.P0), DTYPE(c.RCP))
    out = state.scratch((nz, ny, nx), "refl_10cm")
    launch_refl(row, th, cp.ascontiguousarray(pii), state.p, state.qv,
                state.qr, state.qs, state.qg, out, switch_values(row, cfg))
    return out


__all__ = ["ColumnScheme", "MP_COLUMN_SCHEMES", "SIX_MASS_TABLE_SCHEMES",
           "TABLE_SCHEMES", "apply", "device_table", "host_table",
           "launch_column", "launch_refl", "reflectivity", "switch_values",
           "table_schemes_carrying"]
