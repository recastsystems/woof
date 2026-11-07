"""Table-driven runner for WRF's single-moment WSM family column kernels.

A WSM-family scheme is admitted by a ROW in
``woof/data/wsm_family/schemes.json`` -- its CUDA unit and entry points,
the mass species the kernel reads and writes in its argument order, the
precipitating classes whose surface accumulators it carries, its compiled
level tiers and the WRF words file its constants come from -- not by a
per-scheme adapter.  This module turns any row into a launch: it validates
the fields, orders the kernel arguments from the row, compiles the unit and
binds the frozen WRF words.  ``mp_physics = 24`` (WSM7) is the first row.

Arithmetic.  Every unit listed here is graded word for word against
gfortran -O0 WRF, so it compiles with ``--fmad=false --ftz=false`` (no FMA
contraction, no flush to zero) in the DEFAULT build as well as under
``GPUWM_WRF_EXACT=1``: the two builds are the same arithmetic.  The compile
goes through NVRTC directly, as the other unflushed units do, because
``cupy.RawModule`` appends ``-ftz=true`` after the caller's options.

Constants.  The derived coefficients are the float32 words the scheme's
WRF init routine computes (its rgmma Gamma products included) and
radar_init's words, dumped by the column oracle
(tools/wsm7_wrf461_column_oracle) and frozen in the row's words file; the
oracle's agreement proves them, because a wrong word cannot reproduce WRF's
outputs bit for bit.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

_DATA = Path(__file__).resolve().parents[1] / "data" / "wsm_family"
#: Compile options of every WSM-family unit, default and strict alike.
COMPILE_OPTIONS = ("-std=c++17", "--fmad=false", "--ftz=false")
_COLUMN_TPB = 32


@dataclass(frozen=True)
class FamilyRow:
    mp_physics: int
    name: str
    label: str
    unit: str
    column_kernel: str
    refl_kernel: str
    kmax_define: str
    level_tiers: tuple[int, ...]
    min_levels: int
    local_frame_bytes: tuple[tuple[int, int], ...]
    kernel_species: tuple[str, ...]
    refl_species: tuple[str, ...]
    precip: tuple[str, ...]
    words_file: str
    init_words: tuple[str, ...]
    radar_f32_words: tuple[str, ...]
    radar_f64_words: tuple[str, ...]
    restart_identity: str
    declared_divergences: tuple[dict, ...]
    #: Correct-mode finding WB5 (woof/correct_mode.py): the RunConfig key
    #: that turns on the unit's conservative surface booking and the macro
    #: that compiles it (row ``conservative_surface``); empty when none.
    conservative_surface_key: str = ""
    conservative_surface_define: str = ""

    @property
    def vertical_level_bounds(self) -> tuple[int, int]:
        return (self.min_levels, max(self.level_tiers))

    @property
    def state_species(self) -> tuple[str, ...]:
        """Mass species the scheme adds to qv/qc/qr (allocation order)."""
        return tuple(s for s in self.kernel_species if s not in ("qc", "qr"))


@lru_cache(maxsize=1)
def _table() -> dict[int, FamilyRow]:
    raw = json.loads((_DATA / "schemes.json").read_text(encoding="utf-8"))
    rows = {}
    for key, r in raw["schemes"].items():
        rows[int(key)] = FamilyRow(
            mp_physics=int(key), name=r["name"], label=r["label"],
            unit=r["unit"], column_kernel=r["column_kernel"],
            refl_kernel=r["refl_kernel"], kmax_define=r["kmax_define"],
            level_tiers=tuple(int(v) for v in r["level_tiers"]),
            min_levels=int(r["min_levels"]),
            local_frame_bytes=tuple((int(k), int(v)) for k, v in
                                    r["local_frame_bytes"].items()),
            kernel_species=tuple(r["kernel_species"]),
            refl_species=tuple(r["refl_species"]),
            precip=tuple(r["precip"]), words_file=r["words_file"],
            init_words=tuple(r["init_words"]),
            radar_f32_words=tuple(r["radar_f32_words"]),
            radar_f64_words=tuple(r["radar_f64_words"]),
            restart_identity=r["restart_identity"],
            declared_divergences=tuple(r["declared_divergences"]),
            conservative_surface_key=r.get(
                "conservative_surface", {}).get("key", ""),
            conservative_surface_define=r.get(
                "conservative_surface", {}).get("define", ""))
    return rows


def family_selectors() -> frozenset[int]:
    """The ``mp_physics`` values this table admits."""
    return frozenset(_table())


def row(mp_physics: int) -> FamilyRow:
    try:
        return _table()[int(mp_physics)]
    except KeyError:
        raise ValueError(
            f"mp_physics={mp_physics} has no row in the WSM-family table "
            f"(woof/data/wsm_family/schemes.json admits "
            f"{sorted(_table())})") from None


@lru_cache(maxsize=None)
def _words(mp_physics: int) -> dict:
    r = row(mp_physics)
    return json.loads((_DATA / r.words_file).read_text(encoding="utf-8"))


def _f32(bits) -> np.ndarray:
    return np.asarray([int(b, 16) for b in bits], np.uint32).view(np.float32)


def _f64(bits) -> np.ndarray:
    return np.asarray([int(b, 16) for b in bits], np.uint64).view(np.float64)


def init_words(mp_physics: int) -> np.ndarray:
    """The kernel's ``coef`` array: the init routine's float32 words in the
    row's order."""
    r = row(mp_physics)
    w = _words(mp_physics)["init_f32"]
    return _f32([w[name] for name in r.init_words])


def radar_words(mp_physics: int) -> tuple[np.ndarray, np.ndarray]:
    """``(rf, rd)``: radar_init's float32 and float64 words, row order."""
    r = row(mp_physics)
    w = _words(mp_physics)
    f32 = _f32([w["radar_f32"][name] for name in r.radar_f32_words])
    f64 = []
    for name in r.radar_f64_words:
        value = w["radar_f64"][name]
        f64 += value if isinstance(value, list) else [value]
    return f32, _f64(f64)


_FORCED_TIER: int | None = None


def force_level_tier(tier: int | None) -> None:
    """Grade a deeper compiled tier than ``nz`` needs (oracle tooling only)."""
    global _FORCED_TIER
    _FORCED_TIER = None if tier is None else int(tier)


def level_tier(mp_physics: int, nz: int) -> int:
    """The shallowest compiled ``*_KMAX`` tier holding ``nz`` levels.

    One CUDA thread owns a column, so the tier sizes its per-thread arrays;
    compiling a tier rather than ``nz`` keeps a nested tree with several
    level counts to at most two modules.
    """
    r = row(mp_physics)
    nz = int(nz)
    lo, hi = r.vertical_level_bounds
    if not lo <= nz <= hi:
        raise ValueError(f"{r.label} requires {lo} <= nz <= {hi}, got {nz}")
    tier = next(t for t in sorted(r.level_tiers) if nz <= t)
    if _FORCED_TIER is not None:
        if _FORCED_TIER not in r.level_tiers:
            raise ValueError(f"forced tier {_FORCED_TIER} is not a compiled tier")
        # A requested test tier is a minimum. Deeper columns still need
        # their normal tier, so the 72-level fixture also runs at tier 80.
        tier = max(tier, _FORCED_TIER)
    return tier


def local_frame_bytes(mp_physics: int, nz: int) -> int:
    """Driver-measured ceiling for the tier this domain launches."""
    return dict(row(mp_physics).local_frame_bytes)[level_tier(mp_physics, nz)]


def preparation_slots(mp_physics: int) -> tuple[str, ...]:
    """Per-call driver buffers, all rebuilt before the column launch."""
    tag = row(mp_physics).name
    return tuple(f"{tag}_{field}"
                 for field in ("theta", "rho", "pii", "dz", "z8w"))


def module_key(unit: str, kmax_define: str, tier: int,
               extra_define: str = "") -> str:
    from woof.core.kernels import MODULE_KEY_ROOT
    extra = f",{extra_define}=1" if extra_define else ""
    return f"{MODULE_KEY_ROOT}:{unit}[{kmax_define}={tier}{extra}]"


@lru_cache(maxsize=None)
def _module(unit: str, kmax_define: str, tier: int, device: int,
            extra_define: str = ""):
    import cupy as cp
    from cupy.cuda import compiler
    from woof.certify.kernel_manifest import record_module
    from woof.core.kernels import module_source_int_defines
    from woof.kernel_compile_notice import observe_module_compile

    key = module_key(unit, kmax_define, tier, extra_define)
    # extra_define: the row's correct-mode macro (finding WB5).  Empty
    # compiles exactly the source this function compiled before it existed.
    defines = ((kmax_define, int(tier)),) + (
        ((extra_define, 1),) if extra_define else ())
    source = module_source_int_defines(unit, defines)
    with cp.cuda.Device(device), observe_module_compile(key):
        ptx, _ = compiler.compile_using_nvrtc(
            source, ("-std=c++17", "--fmad=false", "--ftz=false"), None,
            f"{unit}.cu")
        module = cp.cuda.function.Module()
        module.load(ptx.encode() if isinstance(ptx, str) else ptx)
    record_module(key, source=source,
                  options=("-std=c++17", "--fmad=false", "--ftz=false"),
                  module=None)
    return module


def _function(r: FamilyRow, symbol: str, nz: int,
              conservative_surface: bool = False):
    import cupy as cp
    tier = level_tier(r.mp_physics, nz)
    if conservative_surface and not r.conservative_surface_define:
        raise ValueError(f"{r.label} has no conservative surface arm")
    extra = r.conservative_surface_define if conservative_surface else ""
    return _module(r.unit, r.kmax_define, tier,
                   cp.cuda.runtime.getDevice(), extra).get_function(symbol)


@lru_cache(maxsize=None)
def _device_words(mp_physics: int, device: int):
    import cupy as cp
    with cp.cuda.Device(device):
        rf, rd = radar_words(mp_physics)
        return (cp.asarray(init_words(mp_physics)), cp.asarray(rf),
                cp.asarray(rd))


def _check(fields: dict, shape, dtype=np.float32) -> None:
    for name, value in fields.items():
        if value is None:
            raise ValueError(f"{name} is missing")
        if tuple(value.shape) != tuple(shape):
            raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
        if value.dtype != dtype:
            raise TypeError(f"{name} must be float32, got {value.dtype}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")


def launch_column(mp_physics: int, *, theta, qv, species: dict, rho, pii,
                  pressure, dz, surface: dict, effc, effi, effs,
                  dt: float, conservative_surface: bool = False) -> None:
    """One call of the scheme over contiguous float32 ``(nz, ny, nx)`` fields.

    ``species`` maps every name in the row's ``kernel_species`` to its
    field; ``surface`` holds ``<class>nc``/``<class>ncv`` for every class in
    the row's ``precip`` plus ``sr``, all float32 ``(ny, nx)``.
    ``conservative_surface`` is correct-mode finding WB5 (the row's
    ``conservative_surface`` key): the unit compiled with the row's macro.
    False is the matched WRF arithmetic.
    """
    import cupy as cp
    if not isinstance(conservative_surface, bool):
        raise TypeError("conservative_surface must be a bool, got "
                        f"{conservative_surface!r}")
    r = row(mp_physics)
    shape = tuple(theta.shape)
    if len(shape) != 3:
        raise ValueError(f"{r.label} fields must be 3-D, got {shape}")
    nz, ny, nx = shape
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    volume = {"theta": theta, "qv": qv, "rho": rho, "pii": pii,
              "pressure": pressure, "dz": dz, "effc": effc, "effi": effi,
              "effs": effs}
    volume.update({name: species.get(name) for name in r.kernel_species})
    _check(volume, shape)
    surface_names = [f"{c}nc{v}" for c in r.precip for v in ("", "v")] + ["sr"]
    _check({n: surface.get(n) for n in surface_names}, (ny, nx))
    coef, _rf, _rd = _device_words(r.mp_physics, cp.cuda.runtime.getDevice())
    blocks = (ny * nx + _COLUMN_TPB - 1) // _COLUMN_TPB
    args = ((theta, qv) + tuple(species[n] for n in r.kernel_species)
            + (rho, pressure, pii, dz)
            + tuple(surface[n] for n in surface_names)
            + (effc, effi, effs, coef, np.float32(dt),
               np.int32(nz), np.int32(ny), np.int32(nx)))
    _function(r, r.column_kernel, nz, conservative_surface)(
        (blocks,), (_COLUMN_TPB,), args)


def launch_refl10cm(mp_physics: int, *, qv, species: dict, t, pressure,
                    refl) -> None:
    """The scheme's REFL_10CM (dBZ, floored at -35) from post-call T and p."""
    import cupy as cp
    r = row(mp_physics)
    shape = tuple(refl.shape)
    if len(shape) != 3:
        raise ValueError(f"refl fields must be 3-D, got {shape}")
    nz, ny, nx = shape
    fields = {"qv": qv, "t": t, "pressure": pressure, "refl": refl}
    fields.update({n: species.get(n) for n in r.refl_species})
    _check(fields, shape)
    coef, rf, rd = _device_words(r.mp_physics, cp.cuda.runtime.getDevice())
    blocks = (ny * nx + _COLUMN_TPB - 1) // _COLUMN_TPB
    args = ((qv,) + tuple(species[n] for n in r.refl_species)
            + (t, pressure, coef, rf, rd, refl,
               np.int32(nz), np.int32(ny), np.int32(nx)))
    _function(r, r.refl_kernel, nz)((blocks,), (_COLUMN_TPB,), args)


def check_volume(fields, shape):
    """Common WSM3/6/6R and WDM6 column-array contract."""
    for name, value in fields.items():
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
        if value.dtype != np.float32:
            raise TypeError(f"{name} must be float32, got {value.dtype}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")


def check_surface(fields, shape):
    """These adapters retain their combined surface shape/dtype error."""
    for name, value in fields.items():
        if value.shape != shape or value.dtype != np.float32:
            raise ValueError(f"{name} must be float32 {shape}, got "
                             f"{value.dtype} {value.shape}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")


def prepare_fields(state, theta, rho, pii, dz, z8w, *, rounded_exner=False):
    """WRF driver arithmetic; only WSM6 uses the rounded physics Exner."""
    import cupy as cp
    from woof.core import constants as c
    from woof.core.state import DTYPE
    thb = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
    phb = state.phb if state.phb.ndim == 3 else state.phb[:, None, None]
    theta[...] = (state.total_theta() if getattr(state, "use_theta_m", 0)
               else thb + state.thp)
    rho[...] = 1.0 / state.alt
    if rounded_exner:
        from woof.core.microphysics import prepare_microphysics_exner
        prepare_microphysics_exner(state.p, pii)
    else:
        pii[...] = cp.power(state.p / DTYPE(c.P0), DTYPE(c.RCP))
    z8w[...] = (phb + state.php) / DTYPE(c.G)
    dz[...] = z8w[1:] - z8w[:-1]


def precipitation_surface(state, precip=("rain", "snow", "graupel")):
    """Named accumulators shared by the fixed-species column adapters."""
    shape = state.p.shape[1:]
    surface = {"rainnc": state.scratch(shape, "mp_rainnc"),
               "rainncv": state.scratch(shape, "mp_rainncv")}
    if "snow" in precip:
        surface.update(snownc=state.scratch(shape, "mp_snownc"),
                       snowncv=state.scratch(shape, "mp_snowncv"))
    if "graupel" in precip:
        surface.update(graupelnc=state.scratch(shape, "mp_graupelnc"),
                       graupelncv=state.scratch(shape, "mp_graupelncv"))
    surface["sr"] = state.scratch(shape, "mp_sr")
    return surface


def finish_fields(state, cfg, theta, pii, dt, surface, *, refl_10cm_due=False):
    """Post-call reflectivity, retained heating, and surface diagnostics."""
    from woof.core.microphysics import MicrophysicsDiagnostics, moist_physics_finish
    if refl_10cm_due:
        from woof.core.refl import compute_and_stash_refl_10cm
        refl_t = state.scratch(state.p.shape, "refl_t")
        refl_t[...] = theta * pii
        compute_and_stash_refl_10cm(state, cfg, refl_t, state.p)
    moist_physics_finish(state, cfg, theta, dt)
    return MicrophysicsDiagnostics(**surface)


def apply(state, cfg, dt: float, *, refl_10cm_due: bool = False):
    """Prepare WRF's driver fields, run the row's scheme, finish MP heating.

    The same preparation WRF's microphysics driver makes for every WSM
    scheme (theta = thb + thp, rho = 1/alt, pii = (p/p0)**rcp, dz from the
    geopotential), then the scheme call, then REFL_10CM from th*pii when
    output is due, then moist_physics_finish.
    """
    from woof.core.microphysics import save_pre_mp_theta

    r = row(cfg.mp_physics)
    nz, ny, nx = state.p.shape
    required = r.state_species + ("effc", "effi", "effs")
    missing = [n for n in required if getattr(state, n, None) is None]
    if missing:
        raise ValueError(f"mp_physics={r.mp_physics} state lacks {r.label} "
                         "fields: " + ", ".join(missing))
    tag = r.name
    theta = state.scratch((nz, ny, nx), f"{tag}_theta")
    rho = state.scratch((nz, ny, nx), f"{tag}_rho")
    pii = state.scratch((nz, ny, nx), f"{tag}_pii")
    dz = state.scratch((nz, ny, nx), f"{tag}_dz")
    z8w = state.scratch((nz + 1, ny, nx), f"{tag}_z8w")
    prepare_fields(state, theta, rho, pii, dz, z8w)
    surface = {}
    for cls in r.precip:
        surface[f"{cls}nc"] = state.scratch((ny, nx), f"mp_{cls}nc")
        surface[f"{cls}ncv"] = state.scratch((ny, nx), f"mp_{cls}ncv")
    surface["sr"] = state.scratch((ny, nx), "mp_sr")
    species = {n: getattr(state, n) for n in r.kernel_species}
    save_pre_mp_theta(state)
    launch_column(r.mp_physics, theta=theta, qv=state.qv, species=species,
                  rho=rho, pii=pii, pressure=state.p, dz=dz, surface=surface,
                  effc=state.effc, effi=state.effi, effs=state.effs, dt=dt,
                  conservative_surface=bool(r.conservative_surface_key) and bool(
                      getattr(cfg, r.conservative_surface_key, False)))
    if refl_10cm_due:
        from woof.core.refl import stash_refl_10cm
        refl_t = state.scratch((nz, ny, nx), "refl_t")
        refl_t[...] = theta * pii
        refl = state.scratch((nz, ny, nx), "refl_10cm")
        launch_refl10cm(r.mp_physics, qv=state.qv, species=species,
                       t=refl_t, pressure=state.p, refl=refl)
        stash_refl_10cm(state, refl)
    return finish_fields(state, cfg, theta, pii, dt, surface)


__all__ = ["COMPILE_OPTIONS", "FamilyRow", "apply", "family_selectors",
           "force_level_tier", "init_words", "launch_column",
           "launch_refl10cm", "level_tier", "module_key", "preparation_slots", "radar_words", "row"]
