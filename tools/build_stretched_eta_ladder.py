"""Build a surface-clustered eta ladder and score its boundary-layer levels.

WOOF-ArWen shares one vertical grid across every domain of a tree
(``woof/experiment.py:193`` rejects per-domain vertical keys), so an LES
child runs its root's levels.  The shipped 49-level ladder puts only 18
half levels below 1.7 km -- an effective dz of 96.7 m in the boundary
layer (``docs/public/LES.md:263-303``).  A resolved-turbulence child needs
more than that, and the only lever is the shared ladder itself.

This tool is the missing generator.  It builds ``nz + 1`` full levels from
a geometrically stretched layer-thickness profile, converts those heights
to WRF's analytic base-state dry pressure, normalises to eta, and then
scores the result THROUGH THE SAME PATH the published receipts use, so the
count it reports is the count the model will have.

The scoring path, and why it is exact at the reference column
-------------------------------------------------------------
Half-level dry pressure in WRF v4 is
``pd[k] = c3h[k]*(ps - p_top) + c4h[k] + p_top``.  With
``c4h = (znu - c3h)*(P0 - p_top)`` (``woof/core/grid.py:131``) this
collapses at ``ps = P0`` to ``pd[k] = znu[k]*(P0 - p_top) + p_top`` --
the hybrid coefficients cancel exactly.  So at the reference column the
half-level heights depend on ``hybrid_opt``/``etac`` not at all, and
``analytic_base_terrain_height`` (``grid.py:185``, WRF's own
``module_initialize_real.F:3787-3803`` inverted) turns them into metres
with no state, no sounding, and no GPU.

Verified against the published number: run with ``--certified-ladder`` and
the tool reports 18 half levels below 1700 m for the shipped 49-level
ladder, matching ``docs/public/LES.md:265-266``.

Generic by construction: no case, campaign, or configuration name appears
here or in anything it emits.

Usage
-----
    python tools/build_stretched_eta_ladder.py --nz 72 --dz0 20 \
        --dz-max 650 --bl-top 1700 --emit-toml
    python tools/build_stretched_eta_ladder.py --certified-ladder
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from woof.core import constants as c              # noqa: E402
from woof.core.grid import (                      # noqa: E402
    analytic_base_terrain_height,
    compute_hybrid_coeffs,
)
from woof.native_wrf_contract import CERTIFIED_ETA_LEVELS  # noqa: E402

#: WRF's analytic base-state lapse (``grid.py:148``, Registry ``base_lapse``).
BASE_LAPSE_K = 50.0


def analytic_base_pressure(z: np.ndarray, base_temp: float = 290.0
                           ) -> np.ndarray:
    """Forward of :func:`analytic_base_terrain_height`.

    ``p_s = P0*exp(-t00/a + sqrt((t00/a)^2 - 2*g*z/(a*Rd)))``, WRF
    ``module_initialize_real.F:3787-3803``.  Vectorised; float64.
    """

    z = np.asarray(z, dtype=np.float64)
    ratio = float(base_temp) / BASE_LAPSE_K
    radicand = ratio ** 2 - 2.0 * c.G * z / (BASE_LAPSE_K * c.RD)
    if np.any(radicand < 0.0):
        raise ValueError(
            "requested column depth exceeds the analytic base state's "
            "reach; lower --ztop or raise --p-top")
    return c.P0 * np.exp(-ratio + np.sqrt(radicand))


def stretched_thicknesses(nz: int, dz0: float, dz_max: float,
                          total: float) -> np.ndarray:
    """``nz`` layer thicknesses: geometric from ``dz0``, capped at ``dz_max``.

    The growth ratio is bisected so the thicknesses sum to ``total``
    exactly.  Capping at ``dz_max`` keeps the upper troposphere from
    running away once the near-surface spacing is set aggressively; the
    cap is what makes the profile a ramp-then-uniform rather than a pure
    geometric, which is the shape WRF's own auto-levels produce.
    """

    def build(ratio: float) -> np.ndarray:
        out = np.empty(nz, dtype=np.float64)
        dz = float(dz0)
        for k in range(nz):
            out[k] = min(dz, dz_max)
            dz *= ratio
        return out

    lo, hi = 1.0, 1.5
    if build(hi).sum() < total:
        raise ValueError(
            f"nz={nz} layers from dz0={dz0} m capped at dz_max={dz_max} m "
            f"cannot span {total:.1f} m even at ratio {hi}; raise --dz-max "
            f"or --dz0, or lower the model top")
    if build(lo).sum() > total:
        raise ValueError(
            f"nz={nz} uniform layers of dz0={dz0} m already exceed "
            f"{total:.1f} m; lower --dz0")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if build(mid).sum() < total:
            lo = mid
        else:
            hi = mid
    thick = build(0.5 * (lo + hi))
    return thick * (total / thick.sum())


def eta_from_heights(z_full: np.ndarray, p_top: float,
                     base_temp: float = 290.0) -> np.ndarray:
    """Full-level eta for a full-level height profile, endpoints pinned."""

    p_full = analytic_base_pressure(z_full, base_temp)
    eta = (p_full - p_top) / (p_full[0] - p_top)
    eta[0] = 1.0
    eta[-1] = 0.0
    return eta


def half_level_heights(eta: np.ndarray, *, p_top: float, hybrid_opt: int,
                      etac: float, base_temp: float = 290.0) -> np.ndarray:
    """Half-level heights in metres at the reference column."""

    eta = np.asarray(eta, dtype=np.float64)
    hy = compute_hybrid_coeffs(eta, hybrid_opt, etac, c.P0, p_top)
    pd_half = hy["c3h"] * (c.P0 - p_top) + hy["c4h"] + p_top
    return np.array([analytic_base_terrain_height(float(p), base_temp)
                     for p in pd_half])


def full_level_heights(eta: np.ndarray, *, p_top: float, hybrid_opt: int,
                       etac: float, base_temp: float = 290.0) -> np.ndarray:
    """Full-level heights in metres at the reference column.

    The full levels are where ``w`` lives and where a layer begins and
    ends, so these are the heights a LAYER THICKNESS is differenced from.
    Same reference-column identity as :func:`half_level_heights`, one
    stagger up.
    """

    eta = np.asarray(eta, dtype=np.float64)
    hy = compute_hybrid_coeffs(eta, hybrid_opt, etac, c.P0, p_top)
    pd_full = hy["c3f"] * (c.P0 - p_top) + hy["c4f"] + p_top
    return np.array([analytic_base_terrain_height(float(p), base_temp)
                     for p in pd_full])


def layer_thicknesses(eta: np.ndarray, *, p_top: float, hybrid_opt: int,
                      etac: float, base_temp: float = 290.0) -> np.ndarray:
    """The ``nz`` layer depths, full level to full level.

    This, and not the spacing between half levels, is the ``dz`` a
    vertical Courant number is formed against: ``w`` sits on the full
    levels that bound a layer, and the limiter asks how long a parcel
    takes to cross the LAYER.  The two agree where the ladder is uniform
    and part company by the stretch ratio where it is not; the first
    entry parts company by a factor of two whatever the ladder does,
    because differencing the half levels from the ground measures the
    distance to the middle of layer 0 rather than across it.
    """

    return np.diff(full_level_heights(
        eta, p_top=p_top, hybrid_opt=hybrid_opt, etac=etac,
        base_temp=base_temp))


def price_time_step(eta: np.ndarray, *, p_top: float, hybrid_opt: int,
                    etac: float, dt: float, w_max: float,
                    band: tuple[float, float],
                    target_courant: float = 0.8,
                    base_temp: float = 290.0) -> dict:
    """What this ladder costs in TIME STEP, which is the other half of it.

    A ladder is not a free choice of resolution.  The vertical Courant
    number of an updraft through a layer is ``w*dt/dz``, so halving the
    layer doubles it, and past 1 the engine's vertical-velocity limiter
    (``woof/core/dycore.py::apply_w_damping``, WRF ``w_damping = 1``)
    begins pushing the w tendency against the motion.  It is a limiter,
    not physics: the run stays up and the storm pays.

    WHAT BREAKAGE THIS PREVENTS (the gate law): a ladder refined
    to 200 m layers under the step that was chosen for 680 m ones.
    Measured on the card, same storm, same minute, same analysis: the
    49-level ladder ran that height band at 643 m layers and a Courant
    number of 0.79 with no cell over 1; the 80-level ladder ran it at
    198 m and 1.41 with 606 cells over 1, its strongest updraft halved
    from 33.9 m/s to 18.5, and its 45 dBZ area lived 25 minutes against
    53.  The ladder was right and nothing priced the step it needed.

    ``w_max`` is the updraft the caller expects; the default is the
    measured 35 m/s of a supercell, and a caller with a different storm
    states its own.  Returns the Courant number in the band, whether the
    ladder holds at ``dt``, and the step that would bring it to
    ``target_courant``.

    ``dz`` is the FULL-LEVEL layer thickness (:func:`layer_thicknesses`),
    because that is the distance the limiter asks a parcel to cross.
    Differencing the half levels instead measures half of layer 0 and,
    where the ladder stretches, a blend of two neighbouring layers: on
    the shipped 49-level ladder that read the thinnest convective layer
    as 614.90 m where it is 630.30 m, and priced its Courant number at
    0.8270 where the layer gives 0.8068.
    """

    z_half = half_level_heights(eta, p_top=p_top, hybrid_opt=hybrid_opt,
                                etac=etac, base_temp=base_temp)
    dz = layer_thicknesses(eta, p_top=p_top, hybrid_opt=hybrid_opt,
                           etac=etac, base_temp=base_temp)
    inside = (z_half >= float(band[0])) & (z_half <= float(band[1]))
    if not inside.any():
        raise ValueError(
            f"no half level lies in the band {band}; nothing to price")
    dz_band = float(dz[inside].min())
    courant = float(w_max) * float(dt) / dz_band
    return {
        "dt_s": float(dt),
        "w_max_ms": float(w_max),
        "band_m": [float(band[0]), float(band[1])],
        "thinnest_layer_in_band_m": dz_band,
        "mean_layer_in_band_m": float(dz[inside].mean()),
        "vertical_courant_in_band": courant,
        "target_courant": float(target_courant),
        "holds": bool(courant <= 1.0),
        "dt_for_target_s": float(target_courant) * dz_band / float(w_max),
        "limiter": "woof/core/dycore.py::apply_w_damping (WRF w_damping = 1)",
    }


def score_ladder(eta: np.ndarray, *, p_top: float, hybrid_opt: int,
                 etac: float, bl_top: float, base_temp: float = 290.0,
                 dt: float | None = None, w_max: float | None = None,
                 band: tuple[float, float] | None = None,
                 target_courant: float = 0.8) -> dict:
    """Half-level heights and boundary-layer level count for one ladder.

    Evaluated at the reference column ``ps = P0`` through
    :func:`compute_hybrid_coeffs`, so the hybrid identity above is
    exercised rather than assumed.
    """

    eta = np.asarray(eta, dtype=np.float64)
    hy = compute_hybrid_coeffs(eta, hybrid_opt, etac, c.P0, p_top)
    pd_half = hy["c3h"] * (c.P0 - p_top) + hy["c4h"] + p_top
    z_half = np.array([analytic_base_terrain_height(float(p), base_temp)
                       for p in pd_half])
    # Layer depths, not half-level spacings: the same distinction
    # price_time_step makes, kept here so one file reports one dz.
    dz_layer = layer_thicknesses(eta, p_top=p_top, hybrid_opt=hybrid_opt,
                                 etac=etac, base_temp=base_temp)
    below = int(np.count_nonzero(z_half < bl_top))
    # A step and a band, or no opinion at all: pricing a ladder against a
    # step nobody stated would mean inventing the updraft to price it with.
    price = None
    if dt is not None and w_max is not None and band is not None:
        price = price_time_step(
            eta, p_top=p_top, hybrid_opt=hybrid_opt, etac=etac, dt=dt,
            w_max=w_max, band=band, target_courant=target_courant,
            base_temp=base_temp)
    return {
        "time_step": price,
        "nz": int(eta.size - 1),
        "p_top_pa": float(p_top),
        "hybrid_opt": int(hybrid_opt),
        "etac": float(etac),
        "bl_top_m": float(bl_top),
        "levels_below_bl_top": below,
        "levels_below_2000m": int(np.count_nonzero(z_half < 2000.0)),
        "first_half_level_m": float(z_half[0]),
        "top_half_level_m": float(z_half[-1]),
        "mean_dz_below_bl_top_m": (float(bl_top / below) if below else None),
        "max_dz_below_bl_top_m": (float(dz_layer[:below].max())
                                  if below else None),
        "max_dz_m": float(dz_layer.max()),
        "half_level_heights_m": [round(float(v), 2) for v in z_half],
    }


def format_toml_array(eta: np.ndarray, per_line: int = 5,
                      indent: str = "    ") -> str:
    """The ``eta_levels = [...]`` body, in the shipped configs' layout."""

    cells = []
    for value in eta:
        if value == 1.0:
            cells.append("1.0")
        elif value == 0.0:
            cells.append("0.0")
        else:
            cells.append(f"{value:.5f}".rstrip("0"))
    lines = [indent + " ".join(f"{v}," for v in cells[i:i + per_line])
             for i in range(0, len(cells), per_line)]
    return "eta_levels = [\n" + "\n".join(lines) + "\n]"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nz", type=int, default=72,
                    help="mass levels; the ladder has nz+1 entries")
    ap.add_argument("--dz0", type=float, default=20.0,
                    help="thickness of the lowest layer, metres")
    ap.add_argument("--dz-max", type=float, default=650.0,
                    help="thickness cap, metres")
    ap.add_argument("--p-top", type=float, default=10000.0,
                    help="model top pressure, Pa")
    ap.add_argument("--bl-top", type=float, default=1700.0,
                    help="boundary-layer top to count levels below, metres")
    ap.add_argument("--hybrid-opt", type=int, default=2)
    ap.add_argument("--etac", type=float, default=0.2)
    ap.add_argument("--base-temp", type=float, default=290.0)
    ap.add_argument("--require-below", type=int, default=None,
                    help="exit 1 unless at least this many half levels "
                         "fall below --bl-top")
    ap.add_argument("--dt", type=float, default=None, metavar="SECONDS",
                    help="the parent time step this ladder will run at. "
                         "Given it, the tool prices the ladder's vertical "
                         "Courant number in the band and names the step "
                         "that would hold it; without it the tool has no "
                         "opinion about the step, because it would have "
                         "to invent the updraft to have one.")
    ap.add_argument("--w-max", type=float, default=35.0, metavar="M/S",
                    help="the updraft the ladder has to carry (default "
                         "35, the measured peak of a strong convective "
                         "storm)")
    ap.add_argument("--price-band", default=None, metavar="BOTTOM,TOP",
                    help="the height band the step is priced over, in "
                         "metres; the part of the column an updraft "
                         "occupies")
    ap.add_argument("--target-courant", type=float, default=0.8,
                    help="the vertical Courant number the named step aims "
                         "at (default 0.8, the value the shipped ladder "
                         "runs at)")
    ap.add_argument("--certified-ladder", action="store_true",
                    help="score the shipped 49-level ladder instead of "
                         "building one (the tool's own control)")
    ap.add_argument("--emit-toml", action="store_true",
                    help="print the eta_levels TOML block")
    ap.add_argument("--json", type=Path, default=None,
                    help="write the score to this path")
    args = ap.parse_args(argv)

    if args.certified_ladder:
        eta = np.asarray(CERTIFIED_ETA_LEVELS, dtype=np.float64)
        source = "woof/native_wrf_contract.py:CERTIFIED_ETA_LEVELS"
    else:
        depth = analytic_base_terrain_height(args.p_top, args.base_temp)
        thick = stretched_thicknesses(args.nz, args.dz0, args.dz_max, depth)
        z_full = np.concatenate(([0.0], np.cumsum(thick)))
        eta = eta_from_heights(z_full, args.p_top, args.base_temp)
        source = (f"stretched dz0={args.dz0} dz_max={args.dz_max} "
                  f"nz={args.nz} p_top={args.p_top}")

    if eta[0] != 1.0 or eta[-1] != 0.0:
        raise SystemExit("ladder endpoints are not 1.0 / 0.0")
    if np.any(np.diff(eta) >= 0.0):
        raise SystemExit("ladder is not strictly decreasing")

    price_band = None
    if args.price_band is not None:
        parts = [float(v) for v in args.price_band.split(",")]
        if len(parts) != 2 or parts[0] >= parts[1]:
            raise SystemExit(
                "--price-band takes BOTTOM,TOP in metres with BOTTOM below "
                f"TOP, not {args.price_band!r}")
        price_band = (parts[0], parts[1])
    if args.dt is not None and price_band is None:
        raise SystemExit(
            "--dt prices the ladder over a height band and none was "
            "given: state --price-band BOTTOM,TOP, the part of the column "
            "the updraft occupies")

    score = score_ladder(eta, p_top=args.p_top, hybrid_opt=args.hybrid_opt,
                         etac=args.etac, bl_top=args.bl_top,
                         base_temp=args.base_temp, dt=args.dt,
                         w_max=(args.w_max if args.dt is not None else None),
                         band=price_band,
                         target_courant=args.target_courant)
    score["source"] = source
    score["dz0_m"] = None if args.certified_ladder else args.dz0
    score["dz_max_m"] = None if args.certified_ladder else args.dz_max

    print(f"source                 : {source}")
    print(f"nz                     : {score['nz']} "
          f"({score['nz'] + 1} full levels)")
    print(f"first half level       : {score['first_half_level_m']:.2f} m")
    print(f"levels below {args.bl_top:.0f} m    : "
          f"{score['levels_below_bl_top']}")
    print(f"levels below 2000 m    : {score['levels_below_2000m']}")
    if score["mean_dz_below_bl_top_m"]:
        print(f"effective dz in the BL : "
              f"{score['mean_dz_below_bl_top_m']:.2f} m")
        print(f"largest dz in the BL   : "
              f"{score['max_dz_below_bl_top_m']:.2f} m")
    print(f"largest dz in the column: {score['max_dz_m']:.2f} m")
    print(f"top half level         : {score['top_half_level_m']:.1f} m")

    if score.get("time_step") is not None:
        t = score["time_step"]
        print()
        print(f"priced at dt           : {t['dt_s']:.2f} s, "
              f"updraft {t['w_max_ms']:.1f} m/s, band "
              f"{t['band_m'][0]:.0f} to {t['band_m'][1]:.0f} m")
        print(f"thinnest layer in band : "
              f"{t['thinnest_layer_in_band_m']:.1f} m")
        print(f"vertical Courant       : "
              f"{t['vertical_courant_in_band']:.2f}"
              f"  {'-- HOLDS' if t['holds'] else '-- OVER 1: the limiter'}"
              f"{'' if t['holds'] else ' (' + t['limiter'] + ') fires here'}")
        print(f"step for Courant {t['target_courant']:.2f}   : "
              f"{t['dt_for_target_s']:.2f} s")

    if args.emit_toml:
        print()
        print(format_toml_array(eta))

    if args.json is not None:
        args.json.write_text(json.dumps(score, indent=2) + "\n",
                             encoding="utf-8")

    if args.require_below is not None \
            and score["levels_below_bl_top"] < args.require_below:
        print(f"REFUSED: {score['levels_below_bl_top']} half levels below "
              f"{args.bl_top:.0f} m, required >= {args.require_below}",
              file=sys.stderr)
        return 1
    return 0


__all__ = [
    "analytic_base_pressure", "stretched_thicknesses", "eta_from_heights",
    "score_ladder", "format_toml_array", "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
