"""Mayfield, KY, 2021-12-10/11: the tornado-scale LES attempt #1 case.

This module is the case's home outside its TOML. Under the standing rule
that a case name reaches only its config and its case module, everything
about *which storm and which ground* attempt #1 integrates is pinned here
and nowhere else in the package.

It registers as a ``script`` capability, not ``verify``: the graded
screens live in ``docs/les/ATTEMPT1-EXPECTATIONS.md`` and are computed
from a completed run's outputs, so there are no ``GATES`` here to be
checked before one exists. What :func:`main` does instead is the one
check that is useful *before* the run and cheap enough to run in CI: it
re-derives the ratified pins from the shipped config and refuses if the
two have drifted apart.

That drift is a real failure mode. The ratifications live in
``docs/superpowers/specs/P6-LES-DECISIONS-RATIFIED-2026-08-05.md`` as
prose, the geometry lives in the TOML as numbers, and nothing else
connects them -- so a later edit to either could leave the run
integrating something the owner never ruled on, with both documents
still reading plausibly.

This case replaced a Dodge City 2016-05-24 pick on 2026-08-06. The
sibling module ``les_tornado_dodgecity_20160524`` and its config remain
in the tree, PARKED behind an unadjudicated ingest question; see that
module's docstring.

Run it::

    python -m woof.verify.cases.les_tornado_mayfield_20211210 --help
    python -m woof.verify.cases.les_tornado_mayfield_20211210
    python -m woof.verify.cases.les_tornado_mayfield_20211210 \\
        --config PATH/TO/les_tornado_100m_mayfield_20211210.toml

The config is a repository file under ``configs/``, which gpuwm 2.5.0 and
later ship beside the package, so an install finds it with no argument.
Where it is missing -- an older install, or a config kept outside the
tree -- name it with ``--config`` or point ``WOOF_CONFIGS_ROOT`` at the
directory holding it; the refusal says so itself.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from woof.verify.cases import _repo_config

#: This module's dotted name, correct under ``python -m`` too.
MODULE = _repo_config.module_name(__name__, globals().get("__spec__"))

#: The name of the config this case is the module half of.
CONFIG_NAME = "les_tornado_100m_mayfield_20211210.toml"

#: Where that config lives in a source checkout.  A wheel does not ship
#: `configs/`, so this path is a DEFAULT and a display value, never an
#: assumption -- :func:`main` takes ``--config`` and refuses by name when
#: neither it nor ``WOOF_CONFIGS_ROOT`` finds the file.
CONFIG = _repo_config.default_path(CONFIG_NAME)

#: The owner ruling this case implements (session 3).
RATIFICATION = ("docs/superpowers/specs/"
                "P6-LES-DECISIONS-RATIFIED-2026-08-05.md")
#: The screens the run is graded on, registered before it.
EXPECTATIONS = "docs/les/ATTEMPT1-EXPECTATIONS.md"

#: The event, as ruled. Times UTC; the local date is 2021-12-10 (CST).
#:
#: The quad-state supercell: initiation in northeast Arkansas around 02Z
#: on 2021-12-11, a roughly 250 km long-track tornado through western
#: Kentucky, Mayfield struck near 03:27Z.
CASE_DAY_LOCAL = datetime(2021, 12, 10)
INITIATION_UTC = datetime(2021, 12, 11, 2)
MAYFIELD_STRUCK_UTC = datetime(2021, 12, 11, 3, 27)

#: The 100 m box centre, and the landmarks that fix it. The centre sits
#: about 6 km southwest of Mayfield, so the storm -- tracking WSW to ENE --
#: enters the box already mature and tornadic and transits past the town.
BOX_CENTER_LAT, BOX_CENTER_LON = 36.72, -88.70
MAYFIELD_LAT, MAYFIELD_LON = 36.7417, -88.6367
#: Upstream, inside d03 and outside d04: the ground the storm crosses while
#: the LES child grows its own turbulence around it.
CAYCE_LAT, CAYCE_LON = 36.5081, -89.0392
#: Convective initiation. Inside d01/d02, deliberately outside d03/d04 --
#: the parents resolve genesis, the LES children resolve the transit.
INITIATION_LAT, INITIATION_LON = 35.8909, -90.3437
#: Downstream, beyond this geometry: the track continues past what a fixed
#: 20 km box can hold, which is a documented limitation and not a defect.
PRINCETON_LAT, PRINCETON_LON = 37.1092, -87.8814
BREMEN_LAT, BREMEN_LON = 37.3620, -87.2308

#: The HRRR cycle. The 00Z analysis is the window start, so the tree
#: initializes from an analysis rather than a 6-hour-old forecast. Probed
#: 2026-08-06: ADMITS against the pinned native contract.
HRRR_CYCLE = datetime(2021, 12, 11, 0)
FORECAST_START_HOUR = 0
FORCING_HOURS = 8

#: The ratified chain: (grid_id, nx, ny, dx_m, dt_s, km_opt, bl_pbl_physics).
CHAIN = (
    (1, 306, 244, 3000.0, 15.0, 4, 1),
    (2, 450, 450, 1000.0, 5.0, 4, 1),
    (3, 300, 300, 500.0, 2.5, 3, 0),
    (4, 200, 200, 100.0, 0.5, 3, 0),
)

#: G2, ratified: nz = 72 with at least this many half levels below 1.7 km.
RATIFIED_NZ = 72
RATIFIED_BL_TOP_M = 1700.0
RATIFIED_MIN_LEVELS_BELOW_BL_TOP = 30

#: The integration window: 00Z -> 06Z, 6 h, the plan's floor.
START_TIME = datetime(2021, 12, 11, 0)
RUN_SECONDS = 21600.0
MIN_RUN_SECONDS = 21600.0

#: Non-zero, unlike the demonstration's, and a whole number of root steps.
RESTART_INTERVAL_S = 3600.0

#: P3 inflow seeding, pilot amplitude, fixed seed.
#:
#: d03 ONLY, ruled 2026-08-06 (G4 in
#: docs/superpowers/specs/P6-LES-DECISIONS-RATIFIED-2026-08-05.md).  The
#: generator takes its vertical extent from the parent-diagnosed PBLH, so it
#: needs a parent that PARAMETERIZES turbulence: d02 (YSU) -> d03 qualifies,
#: d03 -> d04 does not, because d03 is itself LES and its RESOLVED eddies are
#: already d04's inflow turbulence.  Seeding d04 would double-count them.
#: The model refuses the d04 case outright, which is its applicability
#: boundary rather than an obstacle.  Screen E1 -- the d04 D90 fetch receipt
#: -- stays graded and is the registered falsifier of this ruling.
INFLOW_DOMAINS = (3,)
INFLOW_SEED = 20211210
INFLOW_AMPLITUDE_SCALE = 1.0


def _levels_below(eta, p_top: float, bl_top: float) -> int:
    """Half levels below ``bl_top`` metres, through the shipped tool.

    ``tools`` IS a shipped package, so asking for it by name is the route
    that works from a wheel.  The path form below still resolves under an
    installed wheel -- ``<CONFIG>/..`` lands on site-packages and
    ``site-packages/tools`` exists -- but only by coincidence, and while it
    sits on ``sys.path[0]`` its 142 top-level module names shadow the
    standard library.  It stays only as a fallback for a checkout run whose
    repository root is not on ``sys.path``.
    """

    try:
        from tools.build_stretched_eta_ladder import score_ladder
    except ImportError:  # pragma: no cover - checkout-only fallback
        sys.path.insert(0, str(_repo_config.default_path(".").parent.parent
                               / "tools"))
        try:
            from build_stretched_eta_ladder import score_ladder
        finally:
            sys.path.pop(0)
    return score_ladder(list(eta), p_top=p_top, hybrid_opt=2, etac=0.2,
                        bl_top=bl_top)["levels_below_bl_top"]


def domain_grids(exp):
    """The four projected grids, for placement checks.

    Placement comes from the runner's own resolver
    (``grids_from_projection_config``: root on the projection reference,
    children through ``ProjectedGrid.nest`` with the experiment's
    ``i/j_parent_start``/ratio layout), NOT from re-deriving each domain
    centred on the reference.  The previous re-derivation ignored
    ``i_parent_start`` entirely; it agreed with the real geometry only
    because attempt #1's family happens to be exactly concentric, and it
    was wrong for any off-centre placement (recorded in
    docs/les/ATTEMPT2-EXPECTATIONS.md section 6; fixed under the
    off-centre-nest task after attempt #1's audit closed).
    """

    from woof.static.projection import grids_from_projection_config

    return {
        domain.grid_id: (grid, domain.run.nx, domain.run.ny)
        for domain, grid in zip(exp.domains,
                                grids_from_projection_config(exp))
    }


def contains(grids, grid_id: int, lat: float, lon: float) -> bool:
    grid, nx, ny = grids[grid_id]
    i, j = grid.latlon_to_ij(lat, lon)
    return 0.5 <= float(i) <= nx + 0.5 and 0.5 <= float(j) <= ny + 0.5


#: The placement contract, as a table rather than as prose: each landmark
#: and the domains that must and must not hold it.  This is what makes the
#: box a TRANSIT box rather than a genesis box, and it is checkable.
PLACEMENT = (
    ("Mayfield KY", MAYFIELD_LAT, MAYFIELD_LON, (1, 2, 3, 4), ()),
    ("box centre", BOX_CENTER_LAT, BOX_CENTER_LON, (1, 2, 3, 4), ()),
    ("Cayce KY (upstream fetch)", CAYCE_LAT, CAYCE_LON, (1, 2, 3), (4,)),
    ("initiation, NE Arkansas", INITIATION_LAT, INITIATION_LON, (1, 2),
     (3, 4)),
    ("Princeton KY (downstream)", PRINCETON_LAT, PRINCETON_LON, (1, 2, 3),
     (4,)),
    ("Bremen KY (beyond)", BREMEN_LAT, BREMEN_LON, (1, 2), (3, 4)),
)


def audit(config: Path | None = None) -> list[str]:
    """Every way the shipped config could have drifted from the ruling.

    ``config`` defaults to whichever readable copy :mod:`_repo_config`
    finds; :func:`main` resolves it and refuses before calling here, so
    reaching this function with nothing on disk is a programming error
    rather than a user one.
    """

    from woof.experiment import load_experiment

    if config is None:
        config = _repo_config.locate(CONFIG_NAME) or CONFIG
    config = Path(config)
    exp = load_experiment(str(config))
    bad: list[str] = []

    if exp.start_time != START_TIME:
        bad.append(f"start_time {exp.start_time} != ratified {START_TIME}")
    if exp.run_seconds != RUN_SECONDS:
        bad.append(f"run_seconds {exp.run_seconds} != ratified {RUN_SECONDS}")
    if exp.run_seconds < MIN_RUN_SECONDS:
        bad.append(f"run_seconds {exp.run_seconds} is under the plan's "
                   f"{MIN_RUN_SECONDS} s floor")
    end = exp.start_time.timestamp() + exp.run_seconds
    if end < MAYFIELD_STRUCK_UTC.timestamp():
        bad.append("the window ends before Mayfield is struck; d04 would "
                   "never see the transit it exists to resolve")
    if exp.start_time > MAYFIELD_STRUCK_UTC:
        bad.append("the window starts after Mayfield is struck")
    if exp.restart_interval_s != RESTART_INTERVAL_S:
        bad.append(f"restart_interval_s {exp.restart_interval_s} != "
                   f"{RESTART_INTERVAL_S}; the demonstration's 0.0 is the "
                   "defect this case exists not to repeat")

    proj = exp.projection
    if (proj is None or proj.ref_lat != BOX_CENTER_LAT
            or proj.ref_lon != BOX_CENTER_LON):
        bad.append("the projection reference is not the ruled box centre "
                   f"{BOX_CENTER_LAT}, {BOX_CENTER_LON}")

    if len(exp.domains) != len(CHAIN):
        bad.append(f"{len(exp.domains)} domains, ratified {len(CHAIN)}")
    else:
        for domain, (gid, nx, ny, dx, dt, km_opt, pbl) in zip(exp.domains,
                                                              CHAIN):
            run = domain.run
            got = (domain.grid_id, run.nx, run.ny, run.dx, run.dt,
                   run.km_opt, run.bl_pbl_physics)
            want = (gid, nx, ny, dx, dt, km_opt, pbl)
            if got != want:
                bad.append(f"d{gid:02d} resolves to {got}, ratified {want}")
            on = domain.grid_id in INFLOW_DOMAINS
            if bool(run.inflow_perturbation) is not on:
                bad.append(f"d{gid:02d} inflow_perturbation="
                           f"{run.inflow_perturbation}, ratified {on}")
            if on:
                if run.inflow_perturbation_seed != INFLOW_SEED:
                    bad.append(f"d{gid:02d} inflow seed "
                               f"{run.inflow_perturbation_seed} != "
                               f"{INFLOW_SEED}; the seed is committed so the "
                               "draw is reproducible")
                if (run.inflow_perturbation_amplitude_scale
                        != INFLOW_AMPLITUDE_SCALE):
                    bad.append(f"d{gid:02d} inflow amplitude "
                               f"{run.inflow_perturbation_amplitude_scale} "
                               f"!= {INFLOW_AMPLITUDE_SCALE}")

        grids = domain_grids(exp)
        for label, lat, lon, inside, outside in PLACEMENT:
            for gid in inside:
                if not contains(grids, gid, lat, lon):
                    bad.append(f"{label} is OUTSIDE d{gid:02d} and the "
                               "placement contract requires it inside")
            for gid in outside:
                if contains(grids, gid, lat, lon):
                    bad.append(f"{label} is INSIDE d{gid:02d} and the "
                               "placement contract requires it outside")

    vertical = exp.vertical
    if vertical.mass_level_count != RATIFIED_NZ:
        bad.append(f"nz {vertical.mass_level_count} != ratified {RATIFIED_NZ}")
    else:
        below = _levels_below(vertical.eta_levels, vertical.p_top,
                              RATIFIED_BL_TOP_M)
        if below < RATIFIED_MIN_LEVELS_BELOW_BL_TOP:
            bad.append(
                f"{below} half levels below {RATIFIED_BL_TOP_M:.0f} m, "
                f"ratified floor {RATIFIED_MIN_LEVELS_BELOW_BL_TOP}")

    # The [fetch] hints are validated at load and then dropped -- they are
    # advisory, so ExperimentConfig does not carry them -- and the cycle is
    # half the case's identity. Read them back off the file.
    import tomllib

    with config.open("rb") as handle:
        fetch = tomllib.load(handle).get("fetch", {})
    if fetch.get("source") != "hrrr":
        bad.append(f"fetch source {fetch.get('source')!r} is not hrrr")
    if fetch.get("cycle") != f"{HRRR_CYCLE:%Y-%m-%dT%H}":
        bad.append(f"fetch cycle {fetch.get('cycle')!r} != "
                   f"{HRRR_CYCLE:%Y-%m-%dT%H}")
    if fetch.get("forecast_start_hour") != FORECAST_START_HOUR:
        bad.append(f"forecast_start_hour "
                   f"{fetch.get('forecast_start_hour')!r} != "
                   f"{FORECAST_START_HOUR}")
    if fetch.get("hours") != FORCING_HOURS:
        bad.append(f"fetch hours {fetch.get('hours')!r} != {FORCING_HOURS}")
    return bad


def build_parser() -> argparse.ArgumentParser:
    """This module's command line.

    Built and parsed BEFORE anything is read off disk, so ``--help``
    answers on a machine that does not have the config -- which is every
    machine that installed a wheel.  It used to run the drift audit as
    the first statement of ``main``, so ``--help`` itself died on a
    missing file.
    """

    parser = argparse.ArgumentParser(
        prog=f"python -m {MODULE}",
        description=("Check the shipped Mayfield 2021-12-10 LES config "
                     "against the ratified pins.  This is the drift "
                     "audit, not the run: the run is "
                     "`woof run CONFIG.toml`."))
    parser.add_argument(
        "--config", type=Path, default=None, metavar="TOML",
        help=(f"the {CONFIG_NAME} to audit.  Omitted, it is looked for "
              f"under ${_repo_config.CONFIG_ROOT_ENV} and then beside "
              "the package; a wheel ships no `configs/`, so on an "
              "installed woof this flag is how the file is named"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = args.config
    if config is None:
        config = _repo_config.locate(CONFIG_NAME)
    if config is None:
        print(f"{MODULE.rsplit('.', 1)[-1]}: "
              + _repo_config.missing_config_message(CONFIG_NAME),
              file=sys.stderr)
        return 2
    config = Path(config)
    if not config.is_file():
        print(f"{MODULE.rsplit('.', 1)[-1]}: --config is not a readable "
              f"file: {config.resolve()}", file=sys.stderr)
        return 2
    bad = audit(config)
    print(f"case      : Mayfield, KY, {CASE_DAY_LOCAL:%Y-%m-%d} (local CST)")
    print(f"config    : {config}")
    print(f"ruling    : {RATIFICATION}")
    print(f"screens   : {EXPECTATIONS}")
    print(f"box centre: {BOX_CENTER_LAT} N, {abs(BOX_CENTER_LON)} W "
          f"(~6 km SW of Mayfield, which is inside the box)")
    print(f"window    : {START_TIME:%Y-%m-%d %HZ} + "
          f"{RUN_SECONDS / 3600:g} h, HRRR {HRRR_CYCLE:%Y-%m-%dT%H} "
          f"f{FORECAST_START_HOUR:02d}.."
          f"f{FORECAST_START_HOUR + FORCING_HOURS:02d}")
    print(f"transit   : initiation ~{INITIATION_UTC:%HZ} upstream, "
          f"Mayfield struck ~{MAYFIELD_STRUCK_UTC:%H:%MZ}")
    for gid, nx, ny, dx, dt, km_opt, pbl in CHAIN:
        print(f"  d{gid:02d}: {nx}x{ny} dx={dx / 1000:g} km dt={dt:g} s "
              f"km_opt={km_opt} bl_pbl_physics={pbl}")
    if bad:
        print("\nDRIFT -- the shipped config no longer matches the ruling:")
        for line in bad:
            print(f"  {line}")
        return 1
    print("\nthe shipped config matches every ratified pin, and the "
          "placement contract holds")
    return 0


__all__ = ["CONFIG", "CONFIG_NAME", "CASE_DAY_LOCAL", "CHAIN",
           "BOX_CENTER_LAT", "BOX_CENTER_LON", "HRRR_CYCLE", "RATIFIED_NZ",
           "PLACEMENT", "domain_grids", "contains", "audit", "build_parser",
           "main"]


if __name__ == "__main__":
    raise SystemExit(main())
