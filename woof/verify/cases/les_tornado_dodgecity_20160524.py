"""Dodge City, KS, 2016-05-24: a PARKED tornado-scale LES case.

PARKED 2026-08-06, not withdrawn and not adjudicated. This was attempt
#1's case until it moved to Mayfield, KY 2021-12-10
(``les_tornado_mayfield_20211210``) specifically so that the HRRRv1
cloud-ice identity question would not need a ruling in order for the
attempt to proceed. The question is still open, the evidence for it is
measured and committed in ``docs/les/ATTEMPT1-INGEST-FINDING.md``, and no
alias was added to the native contract.

This module and its config stay in the tree so a future Dodge City
attempt starts from a ratified geometry rather than from scratch. The
audit below still runs and still passes; what it does NOT mean is that
the case is runnable, because the ingest route refuses the cycle.


This module is the case's home outside its TOML. Under the standing rule
that a case name reaches only its config and its case module, everything
about *which day and which ground* attempt #1 integrates is pinned here
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

Run it::

    python -m woof.verify.cases.les_tornado_dodgecity_20160524 --help
    python -m woof.verify.cases.les_tornado_dodgecity_20160524
    python -m woof.verify.cases.les_tornado_dodgecity_20160524 \\
        --config PATH/TO/les_tornado_100m_dodgecity_20160524.toml

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
from woof.verify.cases.les_tornado_mayfield_20211210 import _levels_below

#: This module's dotted name, correct under ``python -m`` too.
MODULE = _repo_config.module_name(__name__, globals().get("__spec__"))

#: The name of the config this case is the module half of.
CONFIG_NAME = "les_tornado_100m_dodgecity_20160524.toml"

#: Where that config lives in a source checkout.  A wheel does not ship
#: `configs/`, so this path is a DEFAULT and a display value, never an
#: assumption -- :func:`main` takes ``--config`` and refuses by name when
#: neither it nor ``WOOF_CONFIGS_ROOT`` finds the file.
CONFIG = _repo_config.default_path(CONFIG_NAME)

#: The owner ruling this case implements.
RATIFICATION = ("docs/superpowers/specs/"
                "P6-LES-DECISIONS-RATIFIED-2026-08-05.md")
#: The screens the run is graded on, registered before it.
EXPECTATIONS = "docs/les/ATTEMPT1-EXPECTATIONS.md"
#: The ingest probe that blocks the route until it is ruled on.
INGEST_FINDING = "docs/les/ATTEMPT1-INGEST-FINDING.md"

#: The event, as ruled. Times UTC.
#:
#: The cyclic supercell / tornado family south-southwest to south of
#: Dodge City on the evening of 2016-05-24: initiation around 21-22Z,
#: tornadoes roughly 23:00-01:30Z, storm motion northward.
CASE_DAY = datetime(2016, 5, 24)

#: The 100 m box centre, and the two towns that bracket it. The box is
#: 20 km square, so it spans about 37.509-37.691 N: Dodge City sits just
#: north of its top edge and Minneola just south of its bottom edge, and
#: the tornado family tracked northward through the ground between them.
BOX_CENTER_LAT, BOX_CENTER_LON = 37.6, -100.0
DODGE_CITY_LAT = 37.7528
MINNEOLA_LAT = 37.4442

#: The HRRR cycle the route is pointed at, and the window taken off it.
#: f02..f12 = 20Z through 06Z next day; the 2016 18Z cycle publishes to
#: f15 (probed 2026-08-05), so f12 is inside its horizon.
HRRR_CYCLE = datetime(2016, 5, 24, 18)
FORECAST_START_HOUR = 2
FORCING_HOURS = 10

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

#: The integration window: 20Z -> 04Z, 8 h. The plan's floor is 6 h.
START_TIME = datetime(2016, 5, 24, 20)
RUN_SECONDS = 28800.0
MIN_RUN_SECONDS = 21600.0

#: Non-zero, unlike the demonstration's, and a whole number of root steps.
RESTART_INTERVAL_S = 3600.0

#: P3 inflow seeding, pilot amplitude, fixed seed.
#:
#: d03 ONLY, ruled 2026-08-06 (G4 in :data:`RATIFICATION`).  The
#: generator takes its vertical extent from the parent-diagnosed PBLH,
#: so it needs a parent that PARAMETERIZES turbulence: d02 (YSU) -> d03
#: qualifies, d03 -> d04 does not, because d03 is itself LES and its
#: RESOLVED eddies are already d04's inflow turbulence.  Seeding d04
#: would double-count them, and the model refuses the pairing outright.
#:
#: This case was PARKED (d531fb521) before G4's sweep and carried
#: ``(3, 4)`` until 2026-09-03 -- so ``audit()`` returned ``[]`` for a
#: config the model kills twelve seconds after admission, and this gate
#: would have gone RED the moment the config was corrected.  The
#: shipped file now says false on d04 and this says (3,); the two agree
#: again, and tests/test_les_tornado_attempt1_config.py holds them to
#: the ratification rather than to each other.
INFLOW_DOMAINS = (3,)
INFLOW_SEED = 20160524
INFLOW_AMPLITUDE_SCALE = 1.0


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
    if exp.start_time.date() != CASE_DAY.date():
        bad.append(f"start_time is not on the ruled case day {CASE_DAY:%Y-%m-%d}")
    if exp.run_seconds != RUN_SECONDS:
        bad.append(f"run_seconds {exp.run_seconds} != ratified {RUN_SECONDS}")
    if exp.run_seconds < MIN_RUN_SECONDS:
        bad.append(f"run_seconds {exp.run_seconds} is under the plan's "
                   f"{MIN_RUN_SECONDS} s floor")
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
                               f"{run.inflow_perturbation_amplitude_scale} != "
                               f"{INFLOW_AMPLITUDE_SCALE}")

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
                   f"{FORECAST_START_HOUR}; the window would not start at "
                   f"{START_TIME:%HZ}")
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
        description=("Check the PARKED Dodge City 2016-05-24 LES config "
                     "against the ratified pins.  This is the drift "
                     "audit only: the case is PARKED and the ingest "
                     "route refuses its cycle, so it does not run."))
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
    print(f"case      : Dodge City, KS, {CASE_DAY:%Y-%m-%d}")
    print(f"config    : {config}")
    print(f"ruling    : {RATIFICATION}")
    print(f"screens   : {EXPECTATIONS}")
    print(f"ingest    : {INGEST_FINDING}")
    print(f"box centre: {BOX_CENTER_LAT} N, {abs(BOX_CENTER_LON)} W "
          f"(Dodge City {DODGE_CITY_LAT} N, Minneola {MINNEOLA_LAT} N)")
    print(f"window    : {START_TIME:%Y-%m-%d %HZ} + "
          f"{RUN_SECONDS / 3600:g} h, HRRR {HRRR_CYCLE:%Y-%m-%dT%H} "
          f"f{FORECAST_START_HOUR:02d}..f{FORECAST_START_HOUR + FORCING_HOURS:02d}")
    for gid, nx, ny, dx, dt, km_opt, pbl in CHAIN:
        print(f"  d{gid:02d}: {nx}x{ny} dx={dx / 1000:g} km dt={dt:g} s "
              f"km_opt={km_opt} bl_pbl_physics={pbl}")
    if bad:
        print("\nDRIFT -- the shipped config no longer matches the ruling:")
        for line in bad:
            print(f"  {line}")
        return 1
    print("\nthe shipped config matches every ratified pin")
    return 0


__all__ = ["CONFIG", "CONFIG_NAME", "CASE_DAY", "CHAIN", "BOX_CENTER_LAT",
           "BOX_CENTER_LON", "HRRR_CYCLE", "RATIFIED_NZ", "audit",
           "build_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
