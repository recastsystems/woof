"""woof command line for real-data workflows and verification cases.

``woof fetch`` acquires initialization/boundary data (GFS/HRRR
download with manifests, ERA5 cdsapi template + validation); the
behavior lives in :mod:`woof.fetch`.

``woof adapt`` compiles a WPS Vtable plus a declarative mapping descriptor,
then verifies the caller's actual GRIB2 inventory before creating a runnable,
explicitly non-stock-WRF-certified mapped adapter bundle.

``woof static|ingest|run CONFIG`` sniffs the TOML shape: an
``[experiment]``/``[[domain]]`` file routes to the Phase-5 experiment
path (schema + validation in :mod:`woof.experiment`, declared inputs in
:mod:`woof.case_data`, pipeline in :mod:`woof.runtime`), anything else
is a legacy :class:`RunConfig` dispatched to its registered real case.
``woof resume CONFIG`` is sugar over ``run --restart``: it locates the
newest valid ``gpuwmrst`` checkpoint set in ``--outdir``
(:mod:`woof.resume`) and continues through run's own dispatch, so the
restart machinery's identity refusals apply unchanged.  ``woof render
WRFOUT...`` renders product PNGs (:mod:`woof.render`); ``woof enprod
ENS_ROOT`` renders the ensemble suite -- mean, spread, exceedance
probability, paintball, probability-matched mean -- over a manifest-
declared ensemble of member run directories (:mod:`woof.da.enprod`).
``woof report [RUNDIR]`` collects one anonymous diagnostic bundle --
receipts, the failure, stage logs, this install's provenance, the card,
free space -- into a readable zip a reporter can open before they send
it (:mod:`woof.report_bundle`).
``woof update`` prints -- and only prints -- the upgrade command for
the distribution that provides THIS install and the interpreter running
it, plus the staged asset directories a wheel replacement leaves alone
(:mod:`woof.update_cli`).
``woof run-plan PLAN.json`` is the same run for a PROGRAM rather than
a person: one versioned plan document in (an envelope over the config
system, not a second config format), one append-only JSONL event stream
out -- to ``<run_dir>/events.jsonl`` and to stdout -- in which every
fact the human output prints is a typed field on a typed event, so a
GUI or a scheduler driving woof as a subprocess never parses prose.
``--resolve``/``--estimate``/``--probe`` answer the three questions a
front end asks before it starts anything (:mod:`woof.runplan`).
``woof import-namelist WPS INPUT`` translates WRF namelists into a
resolved experiment TOML and prints the substitution report.  ``woof
verify CASE`` runs either an idealized benchmark or a real case, prints
its metrics dict, and exits 0 when every gate passes or 1 naming the
failing metric.  Gate intervals are single-sourced: each case module
exports ``GATES`` and both this driver and the benchmark tests consume
that one export.

No case is named here.  The case tables below are views over
:mod:`woof.verify.cases`, which discovers case modules and classifies
them by the entry points they declare, so a case joins ``woof verify``
and ``woof static|ingest|run`` by existing -- this driver needs no edit.
``woof cases`` prints that registry (``--json`` for a front end).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from woof import capabilities, data_assets, domain_interactive
from woof.config import load_config
from woof.configuration_recovery import MemoryAdmissionError, error_document as configuration_error_document
from woof.explain import (add_explain_flag, explain_enabled, layered,
                           render, warn)
from woof.verify import cases


def _lazy_register(module):
    """Load command modules when their parser is requested."""
    def register(*args, **kwargs):
        from importlib import import_module
        return import_module(module).register_cli(*args, **kwargs)
    return register


def check_main(args):
    from woof.core.preflight import check_main as run
    return run(args)


#: ``woof check`` exit code when the configuration reads geography the
#: WPS_GEOG tree has not staged (the memory legs are still reported).
EXIT_GEOGRAPHY_MISSING = 6


def _check_geography(args) -> str | None:
    """The refusal ``woof go`` gives for geography this config reads.

    Breakage: ``woof check conus.toml --devices 2`` on the HRRR region
    recipe passed (ADMITTED on both cards) and the ``woof go`` of the same
    file then refused before its fetch: the recipe's terrain-drag fields
    read ``orogwd3_10m``, which plain ``woof fetch-geog`` does not stage
    (2.9.2 box door test).  The one function go refuses by,
    :func:`woof.go_cli.geography_refusal`, answers here too, with the same
    ``woof fetch-geog --config`` remedy, so check and go cannot disagree.
    A prepared bundle (``--prepared-root``) carries its statics and is not
    asked, as go does not ask it.
    """

    if getattr(args, "prepared_root", None) is not None:
        return None
    try:
        if not is_experiment_toml(args.config):
            return None
    except Exception:
        return None  # check_main names a config it cannot read
    from woof.geog_assets import config_geog_root
    from woof.go_cli import geography_refusal

    try:
        return geography_refusal(config_geog_root(Path(args.config)), config=Path(args.config))
    except FileNotFoundError:
        # The datasets a run reads come from its WPS namelist.  A
        # configuration that names a namelist (or forcing) not on disk has
        # no geography verdict yet: the input leg below names that file,
        # with its remedy, instead of a traceback from here.  Measured:
        # tests/test_check_display.py's host-refusal case names a namelist
        # it never writes and expects the host verdict (exit 5).
        return None


def combined_check_main(args):
    # Keep the verdict's origin for in-process configuration creators:
    # the memory check also uses exit 6 for a refused execution plan.
    args._geography_missing_only = False
    if args.config is None:
        print('woof check: CONFIG is required for preflight; bare inventory uses woof check --json', file=sys.stderr)
        return 2
    # A known source ceiling must refuse before ingest can decode files or
    # memory preflight can probe a GPU, even when those later legs cannot run.
    import tomllib
    from woof.config_authority import read_config_authority
    from woof.vertical_contract import validate_config_source_top

    raw = tomllib.loads(read_config_authority(args.config).payload.decode("utf-8"))
    top_verdict = validate_config_source_top(raw, context=str(args.config))
    if top_verdict is not None:
        print(f"woof check: {top_verdict}", file=sys.stderr)
    ingest = args.ingest_preflight_handler(args)
    if ingest:
        return ingest
    geography = _check_geography(args)
    if geography is not None:
        # Before the memory legs, the way go refuses before its fetch; on
        # stderr so `--json` stdout stays one document.
        print(f"woof check: geography BLOCKER (exit {EXIT_GEOGRAPHY_MISSING} "
              "unless a memory leg fails harder): " + geography, file=sys.stderr)
    code = check_main(args)
    if geography is not None and code == 0:
        args._geography_missing_only = True
        return EXIT_GEOGRAPHY_MISSING
    return code


def is_experiment_toml(path):
    from woof.experiment import is_experiment_toml as check
    return check(path)


def is_experiment_toml_bytes(payload):
    from woof.experiment import is_experiment_toml_bytes as check
    return check(payload)


adapt_register_cli = _lazy_register("woof.adapt")
branch_register_cli = _lazy_register("woof.branch")
bridge_assets_register_cli = _lazy_register("woof.bridge_assets")
certify_register_cli = _lazy_register("woof.certify.cli")
preflight_register_cli = _lazy_register("woof.core.preflight")
enprod_register_cli = _lazy_register("woof.da.enprod")
cycle_register_cli = _lazy_register("woof.cycle.cli")
doctor_register_cli = _lazy_register("woof.doctor")
domain_register_cli = _lazy_register("woof.domain_wizard")
tc_tracks_register_cli = _lazy_register("woof.tc_tracks")
tc_strata_register_cli = _lazy_register("woof.tc_strata")
cyclone_setup_register_cli = _lazy_register("woof.cyclone_setup")
earth2_register_cli = _lazy_register("woof.earth2_door")
earth2_diagnostics_register_cli = _lazy_register("woof.earth2_diagnostics")
downscale_register_cli = _lazy_register("woof.downscale")
fetch_register_cli = _lazy_register("woof.fetch")
fire_static_register_cli = _lazy_register("woof.static.sfire")
fire_ideal_register_cli = _lazy_register("woof.fire_ideal")
geog_register_cli = _lazy_register("woof.geog_assets")
go_register_cli = _lazy_register("woof.go_cli")
race_register_cli = _lazy_register("woof.race_cli")
hydro_register_cli = _lazy_register("woof.hydro.cli")
speedrun_register_cli = _lazy_register("woof.speedrun_cli")
ingest_register_cli = _lazy_register("woof.ingest.preflight")
mesh_register_cli = _lazy_register("woof.mpas_mesh")
ml_export_register_cli = _lazy_register("woof.ml_export")
scoreboard_register_cli = _lazy_register("woof.scoreboard")
grib2_export_register_cli = _lazy_register("woof.grib2_export")
multi_run_register_cli = _lazy_register("woof.multi_run")
obs_register_cli = _lazy_register("woof.obs.cli")
render_register_cli = _lazy_register("woof.render")
simulated_radar_register_cli = _lazy_register("woof.simulated_radar")
verification_visuals_register_cli = _lazy_register("woof.verification_visuals")
remote_register_cli = _lazy_register("woof.remote_cli")
report_register_cli = _lazy_register("woof.report_bundle")
timeline_register_cli = _lazy_register("woof.timeline.report")
run_plan_register_cli = _lazy_register("woof.runplan")
setup_register_cli = _lazy_register("woof.setup_cli")
sources_register_cli = _lazy_register("woof.sources_cli")
spectral_op_register_cli = _lazy_register("woof.spectral_ops.cli")
stage_register_cli = _lazy_register("woof.stage_cli")
stream_register_cli = _lazy_register("woof.stream")
table_assets_register_cli = _lazy_register("woof.table_assets")
update_register_cli = _lazy_register("woof.update_cli")
version_register_cli = _lazy_register("woof.version_cli")
warm_kernels_register_cli = _lazy_register("woof.warm_kernels")
kernels_register_cli = _lazy_register("woof.kernel_warm_pack")
spectral_register_cli = _lazy_register("woof.verify.spectral_cli")
cells_register_cli = _lazy_register("woof.cells.cli")

#: Discovered verification cases: name -> case module exposing
#: ``run(outdir) -> dict`` and ``GATES``.  Membership comes from the
#: registry, and a module is imported only when it is indexed, so building
#: ``--help`` no longer drags every real-data case into the process.
_CASES = cases.CaseMapping(cases.VERIFY)

#: Config-driven real cases.  Modules own their static, ingest, and run
#: pipeline entry points; the CLI only validates/dispatches the case name.
_REAL_CASES = cases.CaseMapping(cases.CONFIG)

#: Pass gates per case, single-sourced from the case modules' GATES
#: exports: metric -> (lo, hi) with strict bounds lo < value < hi;
#: None = unbounded on that side.  Resolved through the registry rather
#: than through ``_CASES`` so a stubbed runner cannot relax a real gate.
_GATES = cases.GateMapping()


#: Flags whose value is a signed coordinate or coordinate list.
#:
#: argparse treats any token starting with ``-`` as an option string
#: unless it matches its own negative-number matcher,
#: ``^-\d+$|^-\d*\.\d+$``.  ``-33.87,151.21`` has a comma in it and so
#: does not match, which made every southern- or western-hemisphere
#: value fail with ``expected one argument`` -- exactly the worldwide
#: story the product leads with, and every documented example was CONUS
#: with a positive latitude, so the failure was invisible until someone
#: pointed the wizard at Sydney.
_COORDINATE_FLAGS = frozenset({"--point", "--area"})


def _looks_like_coordinates(value: str) -> bool:
    """A leading-minus token that is entirely comma-separated numbers."""

    if not value.startswith("-"):
        return False
    fields = value.split(",")
    if not 1 <= len(fields) <= 4:
        return False
    try:
        for field in fields:
            float(field)
    except ValueError:
        return False
    return True


def _join_negative_coordinates(argv: list[str]) -> list[str]:
    """Rewrite ``--point -33.87,151.21`` as ``--point=-33.87,151.21``.

    Only a coordinate flag immediately followed by an all-numeric
    leading-minus token is joined, so nothing else about option parsing
    changes and ``--point --help`` still errors the way it should.
    """

    joined: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        following = argv[index + 1] if index + 1 < len(argv) else None
        if (token in _COORDINATE_FLAGS and following is not None
                and _looks_like_coordinates(following)):
            joined.append(f"{token}={following}")
            index += 2
            continue
        joined.append(token)
        index += 1
    return joined


def parse_fetch_arguments(arguments) -> argparse.Namespace:
    """``woof fetch``'s own parse of an argv list, read as the command line reads it.

    Every door that hands a stored argv to the fetch parser (a run plan's
    ``fetch.args``, the ``[fetch]`` table ``woof go`` downloads from, the
    download budget) comes through here, so it gets the same coordinate
    join :func:`main` gives a typed command.  Named breakage: those doors
    called the parser directly, so ``woof go`` stopped at the fetch
    stage with ``argument --area: expected one argument`` for every
    configuration south of the equator, whose ``[fetch].area`` leads with
    a minus.
    """

    return build_parser().parse_args(
        ["fetch", *_join_negative_coordinates(list(arguments))])


#: The spelling every other command-line tool answers to.  `woof
#: version` is the door and always has been; typing `woof --version`
#: got an argparse usage error at exit 2, on all three persona walks
#: (UX finding N24).  Rewritten to the subcommand here, in LEADING
#: position only, so a subcommand that ever grows a flag of this name
#: still reaches its own parser -- and so the version door's own flags
#: (`--offline`, `--pypi-timeout`) work through the alias unchanged.
_VERSION_ALIAS = "--version"


def _rewrite_version_alias(argv: list[str]) -> list[str]:
    """``woof --version [...]`` -> ``woof version [...]``."""

    if argv and argv[0] == _VERSION_ALIAS:
        return ["version", *argv[1:]]
    return list(argv)


def _failing_gate(case: str, metrics: dict) -> str | None:
    """Name of the first failing gate metric, or None if all pass.

    Gates are open intervals with strict bounds, ``lo < v < hi``; ``None``
    leaves that side unbounded.  NaN metric values compare False and
    therefore fail their gate.
    """
    if metrics["nan"]:
        return "nan"
    for name, (lo, hi) in _GATES[case].items():
        v = metrics[name]
        if not ((lo is None or lo < v) and (hi is None or v < hi)):
            return name
    return None


#: The one experiment/config positional description, shared by
#: static/ingest/run/resume (the old "[run].case" text predated the
#: experiment front door and was stale).
_CONFIG_HELP = (
    "experiment TOML ([experiment]/[[domain]] tables, as emitted by "
    "`woof domain` or `woof import-namelist`; config-driven runs "
    "declare their inputs in [case_data]).  A legacy [run]-table "
    "RunConfig naming a registered case is also accepted.")


#: Ctrl-C's exit code, by the shell convention 128 + SIGINT.
_INTERRUPT_EXIT_CODE = 130

#: Commands that can run for minutes, where "Ctrl-C does nothing" is a
#: surprise worth one line rather than a discovery.
_LONG_RUNNING_COMMANDS = frozenset({
    "go", "run", "run-plan", "resume", "branch", "fetch", "fetch-geog",
    "fetch-tables", "fetch-bridges", "setup", "verify", "downscale",
    "enprod", "render", "dual-run", "certify", "spectral",
    # The unbundled stages run for exactly as long as the welded ones
    # they were split out of: preprocessing is minutes of static build,
    # the forecast is the forecast.
    "prep", "sim", "warm-kernels", "fire-ideal",
    # `woof ensemble` is `go` by another name; argparse stores the name
    # the reader typed, so the alias needs its own row to get the notice.
    "ensemble",
})


def _warn_if_interrupt_is_ignored(command: str) -> None:
    """Say so when this process cannot see a Ctrl-C at all.

    A shell that starts a job in the BACKGROUND sets SIGINT (and
    SIGQUIT) to ``SIG_IGN`` for it, so a Ctrl-C at the terminal cannot
    stop the background job -- that is POSIX job control working as
    designed.  CPython then declines to install its own SIGINT handler
    over an inherited ``SIG_IGN``, so no ``KeyboardInterrupt`` is ever
    raised and every interrupt path in this file is unreachable.

    That is exactly what the fleet's hostile-input node measured when it
    filed "``woof go`` ignores SIGINT": the runs were launched into the
    background, so SIGINT was ignored before woof's first bytecode ran,
    while SIGTERM and SIGKILL -- which a shell does not mask -- worked.

    Re-installing a handler here would defeat the shell's intent, so the
    disposition is left alone and named instead.  One line, stderr,
    warn-not-block; nothing about the run changes.
    """

    if command not in _LONG_RUNNING_COMMANDS:
        return
    try:
        import signal
        if signal.getsignal(signal.SIGINT) is not signal.SIG_IGN:
            return
    except (ImportError, ValueError, OSError, AttributeError):
        return  # no signal support here; nothing to say
    warn("SIGINT is set to ignore in this process, so Ctrl-C cannot "
         f"stop `woof {command}`; send SIGTERM to stop it",
         why="A shell that starts a job in the background masks SIGINT "
             "and SIGQUIT for it (POSIX job control), and Python does "
             "not install its own SIGINT handler over an inherited "
             "SIG_IGN -- so no interrupt reaches this process at all. "
             "Run it in the foreground for Ctrl-C, or stop this one "
             "with `kill -TERM <pid>`, which is not masked and which "
             "every stage handles.")


def _layer(error: BaseException, args) -> str:
    """One refusal, at the layer this invocation asked for.

    The pointer names the command the reader just ran, so re-running it
    with ``--explain`` is a copy of their own last line plus one word --
    never a different command they then have to reconstruct.
    """

    return render(str(error), explain=explain_enabled(args),
                  command=f"woof {args.command}")


def _run_description() -> str:
    """`woof run --help`, told what 2.5.0 moved around it.

    This door is unchanged from 2.4.1 and its help was byte-identical
    to 2.4.1's -- so an upgrader reading it learned nothing about the
    two stages that now stand alone, the run folder each forecast
    claims, or the layout `woof render` writes (UX finding N15).  The
    door still behaves exactly as it did; the paragraph says what else
    exists now, composed from the modules that own those layouts so it
    cannot drift from them.

    It also said "in this process", which for an experiment TOML -- the
    wizard's own emission, and every multi-domain config -- has not been
    true since Task 15: the forecast runs in a supervised WORKER whose
    output is redirected to files in --outdir.  A user whose run died
    read their terminal for a traceback that was, by design, on disk,
    and reported that woof printed nothing.  The worker log names come
    from :mod:`woof.supervisor`, which writes them, for the same
    no-drift reason as the two layouts above.
    """

    from woof import render_layout, run_stamp
    from woof.supervisor import (HEARTBEAT_NAME, WORKER_STDERR_NAME,
                                  WORKER_STDOUT_NAME)

    return (
        "Integrate a config-driven real case, writing straight into "
        "--outdir.  An experiment TOML is SUPERVISED: this process "
        "watches, and the forecast itself runs in a fresh worker whose "
        "stdout and stderr go to --outdir/"
        + WORKER_STDOUT_NAME.format(attempt=1) + " and --outdir/"
        + WORKER_STDERR_NAME.format(attempt=1) + " (the number "
        "increments once per recovery attempt).  A failed run's "
        "traceback is in those files rather than in this terminal, and "
        "--outdir/" + HEARTBEAT_NAME + " names the phase it reached; "
        "--no-supervise integrates in this process instead.  The same "
        "forecast is also reachable as "
        "two stages that stand alone: `woof prep` builds the prepared "
        "tree and `woof sim` runs it (`woof go` drives both).  `woof "
        "sim` gives every forecast its own timestamped run folder -- "
        + run_stamp.describe("--outdir")
        + " -- and `woof render` writes "
        # sep="/" because this paragraph is help text, shared by every
        # reader on every platform, not a path this process is about to
        # write; the printed render line is the one that follows the
        # reader's own separator.
        + render_layout.describe("--out", sep="/")
        + ".  Both layouts have a documented flat fallback, labelled a "
        "workaround.")


def _register_run_parser(sub):
    """The run options shared by the complete and command-specific parsers."""
    run = sub.add_parser("run", help="integrate a config-driven real case",
                         description=_run_description())
    from woof.output_formats import add_arguments
    add_arguments(run)
    run.add_argument("config", type=Path, metavar="CONFIG", nargs="?",
                     help=_CONFIG_HELP)
    run.add_argument("--wrfinput", type=Path, metavar="DIR",
                     help="WRF real.exe directory containing wrfinput_d0*, "
                          "wrfbdy_d01 and producing namelist.input (instead of CONFIG)")
    run.add_argument("--hydro", default=None, metavar="on|off|SETTINGS.toml",
                     help="hydro is opt-in: on couples it wherever a hydrofabric covers the domain, "
                          "a TOML file selects settings (as on woof go); without it the run has no hydro")
    run.add_argument("--restart-from", type=Path, default=None, metavar="WRFRST",
                     help="continue a WRF run from its restart file (wrfrst_d01_<time>, any "
                          "suffix): the state, physics and clock are the file's, and the "
                          "namelist.input and wrfbdy_d01 beside it (or in --wrfinput DIR) "
                          "supply the settings and the lateral forcing.  With --run-seconds, "
                          "the window after the restart instant")
    run.add_argument("--checkpoint-at", default=None, metavar="TIMES",
                     help="WRF inputs: write only these scheduled checkpoints, as "
                          "comma-separated WRF valid times (2022-06-13_22:00:00) or "
                          "`end`; each must be a restart_interval instant of the run. "
                          "Absent writes every scheduled checkpoint")
    run.add_argument("--benchmark", action="store_true",
                     help="WRF inputs: time the forecast as a benchmark. Draws no "
                          "products and runs no post-run verification, then prints "
                          "the read, init, first step, steps and write times WRF's "
                          "rsl.out reports, and the process total "
                          "(also OUTDIR/benchmark-timing.json)")
    run.add_argument("--met-em", type=Path, metavar="DIR",
                     help="WPS metgrid directory with met_em.d0*.nc and producing namelist.input; native WOOF initialization")
    run.add_argument("--soil-source", type=Path, default=None, metavar="DIR",
                     help="WRF inputs: original met_em and Vtable directory for automatic soil-water recovery; defaults to the input directory")
    run.add_argument("--rrtmg-variant", choices=("rrtmg_legacy", "rte-rrtmgp"), default=None,
                     help="WRF inputs: preserve legacy RRTMG by default; choose rte-rrtmgp to change radiation")
    run.add_argument("--thompson-version", choices=("wrf_461", "wrf_44"), default=None,
                     help="WRF inputs: override the classic Thompson generation selected from producer TITLE")
    run.add_argument("--urban-ucm-generation", choices=("wrf_471","wrf_480"), default=None)
    run.add_argument("--urban-treeoption", type=int, choices=(0,1), default=None)
    run.add_argument("--urban-ttscheme", type=int, choices=(0,1), default=None)
    run.add_argument("--urban-fvg", type=float, default=None)
    run.add_argument("--urban-table-dir", default=None)
    run.add_argument("--vertical-grid", default=None,
                     help="met_em: native, wrf-auto, or explicit:PATH eta grid")
    run.add_argument("--vertical-levels", type=int, default=None,
                     help="met_em: requested level count for the selected vertical grid")
    run.add_argument("--run-seconds", type=float, default=None,
                     help="shorten a --wrfinput or --met-em run inside its forcing coverage")
    run.add_argument("--outdir", type=Path, default=Path("out/run"),
                     metavar="OUT", help="wrfout output directory")
    add_history_compression_argument(run)
    run.add_argument(
        "--preprocess-backend", choices=("cuda", "cpu", "auto"), default=None,
        help="CONFIG: where the root domain's preparation runs, overriding "
             "[case_data] preprocess_backend (default: that key, else auto, "
             "which prepares on the CPU when the card reads busy or cannot "
             "hold it); pin it so two runs you compare start from the same "
             "preparation. Nests prepare on the card either way")
    from woof.da.analysis_door import add_analysis_arguments
    add_analysis_arguments(run)
    run.add_argument("--restart", type=Path, default=None, metavar="RST",
                     help="resume from a gpuwmrst restart file written by "
                          "an earlier run of the SAME config (only the "
                          "forecast length / output and restart cadence "
                          "and each domain's history window, "
                          "history_begin_s / history_end_s, may differ); "
                          "restart writing itself is the "
                          "restart_interval_s config key")


def add_history_compression_argument(parser) -> None:
    """``--history-compression``: how history frames are stored.

    ``deflate`` (default) writes lossless compressed NetCDF-4 frames, the
    same values in about a third of the disk; ``none`` writes classic CDF-2
    frames for readers that need raw files.  Applied by :func:`main` as the
    process environment, so every writer the command starts (worker
    processes included) stores frames the same way.
    """

    parser.add_argument(
        "--history-compression", choices=("deflate", "none"), default=None,
        help="history frame storage: deflate (default; lossless NetCDF-4, "
             "the same values in about a third of the disk) or none "
             "(classic CDF-2). Also GPUWM_HISTORY_COMPRESSION")


def apply_history_compression(args) -> None:
    value = getattr(args, "history_compression", None)
    if value:
        os.environ["GPUWM_HISTORY_COMPRESSION"] = value


def build_parser(*, render_only: bool = False, preset_only: bool = False,
                 run_only: bool = False, go_only: bool = False, race_only: bool = False) -> argparse.ArgumentParser:
    """The whole woof command surface, assembled.

    Split out of :func:`main` so the parser can be inspected without
    running anything.  The layering convention needs that: `--explain`
    is swept onto every subcommand here, and the test that keeps the
    sweep accurate has to be able to enumerate the subcommands rather
    than transcribe a list that goes stale the next time one is added.
    """

    from woof.cli_help import AllCommandsHelp, ForecastParser
    parser = ForecastParser(
        prog="woof", usage="%(prog)s COMMAND [OPTIONS]",
        description="GPU-native WRF-ARW-like weather model.",
        epilog="`woof --version` is an alias for `woof version`.")
    parser.add_argument("--help-all", nargs=0, action=AllCommandsHelp,
                        help="show every command")
    sub = parser.add_subparsers(prog=parser.prog, dest="command", metavar="COMMAND", required=True)
    if race_only:
        race_register_cli(sub)
        add_explain_flag(sub.choices["race"])
        return parser
    if go_only:
        # Use the complete go registrar without importing unrelated products.
        go_register_cli(sub)
        from woof.earth2_licensing import add_licence_argument
        add_licence_argument(sub.choices["go"])
        add_explain_flag(sub.choices["go"])
        return parser
    if run_only:
        # A native forecast does not need to import every product's parser.
        # Keep one run registrar so this path preserves the full flag surface.
        _register_run_parser(sub)
        _lazy_register("woof.supervisor")(sub)
        add_explain_flag(sub.choices["run"])
        return parser
    _lazy_register("woof.nwpsuite.cli")(sub)
    if preset_only:
        # Metadata inspection does not need the numerical or NetCDF commands.
        # Importing their parsers made a preset list fail on an otherwise
        # sufficient CPU install before its own handler could run.
        _lazy_register("woof.model_presets")(sub)
        add_explain_flag(parser)
        add_explain_flag(sub.choices["preset"])
        return parser
    if render_only:
        # A live frame needs this parser and the common dispatch checks, not
        # imports and registration for every forecast and preparation door.
        render_register_cli(sub)
        add_explain_flag(parser)
        add_explain_flag(sub.choices["render"])
        return parser
    _lazy_register("woof.hubwind")(sub)
    preflight_register_cli(sub)
    _lazy_register("woof.ensemble.family_products")(sub)
    ingest_register_cli(sub)
    # Combined check policy (T3 handoff): the cheap CPU input/static/table
    # preflight runs first; only a zero return advances to the memory
    # estimator, so --alloc is gated on valid inputs.
    sub.choices["check"].set_defaults(func=combined_check_main)
    fetch_register_cli(sub)
    _lazy_register("woof.hf_fetch")(sub)
    fire_static_register_cli(sub)
    _lazy_register("woof.fire_ignition")(sub)
    fire_ideal_register_cli(sub)
    cds_credentials_register_cli = _lazy_register("woof.cds_credentials")
    cds_credentials_register_cli(sub)
    companion_query_register_cli = _lazy_register("woof.companion_query")
    companion_query_register_cli(sub)
    companion_domains_register_cli = _lazy_register("woof.companion_domains")
    companion_domains_register_cli(sub)
    companion_forcing_register_cli = _lazy_register("woof.companion_forcing")
    companion_forcing_register_cli(sub)
    companion_setups_register_cli = _lazy_register("woof.companion_setups")
    companion_setups_register_cli(sub)
    stream_register_cli(sub)
    geog_register_cli(sub)
    domain_register_cli(sub)
    cyclone_setup_register_cli(sub)
    _lazy_register("woof.solar")(sub)
    earth2_register_cli(sub)
    earth2_diagnostics_register_cli(sub)
    tc_tracks_register_cli(sub)
    tc_strata_register_cli(sub)
    local_da_register_cli = _lazy_register("woof.local_da")
    local_da_register_cli(sub)
    research_register_cli = _lazy_register("woof.research_workspaces")
    research_register_cli(sub)
    case_catalog_register_cli = _lazy_register("woof.case_catalog")
    case_catalog_register_cli(sub)
    render_register_cli(sub)
    # A run's history files as a machine-learning dataset: registered
    # beside render, the other door that turns history files into a
    # product.
    ml_export_register_cli(sub)
    _lazy_register("woof.earth2_export")(sub)
    _lazy_register("woof.soundings")(sub)
    scoreboard_register_cli(sub)
    grib2_export_register_cli(sub)
    simulated_radar_register_cli(sub)
    verification_visuals_register_cli(sub)
    enprod_register_cli(sub)
    downscale_register_cli(sub)
    doctor_register_cli(sub)
    timeline_register_cli(sub)
    obs_register_cli(sub)
    table_assets_register_cli(sub)
    bridge_assets_register_cli(sub)
    setup_register_cli(sub)
    remote_register_cli(sub)
    # The three unbundled stages.  `render` is registered above and has
    # always stood alone; `prep` and `sim` are what `go` used to be the
    # only way to reach.  Registered BEFORE `go` so the help listing
    # reads in pipeline order.
    stage_register_cli(sub)
    _lazy_register("woof.compose_source")(sub)
    _lazy_register("woof.source_intermediate")(sub)
    _lazy_register("woof.storm_diagnostics")(sub)
    go_register_cli(sub)
    _lazy_register("woof.standby_cli")(sub)
    race_register_cli(sub)
    hydro_register_cli(sub)
    _lazy_register("woof.ocean.door")(sub)
    # Registered beside `go` because it IS `go`, on a fixed course, with a
    # clock and a sealed capsule around it.  The internal bench surface:
    # release-facing pages carry capability statements, this one carries
    # seconds.
    speedrun_register_cli(sub)
    adapt_register_cli(sub)
    # The MPAS mesh generator's front door.  `rw_mpas_mesh` reproduced the
    # published NCAR meshes to 1e-11 and had no command anyone could type,
    # which by this project's rule means it was not shipped.
    mesh_register_cli(sub)
    model_presets_register_cli = _lazy_register("woof.model_presets")
    model_presets_register_cli(sub)
    certify_register_cli(sub)
    spectral_op_register_cli(sub)
    multi_run_register_cli(sub)
    report_register_cli(sub)
    run_plan_register_cli(sub)
    # Every physics scheme and suite with its plain meaning, and the check
    # a combination passes before a run: the one answer the browser door
    # and an assistant read.
    physics_catalog_register_cli = _lazy_register("woof.physics_catalog")
    physics_catalog_register_cli(sub)
    # The 0 ULP referee for that catalog: recorded stock WRF runs of every
    # combination, replayed through WOOF and scored word for word.
    verify_exact_register_cli = _lazy_register("woof.verify_exact.cli")
    verify_exact_register_cli(sub)
    # The hex (MPAS-style mesh) model door.  Listed here so the
    # help shows them; the words are dispatched in _dispatch_argv before
    # this parser runs, so each component keeps its own full parser.
    from woof.components import register_component_doors
    register_component_doors(sub)
    # The human view of the same registry `run-plan --sources` serves as
    # JSON.  Registered beside it so the two doors read as one pair in
    # the help listing, and because every no-route refusal in
    # `woof.fetch_routes` now points at this one by name.
    sources_register_cli(sub)
    cycle_register_cli(sub)
    update_register_cli(sub)
    version_register_cli(sub)
    # The first forecast's GPU kernel compile, paid ahead of time.
    warm_kernels_register_cli(sub)
    kernels_register_cli(sub)
    spectral_register_cli(sub)
    cells_register_cli(sub)
    lst = sub.add_parser(
        "cases", help="list the discovered verification cases and the "
                      "entry points each one declares")
    lst.add_argument("--json", action="store_true",
                     help="emit the registry as JSON for a front end")
    ver = sub.add_parser(
        "verify", help="run a benchmark or code-verification case and check its gates")
    ver.add_argument("case", choices=sorted(_CASES),
                     help="benchmark or code-verification case to run (no observation scoring)")
    ver.add_argument("--outdir", type=Path, default=None, metavar="OUT",
                     help="directory for the PNG and wrfout NetCDF output "
                          "(omit to compute metrics only)")
    static = sub.add_parser(
        "static", help="build static fields for a config-driven real case")
    static.add_argument("config", type=Path, metavar="CONFIG",
                        help=_CONFIG_HELP)
    static.add_argument("--output", type=Path,
                        default=Path("out/static_fields.npz"), metavar="NPZ",
                        help="static-field NPZ output")
    ingest = sub.add_parser(
        "ingest", help="build initialized fields for a config-driven case")
    ingest.add_argument("config", type=Path, metavar="CONFIG",
                        help=_CONFIG_HELP)
    ingest.add_argument("--output", type=Path,
                        default=Path("out/initial_state.npz"), metavar="NPZ",
                        help="initialized-state NPZ output")
    _register_run_parser(sub)
    resume = sub.add_parser(
        "resume",
        help="continue a run from its newest valid gpuwmrst checkpoint "
             "(sugar over run --restart; same config, same outdir)")
    from woof.output_formats import add_arguments as add_output_arguments
    add_output_arguments(resume)
    resume.add_argument("config", type=Path, metavar="CONFIG",
                        help="the SAME config the interrupted run used; "
                             "the restart identity check refuses any "
                             "other. An argument that is not a readable "
                             "file is tried with the .toml extension a "
                             "file manager hides, then against --outdir, "
                             "and last against the configuration the run "
                             "in --outdir recorded for itself "
                             "(child.toml, experiment.toml or "
                             "captured-config-<run id>.toml)")
    resume.add_argument("--from", dest="from_checkpoint", default="latest",
                        metavar="CKPT|latest",
                        help="explicit gpuwmrst_*.npz checkpoint, or "
                             "'latest' (default) to take the newest set "
                             "in --outdir whose members validate")
    resume.add_argument("--outdir", type=Path, default=Path("out/run"),
                        metavar="OUT",
                        help="the interrupted run's wrfout/checkpoint "
                             "directory (default out/run)")
    # Compose with the existing memory-preflight registrar above; Task 15
    # owns only the run parser's supervision flags and dispatch helper.
    supervisor_register_cli = _lazy_register("woof.supervisor")
    supervisor_register_cli(sub)
    # resume continues a supervised run, so it carries run's exact
    # supervision surface; the checkpoint is resolved below and dispatch
    # is then run's own path with args.restart set.
    supervisor_register_cli(sub, "resume")
    # `branch` is resume's other half: a NEW run seeded from an existing
    # run's checkpoint (woof.branch).  It integrates exactly as `run`
    # does once its config is written, so it carries run's supervision
    # surface for the same reason resume does.
    branch_register_cli(sub)
    supervisor_register_cli(sub, "branch")
    imp = sub.add_parser(
        "import-namelist",
        help="translate WRF namelist.wps + namelist.input into a "
             "resolved woof experiment TOML plus a substitution report")
    imp.add_argument("wps", type=Path, metavar="WPS",
                     help="WPS namelist.wps (projection + nest layout)")
    imp.add_argument("input", type=Path, metavar="INPUT",
                     help="WRF namelist.input (domains/physics/dynamics/"
                          "bdy_control)")
    imp.add_argument("--output", type=Path, default=None, metavar="TOML",
                     help="write the resolved experiment TOML here "
                          "(omit to print the report only)")
    imp.add_argument("--geogrid-tbl", type=Path, default=None, metavar="PATH",
                     help="GEOGRID.TBL (file or directory) whose HGT_M "
                          "smooth_option/smooth_passes set every domain's "
                          "terrain smoothing (default: the namelist.wps "
                          "opt_geogrid_tbl_path, else ./geogrid/ beside it)")
    imp.add_argument("--terrain-smoothing-precision", default=None,
                     choices=("float64", "wps-float32"),
                     help="arithmetic of every domain whose terrain "
                          "smoother is WPS's default smth-desmth_special x1: "
                          "wps-float32 reproduces geogrid.exe's HGT_M "
                          "exactly, float64 is WOOF's own smoother "
                          "(default: the GEOGRID.TBL HGT_M smooth_precision, "
                          "else float64); every other smoother always runs "
                          "WPS's float32")
    imp.add_argument("--name", default=None, metavar="NAME",
                     help="[experiment].name for the resolved TOML "
                          "(default derived from start time and domain "
                          "count)")
    imp.add_argument("--rrtmg-variant", default=None,
                     choices=("rte-rrtmgp", "rrtmg_legacy"),
                     dest="rrtmg_variant",
                     help="implementation for a WRF RRTMG 4/4 request: "
                          "RTE+RRTMGP by default, or legacy RRTMG when "
                          "the namelist selects GSD MYNN or aer_opt 1, 2 or 3; "
                          "an explicit choice retains its implementation")
    # No argparse default: the report records WHAT chose the line, and
    # "nobody did, the importer's default applied" is one of the answers.
    from woof.namelist_import import wrf_version_argument
    imp.add_argument("--wrf-version", default=None, type=wrf_version_argument,
                     dest="wrf_version",
                     help="the WRF line or release the namelist was written "
                          "for (3, 4, or a release such as 4.6.1).  The line "
                          "selects the Registry default an omitted "
                          "&dynamics/use_theta_m takes (3: 0, dry theta, the "
                          "line operational HRRR v4 runs; 4, the "
                          "default: 1, moist theta, booked as a "
                          "substitution).  A V4 release also pins the scheme "
                          "generations that release runs (4.6.1: WRF 4.6's "
                          "YSU, sun, Noah-MP snow aging and urban water "
                          "balance; a bare 4 keeps the newest). The report "
                          "names the version and what chose it")
    imp.add_argument("--urban-ucm-generation", choices=("wrf_471", "wrf_480"), default=None,
                     help="single-layer urban numerical generation; omitted imports retain wrf_471")
    imp.add_argument("--urban-treeoption", type=int, choices=(0,1), default=None)
    imp.add_argument("--urban-ttscheme", type=int, choices=(0,1), default=None)
    imp.add_argument("--urban-fvg", type=float, default=None)
    imp.add_argument("--urban-table-dir", default=None)
    imp.add_argument("--ack", action="append", default=[], metavar="ID",
                     help="declared-experiment acknowledgement id to "
                          "write into [experiment].acknowledgements of "
                          "the resolved TOML (repeatable).  WRF "
                          "namelists cannot spell woof governance "
                          "declarations, so an import that needs one -- "
                          "e.g. shortwave-on/longwave-off physics "
                          "across a window that includes local night -- "
                          "names the id it wants in its refusal")
    imp.add_argument("--static-cache-root", type=Path, default=None,
                     metavar="DIR", dest="static_cache_root",
                     help="cache_root of the [static.highres] block a "
                          "land-cover geog_data_res token (cglc_modis_lcz) "
                          "imports as (default: the per-user "
                          "high-resolution cache the engine's default "
                          "terrain already uses); unused when the "
                          "namelist names no such token")
    # ONE layering convention, registered in ONE place, after every
    # registrar has run.  Every subcommand takes --explain, so the
    # pointer the refusal boundary appends -- the reader's own
    # invocation with "--explain" added -- is true whichever command
    # the reader was running.  A per-command opt-in would have made
    # that pointer a lie exactly on the commands nobody remembered to
    # opt in.
    # `add_explain_flag` is idempotent because two registrars share a
    # parser (check, run, resume).
    for subcommand_parser in sub.choices.values():
        add_explain_flag(subcommand_parser)
    from woof.earth2_licensing import add_licence_argument
    for name in ("fetch", "prep", "domain", "go", "ensemble", "source-intermediate", "earth2", "earth2-diagnose",
                 "hf-fetch"):
        if name in sub.choices:
            add_licence_argument(sub.choices[name])
    return parser


def main(argv: list[str] | None = None) -> int:
    """The `woof` front door: parse one command line, dispatch, exit.

    The body is :func:`_dispatch_argv`; this wrapper exists to bound the
    invocation.  ``--explain`` is recorded in module state (library code
    that warns has no ``args`` in reach), and this function is called
    in-process by more than the console script -- an embedder scripting
    the CLI, and the test suite, both call it repeatedly in one
    interpreter.  Stamping that state without giving it back means the
    NEXT invocation inherits it and its warnings grow an explanation
    layer nobody asked for.  ``explain_scope`` makes the flag's lifetime
    this call's, on every exit path including a raised exception.
    """

    if argv is None:
        # A command line on a free-threaded build keeps the interpreter
        # lock off for good (the ranks would otherwise take turns again
        # after netCDF4's import); see woof.free_threading.
        from woof.free_threading import keep_gil_disabled
        keep_gil_disabled()
    from woof.explain import explain_scope
    from woof.progress import line_buffer_stdout

    # THE flush discipline, set once for every subcommand.  CPython
    # block-buffers a redirected stdout, so the GFS route's per-file
    # lines -- printed as each file landed -- arrived in a reader's log
    # together, at exit: 9.1 s of a 9.8 s fetch looked like silence (UX
    # finding N10).  Nothing else in the product has to remember to
    # flush; a `print` streams because the door said so.
    line_buffer_stdout()
    from woof.earth2_licensing import check_argv, licence_scope, _values
    tokens = list(sys.argv[1:] if argv is None else argv)
    try:
        check_argv(tokens)
    except ValueError as error:
        print(f"woof: {error}", file=sys.stderr)
        return 2
    with explain_scope(False), licence_scope(_values(tokens, "--accept-licence")):
        return _dispatch_argv(argv)


def _dispatch_argv(argv: list[str] | None = None) -> int:
    from woof.table_assets import MissingTableAssets, TableAssetError
    tokens = _rewrite_version_alias(
        list(sys.argv[1:] if argv is None else argv))
    from woof.components import dispatch_component
    handled = dispatch_component(tokens)
    if handled is not None:
        return handled
    if tokens and tokens[0] == "suite":
        from woof.nwpsuite.cli import main as suite_main
        return suite_main(tokens[1:])
    if tokens == ["check", "--json"]:
        # The configured check door is unchanged. Bare check is inventory only.
        from woof.nwpsuite.hardware import inventory
        from json import dumps as box_dumps
        print(box_dumps(inventory(), sort_keys=True))
        return 0
    if (len(tokens) > 1 and tokens[0] == "timeline"
            and tokens[1] not in {"report", "project", "--help", "-h"}):
        tokens.insert(1, "report")
    parser = (build_parser(render_only=True)
              if tokens and tokens[0] == "render" else
              build_parser(preset_only=True)
              if tokens and tokens[0] == "preset" else
              build_parser(run_only=True)
              if tokens and tokens[0] == "run" else
              build_parser(go_only=True)
              if tokens and tokens[0] == "go" else
              build_parser(race_only=True)
              if tokens and tokens[0] == "race" else build_parser())
    if not tokens:
        parser.print_help()
        return 0

    # Bare `woof domain` at a terminal asks its four questions instead
    # of printing a usage dump.  The session hands back the argv a
    # reader could have typed, which is then parsed and dispatched by
    # exactly the code below -- there is no second wizard.
    #
    # Off a terminal this is not reached, so a script or CI job still
    # gets the usage error and a nonzero exit.  A prompt nobody can see
    # is a hang, which is a worse answer than an error.
    interactive = False
    if domain_interactive.is_interactive(tokens):
        try:
            tokens = domain_interactive.collect()
        except domain_interactive.PromptAborted as stop:
            print(f"woof domain: stopped ({stop}); nothing was written.",
                  file=sys.stderr)
            return 2
        interactive = True
        print("")
        print(domain_interactive.printable_command(tokens))
        print("")

    args = parser.parse_args(_join_negative_coordinates(tokens))
    apply_history_compression(args)
    if hasattr(args, "output_format"):
        from woof.output_formats import options as output_options, require_binary
        try:
            for row in output_options(args)[0]:
                if row["writer"]:
                    require_binary(row)
        except ValueError as error:
            parser.error(str(error))
    # Provenance: the emitted TOML records which front door authored it.
    args.interactive = interactive
    if args.command == "run":
        if (args.config is not None and args.wrfinput is None
                and args.met_em is None and Path(args.config).is_dir()
                and (Path(args.config) / "namelist.input").is_file()):
            # A WRF run directory named where a CONFIG goes (NCAR's
            # benchmark folders are run this way): the WRF-input door, which
            # restarts when its namelist says restart = .true.
            args.wrfinput, args.config = Path(args.config), None
        if (args.restart_from is not None and args.wrfinput is None
                and args.config is None and args.met_em is None):
            args.wrfinput = Path(args.restart_from).parent
        if args.restart_from is not None and args.wrfinput is None:
            parser.error("--restart-from continues a WRF run; it pairs with --wrfinput DIR "
                         "(or stands alone, reading the directory the file is in)")
        if sum(value is not None for value in (args.config, args.wrfinput, args.met_em)) != 1:
            parser.error("run requires exactly one of CONFIG, --wrfinput DIR or --met-em DIR")
        if args.soil_source is not None and args.wrfinput is None:
            parser.error("--soil-source supplies original soil layers for --wrfinput DIR")
        if args.run_seconds is not None and args.config is not None:
            parser.error("--run-seconds is for --wrfinput or --met-em; set run_seconds in CONFIG")
        if args.rrtmg_variant is not None and args.config is not None:
            parser.error("--rrtmg-variant is for WRF inputs; set radiation in CONFIG")
        if args.thompson_version is not None and (args.config is not None or args.met_em is not None):
            parser.error("--thompson-version is for WRF folders; set thompson_version in CONFIG")
        if getattr(args, "checkpoint_at", None) is not None and args.wrfinput is None:
            # Breakage it prevents: only the WRF-input door reads the
            # selection, so on another route every scheduled checkpoint
            # would still be written while the caller believed it was not.
            parser.error("--checkpoint-at selects the WRF-input door's checkpoints "
                         "(--wrfinput DIR or --restart-from WRFRST); the other routes "
                         "do not read it")
        if args.benchmark and args.wrfinput is None:
            # Breakage it prevents: the flag would be read by nothing on the
            # other routes, so a timed run would still render and verify
            # while its benchmarker believed it did not.
            parser.error("--benchmark times the WRF-input door (a WRF run folder, "
                         "--wrfinput DIR or --restart-from WRFRST); the other "
                         "routes do not read it")
        if args.config is not None and any(getattr(args,key) is not None for key in ("urban_ucm_generation","urban_treeoption","urban_ttscheme","urban_fvg","urban_table_dir")):
            parser.error("urban CLI controls are for --wrfinput; set the shared urban fields in CONFIG")
        if args.vertical_levels is not None and args.met_em is None:
            parser.error("--vertical-levels is for --met-em inputs")
        if args.vertical_grid is not None and args.met_em is None:
            parser.error("--vertical-grid is for --met-em inputs")
        if args.wrfinput is None:
            # Breakage it prevents: the CONFIG and met_em routes have no
            # analysis hook, so the bundle would be read by nothing and an
            # unanalysed forecast would run under an analysed name.
            from woof.da.analysis_door import AnalysisDoorRefusal, refuse_unrouted
            try:
                refuse_unrouted(args, "CONFIG" if args.config is not None else "--met-em")
            except AnalysisDoorRefusal as error:
                parser.error(str(error))
        if args.preprocess_backend is not None and args.config is None:
            parser.error("--preprocess-backend pins CONFIG's preparation; the "
                         "--wrfinput and --met-em routes do not read it, so it "
                         "would be dropped")
    # Library code emits one-line warnings through woof.explain.warn;
    # stamping the flag once here is what lets --explain add their
    # mechanism prose without threading args through every call chain.
    from woof.explain import set_explain, set_invocation
    set_explain(explain_enabled(args))
    # The tokens AS TYPED (or as the interactive session composed them),
    # recorded inside the scope `main` opened, so every refusal tail can
    # print the reader's own line with --explain appended instead of a
    # bare command name that is itself a usage error (UX finding N8).
    set_invocation(tokens)
    # The render stage hands a long forecast's frames over in a file it
    # removes when the stage ends.  Read before the provenance gate and
    # the capability preflight below, so their refusal lines name the
    # frames, not a file that will be gone.
    if getattr(args, "inputs_from", None) is not None:
        from woof.render import _fold_render_inputs

        refusal = _fold_render_inputs(args)
        if refusal is not None:
            print(f"render: {refusal}", file=sys.stderr)
            return 2

    _warn_if_interrupt_is_ignored(args.command)

    # The leaf module, deliberately: highres_production re-exports this
    # class but pulls numpy and the static builders with it, and `woof
    # version` must not pay for that.  The name is needed at except-clause
    # time below, so it cannot be deferred into the handler itself.
    from woof.static.highres_refusal import HighresRefusal
    from woof.ingest.memory_refusal import InitializationMemoryRefused
    from woof.forecast_gpu import ForecastGPURequired

    try:
        # WHICH TREE IS EXECUTING, said out loud before anything runs,
        # and a refusal when the install cannot answer consistently.
        # Inside the try because the refusal is a ValueError and the
        # boundary below is where every refusal in this product prints:
        # one sentence, exit 2, `--explain` for the mechanism.
        #
        # The diagnostic subcommands (`version`, `doctor`, `report`,
        # `update`) still print the banner but are never refused --
        # they are what a reader runs BECAUSE the install is confusing,
        # and a gate that blocks its own diagnostic leaves no way out.
        from woof.provenance_gate import announce

        from woof.ensemble.calibration_admission import refuse_public_arguments
        refuse_public_arguments(args)
        announce(f"woof {args.command}")
        # THE front-door capability preflight, for every subcommand, in
        # one place.  After argparse (so `--help` is never refused) and
        # before dispatch (so nothing has fetched, spawned or allocated
        # yet).  A subcommand cannot forget to call it and cannot grow a
        # private copy that drifts -- which is precisely how `woof
        # verify` came to print a clean CuPy refusal while `woof run`,
        # on the same install, relayed a raw traceback out of a worker
        # it had already spawned.
        #
        # `go --dry-run` is exempt and says so on its own leg: a dry run
        # spends nothing and prints the commands a reader would type, so
        # refusing it would block the one gesture that costs nothing.
        # `sim --print-command` is the same gesture on the unbundled
        # forecast stage -- it answers "what would you run?", which is a
        # question a third party integrating against the boundary asks
        # on a machine that has no card at all.
        # `go --readiness` is the same gesture again: it answers "can this
        # window start?" from HEADs (or, with --no-probe, from the table)
        # and runs nothing, and a site asks it from a scheduler that may
        # have no card (DESIGN A136 3.4, 3.7).
        _spends_nothing = (
            # `ensemble` is `go` by another name, so its dry run (which
            # prints a recipe's member plan) spends nothing either.
            (args.command in ("go", "ensemble")
             and (getattr(args, "dry_run", False)
                  or getattr(args, "readiness", False)))
            or (args.command == "sim"
                and getattr(args, "print_command", False)))
        # `go --prepare-only` fetches and prepares on the CPU backends and
        # stops before the forecast, so the GPU runtime it would refuse
        # for is the forecast's (go_cli._prepare_only_needs_no_card).
        # `ensemble` is go by another name and prepares the same way: its
        # --prepare-only prepares every member on the CPU and stops before
        # the forecast. Breakage it prevents: a DA warm-start box preparing
        # an HRRR time-lagged roster without CuPy was refused at the door.
        _prepares_only = (args.command in ("go", "ensemble")
                          and getattr(args, "prepare_only", False))
        if not _spends_nothing and not _prepares_only:
            capabilities.require_for_command(args.command)
        # --no-memory-gate reaches every process this command starts: the
        # forecast stage `go` runs, the worker `run` supervises, the runner
        # `sim` and `downscale` host.  Their envelope admissions read it
        # there; set here once, it cannot be forgotten by one door.
        from woof.core.resident_admission import memory_gate_override

        with memory_gate_override(getattr(args, "no_memory_gate", False)):
            if args.command == "run" and getattr(args, "hydro", None) is not None:
                from woof.hydro.runtime import run_scope
                with run_scope(args.hydro, getattr(args, "config", None)):
                    return _dispatch(args)
            return _dispatch(args)
    except KeyboardInterrupt:
        # Ctrl-C.  Every subcommand gets the same bounded contract: one
        # sentence, exit 130 (the shell's 128 + SIGINT), no child and no
        # unrelated process signalled by us.  Without this the natural
        # gesture ends in a KeyboardInterrupt traceback, which reads as
        # a crash rather than as the thing the user just asked for.
        #
        # `go` catches its own interrupt one layer down so it can name
        # the stage and the pid it was waiting on; this is the floor
        # under every other command.
        print(f"\nwoof {args.command}: interrupted (Ctrl-C); stopping "
              "here. Any partial output carries no completion receipt.",
              file=sys.stderr)
        return _INTERRUPT_EXIT_CODE
    except ForecastGPURequired as error:
        print(f"woof {args.command}: {error}", file=sys.stderr)
        return 2
    except capabilities.CapabilityMissing as error:
        # The front-door preflight, and every deeper door that raises the
        # same refusal.  The message already names the command, because
        # the `python -m` doors -- which have no prefixing boundary --
        # print the identical string.
        print(render(str(error), explain=explain_enabled(args),
                     command=error.command), file=sys.stderr)
        return 2
    except ModuleNotFoundError as error:
        # A missing dependency is a documented install gap with a
        # remedy, not a traceback -- and the remedy is DERIVED from the
        # module that was missing.  The version this replaces hard-coded
        # CuPy: every other missing module (wrf-rust, scipy, pyshp)
        # skipped the branch and left as a raw traceback.
        requirement = capabilities.requirement_for_module(error.name)
        if requirement is not None:
            print(render(capabilities.refusal(f"woof {args.command}",
                                              requirement),
                         explain=explain_enabled(args),
                         command=f"woof {args.command}"), file=sys.stderr)
            return 2
        raise
    except data_assets.CompanionDataMissing as error:
        # The companion's THIRD failure state: importable, version-matched,
        # and missing a member.  The branch above cannot see it (nothing is
        # unimportable) and a generic handler would relay the NetCDF open's
        # fifteen frames -- measured at this door when the first public-tree
        # companion wheel shipped without its rrtmgp/*.nc.  The message
        # already names the member, the breakage, and the pip line.
        print(f"woof {args.command}: {error}", file=sys.stderr)
        return 2
    except (MissingTableAssets, TableAssetError) as error:
        print(f"woof {args.command}: {error}", file=sys.stderr)
        return 2
    except MemoryAdmissionError as error:
        import json
        print(f"woof {args.command}: " + _layer(error, args), file=sys.stderr)
        print(json.dumps(configuration_error_document(error), ensure_ascii=True, allow_nan=False))
        return 2
    except ValueError as error:
        # Documented refusals (the wizard's latitude/antimeridian
        # windows, fetch's source contracts, downscale's cadence and
        # evidence contracts, the importer's namelist contract, the
        # config loaders' schema errors, ...) are raised as ValueError
        # (incl. DomainFitError).  They are user-facing messages, not
        # programming errors: print the message, exit 2 (argparse's
        # usage-error convention), no traceback -- uniformly, for every
        # subcommand.
        #
        # THE refusal print boundary.  A message composed with
        # woof.explain.layered carries both halves; here is where one
        # of them is chosen.  An unlayered message passes through
        # unchanged, so this is a no-op for the refusals that are
        # already one sentence -- which is most of them.
        print(f"woof {args.command}: "
              + _layer(error, args), file=sys.stderr)
        return 2
    except HighresRefusal as error:
        # [static.highres] refusals are user-facing documented outcomes
        # for EVERY command that builds statics -- `static`, `go`, `run`,
        # `setup` -- not just the fetch family below.  They subclass
        # RuntimeError, so this clause must precede that one or the
        # generic handler would re-raise them as tracebacks.
        #
        # That is exactly what 2.3.2 did: `static` is not in the
        # fetch-family list, so a missing geography stack escaped as a
        # raw traceback at exit 1 after a 160.7 MiB download.  Every
        # refusal in this product is a sentence at exit 2; this one is
        # now no exception.
        print(f"woof {args.command}: " + str(error), file=sys.stderr)
        return 2
    except InitializationMemoryRefused as error:
        print(f"woof {args.command}: " + _layer(error, args), file=sys.stderr)
        if explain_enabled(args):
            import traceback
            traceback.print_exception(error, file=sys.stderr)
        return 2
    except RuntimeError as error:
        # StreamingRefused subclasses RuntimeError, so it lands in this
        # clause -- and used to fall through to the bare ``raise`` below:
        # a [tiles] configuration the config loader refuses (mode = 'on'
        # over a nested tree, an unrouted chain) escaped as a ~20-line
        # traceback out of ``build_experiment`` with the remedy paragraph
        # buried at the bottom, on every door except run-plan (which
        # already translates it to PlanError).  It is a documented
        # refusal with a stated remedy, so it gets the refusal boundary:
        # one message, exit 2, no traceback.  Imported HERE, inside the
        # handler, because woof.core.streaming must stay a thing a
        # resident run never pays for -- this import only runs once a
        # RuntimeError has already been raised.
        from woof.core.streaming import StreamingRefused

        from tilestream.hoststore import BudgetExceeded, HostMemoryExhausted

        if isinstance(error, (BudgetExceeded, HostMemoryExhausted)):
            print(f"woof {args.command}: host-store memory refused: {error}. "
                  "Reduce the prepared grid or use enough available host memory; "
                  "raise a configured host budget only if the machine can fit it.",
                  file=sys.stderr)
            if explain_enabled(args):
                import traceback
                traceback.print_exception(error, file=sys.stderr)
            return 2

        if isinstance(error, StreamingRefused):
            print(f"woof {args.command}: "
                  + _layer(error, args), file=sys.stderr)
            return 2
        # `woof cycle`'s refusals are a RuntimeError by design -- each
        # one is required to carry what it observed -- and every one of
        # them left through the bare `raise` below as a traceback at
        # exit 1: a bad --cycles, a child step that does not divide the
        # parent's, a placement with no parent geography.  They are
        # refusals, so they get the refusal boundary.  Imported here, as
        # StreamingRefused is, only once a RuntimeError has been raised.
        from woof.cycle.contracts import CycleRefusal

        if isinstance(error, CycleRefusal):
            print(f"woof {args.command}: "
                  + _layer(error, args), file=sys.stderr)
            return 2
        # The simulated radar door's refusals (a missing or stale
        # rw_simradar, a native refusal of an option or input) name the
        # breakage and the next step; a traceback buried that sentence.
        from woof.rustwx import SimulatedRadarRefusal

        if isinstance(error, SimulatedRadarRefusal):
            print(f"woof {args.command}: "
                  + _layer(error, args), file=sys.stderr)
            return 2
        if args.command in ("fetch", "stream", "fetch-geog", "fetch-tables",
                            "fetch-bridges", "obs", "report"):
            # fetch-family RuntimeErrors are operational outcomes
            # with a stated remedy (no complete latest cycle on the
            # mirrors; a --wait-for window that timed out with its
            # resume story; a download that must be re-run to resume;
            # another writer holding the staging lock), not
            # programming errors.  `report` is here for the same
            # reason and only that reason: its single RuntimeError is
            # ReportWriteError, an operational outcome ("every volume
            # refused the write") whose message is already layered
            # with the remedy.  Without this it printed both layers
            # glued by the explain sentinel inside a traceback, and
            # exited 1 where its own documentation promised a
            # refusal.  `obs` is here for a third instance of the
            # same shape: every RuntimeError it can raise comes from
            # `woof.obs.frontdoor.FrontDoor`, which is either "the
            # decoder is not built, here is the remedy" or a refusal
            # the decoder itself printed and this layer relayed --
            # a directory holding two nominal times, a pack whose
            # schema is not the one this reader knows.  All are
            # operational outcomes with a next action in the message,
            # and a traceback buries the sentence that carries it.
            # Scoped to these commands: elsewhere a
            # RuntimeError (including CUDA runtime failures) must
            # keep its traceback.
            print(f"woof {args.command}: "
                  + _layer(error, args), file=sys.stderr)
            return 2
        # A supervised run relays its worker's failure as a
        # SupervisorError, and a worker that could not import a
        # dependency is an install gap wearing a RuntimeError's clothes.
        # Named here with the remedy DERIVED from the module the worker
        # said was missing; anything this registry does not recognise
        # keeps its traceback, because a run that crashed for a real
        # reason must not be dressed up as a usage error.
        remedy = capabilities.remedy_for_error(error)
        if remedy is not None:
            print(f"woof {args.command}: the run could not start because "
                  f"a dependency is missing.\n  {error}\n" + remedy,
                  file=sys.stderr)
            return 2
        from woof.supervisor import SupervisorError
        if isinstance(error, SupervisorError):
            print(f"woof {args.command}: " + _layer(error, args), file=sys.stderr)
            if explain_enabled(args):
                import traceback
                traceback.print_exception(error, file=sys.stderr)
            return 1
        raise
    except (FileNotFoundError, NotADirectoryError, IsADirectoryError,
            PermissionError) as error:
        # A path the reader typed that is not there (or is a folder where
        # a file belongs) is their input to correct, whichever handler
        # happened to open it: one sentence naming the option, exit 2.
        # A path the program chose for itself keeps its traceback, because
        # a file this program should have written and did not is a defect.
        from woof.cli_paths import supplied_path_refusal

        refusal = supplied_path_refusal(error, args, parser=parser, tokens=tokens)
        if refusal is None:
            raise
        print(f"woof {args.command}: {refusal}", file=sys.stderr)
        if explain_enabled(args):
            import traceback
            traceback.print_exception(error, file=sys.stderr)
        return 2


def case_door(record: dict) -> str:
    """The command that runs this case.

    `woof cases` advertises every registered case; `woof verify`
    accepts only those carrying the `verify` capability, which is ten of
    twenty-one.  The other eleven are reachable, but only through
    `python -m woof.verify.cases.<name>`, a form that appeared once in
    all of the user-facing documentation.  Printing the door beside each
    case is what makes the difference visible in the listing itself
    rather than in a document a reader has to already know about.
    """

    name = record["name"]
    if "verify" in record.get("capabilities", ()):
        return f"woof verify {name}"
    return f"python -m woof.verify.cases.{name}"


def _plain_member_request(args):
    """The N > 1 request with no recipe this command line and its config make, or None.

    Read from an experiment config only: a ``[grid]``/``[dynamics]``/``[run]``
    config is refused by name further down, by the loader that owns it.
    ``[ensemble] sources`` is refused here in its own words: `run`, `resume`
    and `branch` bind no member source, and the worker would only find
    that out after it had been given a card.
    """
    from woof.ensemble import member_inputs

    config = getattr(args, "config", None)
    if config is not None and Path(config).is_file():
        from woof.config_authority import read_config_authority
        if not is_experiment_toml_bytes(read_config_authority(config).payload):
            return None
    request = member_inputs.config_request(
        config, members=getattr(args, "members", None),
        keep_member_files=getattr(args, "keep_member_files", None))
    if request is not None and request.recipe is None and request.sources:
        from woof.ensemble_admission import unbound_sources_refusal
        raise ValueError(unbound_sources_refusal())
    return request if member_inputs.needs_member_sources(request) else None


def _refuse_members_of_one_checkpoint(args) -> None:
    """``resume`` and ``branch`` continue ONE forecast from ONE checkpoint.

    Breakage it prevents: N > 1 members would each restore that checkpoint
    and run the same forecast N times, published with zero spread.  Refused
    here, before ``branch`` writes its run folder and before a worker or a
    card is taken.

    A config that selects a recipe is refused before this is asked, by
    :func:`woof.ensemble.recipe_door.refuse_continuation`, and neither
    command registers ``--recipe`` or ``--trajectories``.
    """
    request = _plain_member_request(args)
    if request is not None:
        from woof.ensemble import member_inputs
        raise ValueError(member_inputs.one_input_refusal(
            request.members,
            f"woof {args.command} continues one forecast from one checkpoint, "
            "so every member would restore it."))


def _dispatch(args) -> int:
    from woof.output_formats import command_scope
    with command_scope(args) as scope:
        result = _dispatch_with_outputs(args)
        return scope.finish(result) if scope else result


def _dispatch_with_outputs(args) -> int:
    if args.command in ("check", "fetch", "stream", "fetch-geog", "domain", "render",
                        "enprod", "downscale", "doctor", "fetch-tables",
                        "fetch-bridges", "setup", "prep", "sim", "go", "adapt",
                        "certify", "dual-run", "multi-run", "obs", "report",
                        "run-plan", "sources", "cycle", "update", "version",
                        "spectral", "spectral-op") or getattr(args, "func", None) is not None:
        # The `or` clause is not decoration.  The tuple is a transcribed
        # list of subcommand names, and a subcommand added without a
        # line here did not get "unrouted" -- it fell through to the
        # config-driven path below and met
        # `read_config_authority(args.config)` on a namespace that has
        # no `config`, which is an AttributeError traceback at a user
        # for the crime of typing a command that exists.  A registrar
        # that set `func` has said where its command goes; that is the
        # thing to dispatch on, and it cannot go stale.  The tuple stays
        # because `spectral` is routed by name and carries its own
        # sub-dispatch rather than a top-level `func`.
        return args.func(args)

    if args.command == "cases":
        records = cases.manifest()
        if args.json:
            import json
            for record in records:
                record["door"] = case_door(record)
            print(json.dumps(records, indent=2, sort_keys=True))
        else:
            # The DOOR, not only the capability tags.  This listing
            # advertises 21 cases while `woof verify` accepts 10, and a
            # reader had no way to tell which was which or how to reach
            # the other 11: `script` is a tag, not an instruction.  Now
            # every row says what to type.
            width = max((len(r["name"]) for r in records), default=24)
            for record in records:
                print(f"{record['name']:<{width}}  "
                      f"{','.join(record['capabilities']):<14}  "
                      f"{case_door(record)}")
            print()
            print("The door is the command that runs the case.  "
                  "`woof verify` grades a case against its registered "
                  "GATES and only cases carrying the `verify` capability "
                  "have any; a `script` case is run through its module, "
                  "which prints its own result.")
        return 0

    if args.command == "verify":
        metrics = _CASES[args.case].run(outdir=args.outdir)
        # List-valued metrics are time series (wk82 w_max_hist) that would
        # swamp the terminal; gates only ever bound scalars.
        print({k: v for k, v in metrics.items() if not isinstance(v, list)})
        bad = _failing_gate(args.case, metrics)
        if bad is not None:
            print(f"FAIL: gate {bad!r} = {metrics[bad]}", file=sys.stderr)
            return 1
        return 0

    if args.command == "import-namelist":
        from woof.namelist_import import import_namelists
        if args.output is not None and args.output.resolve() in (
                args.wps.resolve(), args.input.resolve()):
            raise ValueError(
                f"--output {args.output} aliases an input namelist; "
                "refusing to overwrite the source file.")
        try:
            toml_text, report = import_namelists(
                args.wps, args.input, name=args.name,
                urban_ucm_generation=args.urban_ucm_generation,
                urban_controls={key:getattr(args,key) for key in ("urban_treeoption","urban_ttscheme","urban_fvg","urban_table_dir") if getattr(args,key) is not None},
                static_cache_root=(None if args.static_cache_root is None
                                   else args.static_cache_root.resolve()),
                **({} if args.rrtmg_variant is None else
                   {"rrtmg_variant": args.rrtmg_variant}),
                acknowledgements=tuple(args.ack), geogrid_tbl=args.geogrid_tbl,
                terrain_smoothing_precision=args.terrain_smoothing_precision,
                **({} if args.wrf_version is None else {
                    "wrf_version": args.wrf_version,
                    "wrf_version_source":
                        f"--wrf-version {args.wrf_version}"}))
        except NotImplementedError as error:
            # validate_run_config raises its "not executable yet" refusals
            # (e.g. ra_lw_physics=1, WRF RRTM longwave) as
            # NotImplementedError.  Reached through the importer they are
            # documented user-facing refusals exactly like the importer's
            # own ValueError contract messages -- print the message on the
            # uniform CLI refusal boundary, never a traceback.  Observed
            # live against a WRF-Runner-generated namelist pair carrying
            # ra_lw_physics=1 (2026-07-30 interop verification).
            print("woof import-namelist: " + _layer(error, args),
                  file=sys.stderr)
            return 2
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            # Atomic publication: a mid-write failure must never leave a
            # truncated TOML behind.
            tmp = args.output.with_suffix(args.output.suffix + ".tmp")
            with open(tmp, "w", newline="\n") as fh:
                fh.write(toml_text)
            os.replace(tmp, args.output)
            print(f"import-namelist: wrote {args.output}")
        print(report.format())
        return 0

    if args.command == "resume":
        # THE RUN DIRECTORY ANSWERS FIRST.  A resume is handed the run's
        # own --outdir, so before anything is refused about the
        # experiment argument the directory is asked what it holds: a
        # downscaled child is a route `woof run` cannot continue at all
        # (woof.resume.offline_child_resume_refusal says why, by name),
        # and the run's own recorded configuration is what a resume
        # should be loading whatever the argument spelled.
        from woof.resume import (offline_child_resume_refusal,
                                  offline_child_run_at,
                                  resolve_resume_checkpoint,
                                  resolve_resume_experiment)
        child = offline_child_run_at(args.outdir)
        if child is not None:
            raise ValueError(offline_child_resume_refusal(child))
        experiment = resolve_resume_experiment(args.config, args.outdir)
        # A recipe is refused here, before a checkpoint is looked for, a
        # card locked or a worker started (the breakage is named there).
        from woof.ensemble import recipe_door
        recipe_door.refuse_continuation("resume", experiment.path)
        if experiment.note is not None:
            # A resolution, not a guess: the file was found by the ladder
            # the refusal would otherwise have printed, so the line says
            # which rung answered and `--explain` shows the rungs that
            # did not.
            print("resume: " + render(layered(
                f"the experiment argument {args.config} is not a readable "
                f"configuration file; loading {experiment.path}, "
                f"{experiment.note}",
                "Tried, in order:\n    "
                + "\n    ".join(tuple(experiment.tried)
                                + (f"{experiment.path}: taken",))),
                explain=explain_enabled(args),
                command=f"woof {args.command}"))
            args.config = experiment.path
        _refuse_members_of_one_checkpoint(args)
        import tomllib
        from woof.experiment import is_experiment_toml

        continuation = tomllib.loads(experiment.path.read_text(encoding="utf-8"))
        if "case_data" not in continuation and is_experiment_toml(experiment.path):
            raise ValueError(
                "prepared-route forecasts resume with woof go CONFIG --restart CHECKPOINT "
                "--prepared-root PREPARED --wps-namelist NAMELIST. Use the original prepared "
                "folder and its bound config and WPS file; resume's declared-input route "
                "cannot reconstruct that preparation's identity.")
        # Locate only; every safety property of the resume (manifest
        # validation, config/setup/physics identity, complete tree set)
        # is the run machinery's own and runs on the path set here.
        resolution = resolve_resume_checkpoint(
            args.outdir, args.from_checkpoint, config=args.config)
        for note in resolution.skipped:
            print(f"resume: skipped newer checkpoint {note}",
                  file=sys.stderr)
        # Disclosures, never conditions: which memory mode this run
        # resolves to and which road wrote the checkpoint.  Printed on
        # the same stream as the continuation line they qualify.
        for note in resolution.notes:
            print(f"resume: {note}")
        print(f"resume: continuing from {resolution.checkpoint}")
        args.restart = resolution.checkpoint
        # Fall through to the run dispatch below.

    if args.command == "branch":
        # The what-if door: prepare a NEW run directory from the source
        # run's checkpoint, then dispatch as an ordinary run against the
        # config that directory now owns.  Every safety property is
        # still the run machinery's -- the branched config carries the
        # checkpoint's restart identity by construction (woof.branch
        # compares the payloads before writing anything), so the restart
        # guard adjudicates this restore exactly as it does a resume's.
        from woof.branch import prepare_branch_from_cli
        from woof.ensemble import recipe_door
        # Before the new run directory is written: a branch continues one
        # checkpointed trajectory, as a resume does.
        recipe_door.refuse_continuation("branch", args.config)
        _refuse_members_of_one_checkpoint(args)
        plan = prepare_branch_from_cli(args)
        if args.prepare_only:
            return 0
        args.config = plan.config_path
        args.restart = plan.checkpoint
        # Fall through to the run dispatch below.

    if args.command == "run":
        from woof.ensemble import recipe_door
        if recipe_door.requested(args) or (
                args.config is not None and recipe_door.configured(args.config)):
            # Time-lagged and multi-model members.  Breakage it prevents on
            # the input-directory doors: --wrfinput and --met-em hold ONE
            # trajectory's prepared files, so every member would be that
            # trajectory and the ensemble would report spread it lacks.
            if args.config is None:
                raise ValueError(
                    "--recipe fetches and prepares each member's own source "
                    "trajectory; --wrfinput and --met-em name one trajectory's "
                    "files. Next: woof ensemble CONFIG --recipe time-lagged")
            # The recipe runs in THIS process, ahead of the supervisor: the
            # supervision flags it cannot consume are refused by name.
            recipe_door.refuse_unsupervised(args)
            from woof.ensemble.door import request_for_config
            request = request_for_config(
                args.config, members=getattr(args, "members", None),
                keep_member_files=getattr(args, "keep_member_files", None),
                **recipe_door.flag_overrides(args))
            return recipe_door.run_recipe_ensemble(args, request)
        # A plain member count (N > 1, no recipe named).  This door prepares
        # ONE trajectory, so running it would be N copies of one forecast
        # with zero spread.  The recipe door plans the source's operational
        # ensemble instead, or refuses by name before anything is fetched.
        # A config that is not a readable file is refused further down, in
        # the sentence that says so.
        request = (_plain_member_request(args)
                   if args.config is not None and Path(args.config).is_file() else None)
        if request is not None:
            # The members run in THIS process like any recipe's, ahead of
            # the supervisor.  Breakage it prevents: --gpu-uuid, --restart
            # and the other supervision flags were accepted and read by
            # nothing on this route, so a pinned run put its members on
            # the cards the pin excludes, and a card that does not exist
            # was not even refused.
            recipe_door.refuse_unsupervised(args)
            return recipe_door.run_recipe_ensemble(args, request)

    if args.command == "run" and (args.wrfinput is not None or args.met_em is not None):
        from woof.wrfinput_forecast import run_wrf_forecast
        from woof.metem_forecast import run_metem_forecast
        launch = run_metem_forecast if args.met_em is not None else run_wrf_forecast
        directory = args.met_em if args.met_em is not None else args.wrfinput
        urban_controls = {key:getattr(args,key) for key in ("urban_treeoption","urban_ttscheme","urban_fvg","urban_table_dir") if getattr(args,key) is not None}
        if args.met_em is not None and (args.urban_ucm_generation is not None or urban_controls):
            raise ValueError("urban generation CLI controls currently require --wrfinput; metgrid imports can select them in the resolved experiment TOML")
        unsupported = [flag for flag, value in (
            ("--prep-timeout", args.prep_timeout),
            ("--directory-input-hash", args.directory_input_hash),
            ("--supervisor-max-restarts", args.supervisor_max_restarts != 3)) if value]
        if unsupported:
            raise ValueError("WRF input execution does not consume " + ", ".join(unsupported)
                             + "; remove these CONFIG supervision options")
        from woof.ensemble.door import request_for_inputs
        ensemble_request = request_for_inputs(members=getattr(args, "members", None),
            keep_member_files=getattr(args, "keep_member_files", None))
        from woof.da.analysis_door import request_from_args
        analysis_request = request_from_args(args)
        return launch(directory, args.outdir,
                               run_seconds=args.run_seconds, restart=args.restart,
                               health_debug=args.health_debug, gpu_uuid=args.gpu_uuid,
                               exclusive_gpu=not args.no_supervise,
                               rrtmg_variant=args.rrtmg_variant,
                               **({} if getattr(args,"thompson_version",None) is None else {"thompson_version":args.thompson_version}),
                               **({} if args.wrfinput is None else {"urban_ucm_generation":args.urban_ucm_generation,"urban_controls":urban_controls}),
                               allow_shared_gpu=args.allow_shared_gpu,
                               **({} if ensemble_request is None else
                                  {"ensemble_request": ensemble_request.receipt()}),
                               **({"vertical_grid":args.vertical_grid,
                                   "vertical_levels":args.vertical_levels}
                                  if args.met_em is not None else
                                  ({} if args.soil_source is None else {"soil_source": args.soil_source})),
                               **({"restart_from": args.restart_from}
                                  if args.met_em is None and args.restart_from is not None else {}),
                               **({"analysis": analysis_request}
                                  if args.met_em is None and analysis_request is not None else {}),
                               **({"benchmark": True} if args.benchmark else {}),
                               **({"checkpoint_at": args.checkpoint_at}
                                  if getattr(args, "checkpoint_at", None) is not None else {}))

    # [[domain]]/[experiment] tables route to the experiment path; the
    # legacy [grid]/[dynamics]/[run] shape stays on the frozen case path.
    #
    # A path that is not a readable regular file is refused HERE, in one
    # sentence, before anything opens it.  It used to "fall through to
    # the legacy loader's own error", which meant a typo surfaced as
    # FileNotFoundError, a directory as IsADirectoryError, a zero-byte
    # file as an eight-argument `RunConfig.__init__()` TypeError, and a
    # FIFO as no return at all -- four tracebacks at exit 1 where every
    # other refusal in this product is a sentence at exit 2.
    #
    # The environment-bound branch is deliberately NOT guarded: a
    # multi-run worker is handed the ORIGINAL config path purely as the
    # identity its payload is bound to, and `read_config_authority`
    # checks that binding itself.  Requiring that path to still be a
    # readable file on the worker would refuse a legitimate run.
    from woof.config_authority import read_config_authority
    config_authority = read_config_authority(args.config)
    if (config_authority is not None
            and is_experiment_toml_bytes(config_authority.payload)):
        from woof import runtime
        from woof.case_data import load_experiment_case
        # `static` builds geography only: runtime.write_static reads
        # geog_root, the WPS namelist and the projection, and never opens
        # the forcing GRIB or the Vtable.  Requiring those on disk made
        # "build 30 m terrain here" refuse until a whole met cycle had
        # been downloaded -- a gate on bytes this command does not read,
        # sitting across the documented terrain path.  Both are still
        # DECLARED; only their presence is scoped to the readers.
        exp, data = load_experiment_case(
            args.config, require_met_inputs=args.command != "static")
        if args.command == "static":
            output = runtime.write_static(exp, data, args.output)
            print(f"static {exp.name}: {output}")
        elif args.command == "ingest":
            output = runtime.write_ingest(exp, data, args.output)
            print(f"ingest {exp.name}: {output}")
        else:
            # BEFORE THE RUN IS LAUNCHED, not inside it.  The machine
            # preconditions (woof.config.RUN_PREPARATION_PRECONDITIONS)
            # are raised by ``initialize_real`` as the floor, and on this
            # route that floor sits inside runtime's time loop -- after
            # the snapshot has been read and interpolated onto the grid.
            # The question is about the install and is answerable here,
            # so the refusal costs the reader nothing.
            from woof.config import validate_experiment_preparation

            from woof.tables import _ensure_for_load as ensure
            ensure(exp)
            validate_experiment_preparation(exp)
            if not args.no_supervise:
                from woof.supervisor import supervise_from_cli
                return supervise_from_cli(args)
            if getattr(args, "preprocess_backend", None) is not None:
                from dataclasses import replace
                data = replace(data, preprocess_backend=args.preprocess_backend)
            from woof.ensemble.door import request_for_payload, production_run_scope
            request = request_for_payload(config_authority.payload,
                members=getattr(args, "members", None),
                keep_member_files=getattr(args, "keep_member_files", None))
            with production_run_scope(request, output_directory=args.outdir,
                                      **({"restart_roster": args.restart_roster}
                                         if getattr(args, "restart_roster", None) is not None else {})):
                summary = runtime.run_experiment(exp, data, args.outdir,
                                                 restart=args.restart,
                                                 health_debug=args.health_debug)
            print({"experiment": exp.name, "outdir": str(args.outdir),
                   "wrfout_count": len(summary.wrfout_paths),
                   "completed_seconds": summary.completed_seconds,
                   "nan_free": summary.nan_free,
                   "microphysics_transition_receipt": (
                       None if getattr(
                           summary, "microphysics_transition_receipt", None
                       ) is None else str(
                           summary.microphysics_transition_receipt)),
                   "microphysics_transition_receipt_sha256":
                       getattr(
                           summary,
                           "microphysics_transition_receipt_sha256", None),
                   "restarted": args.restart is not None})
        return 0

    if getattr(args, "preprocess_backend", None) is not None:
        raise ValueError(
            f"--preprocess-backend pins an experiment config's preparation, "
            f"and {args.config} is a legacy [run] config whose frozen case "
            "path does not read it; refusing to drop it and run the case's "
            "own preparation under your pin")
    if getattr(args, "members", None) is not None or getattr(args, "keep_member_files", None):
        # Breakage it prevents: the frozen case path below integrates one
        # forecast and opens no ensemble session, so the member count
        # would be read and dropped, and one forecast would run under it.
        raise ValueError(
            f"--members and --keep-member-files make an ensemble, and {args.config} is a "
            "[grid]/[dynamics]/[run] config whose case path runs one forecast and opens "
            "no ensemble session; refusing to drop them and run one forecast under an "
            "ensemble's name. Next: woof ensemble CONFIG --members N on the config "
            "woof domain wrote.")
    if args.command == "run":
        from types import SimpleNamespace
        from woof.config import load_device_options
        from woof.core.devices import refuse_unrouted_devices
        refuse_unrouted_devices(SimpleNamespace(devices=load_device_options(args.config)),
                                "woof run legacy case")
    cfg = load_config(args.config)
    if args.command == "run":
        from woof.tables import _ensure_for_load as ensure
        ensure(cfg)
    if not cfg.case:
        raise ValueError(
            f"config file {args.config} must set case in the [run] table")
    if cfg.case not in _REAL_CASES:
        raise ValueError(
            f"config case {cfg.case!r} is not a registered real case; "
            f"choose one of {sorted(_REAL_CASES)}")
    case = _REAL_CASES[cfg.case]
    if args.command == "static":
        output = case.write_static(cfg, args.output)
        print(f"static {cfg.case}: {output}")
    elif args.command == "ingest":
        output = case.write_ingest(cfg, args.output)
        print(f"ingest {cfg.case}: {output}")
    else:
        summary = case.run_config(cfg, args.outdir, restart=args.restart)
        print({"case": cfg.case, "outdir": str(args.outdir),
               "wrfout_count": len(summary.wrfout_paths),
               "completed_seconds": summary.completed_seconds,
               "nan_free": summary.nan_free,
               "restarted": args.restart is not None})
    return 0


if __name__ == "__main__":
    sys.exit(main())
