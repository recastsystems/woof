"""``woof hex moving``: the front door of the moving fine region.

A moving run keeps a fine region (for example 3 km inside a 48 km globe) on a
storm: the run-time tracker fixes the storm on the model's own history
frames, a new mesh member is prepared ahead of each move, and at the move the
state is carried onto the new member through the conservative, balanced
remap.  The dycore is the MPAS-exact port; the physics is whatever the case's
``physics_args`` select through the forecast door.

  woof hex moving case --tracks T --cycle UTC --out C   author from an analysis track fix
  woof hex moving products R --best-track B             completed model track and intensity table
  woof hex moving check                       the reduced-mesh jolt check (no inputs needed)
  woof hex moving stage   --case C --casedir D   initial-condition source -> WPS intermediate
  woof hex moving prepare --case C --casedir D   the start member (and corridor frame/template)
  woof hex moving run     --case C --casedir D --out R --hours H   the closed-loop moving run
  woof hex moving fixed   --case C --casedir D --out R --hours H   the fixed-region control
  woof hex moving summarize R                    rewrite a run's summary

``prepare-member``, ``operator`` and ``zone`` are the steps ``run`` starts in
the background; they are listed so a stage can be rerun by hand.  Every
command runs on any machine with this package, the woof engine and (for
``run``, ``fixed`` and ``check --device gpu``) one CUDA card, or with
``--cards K`` (``run``, ``fixed``, ``check --lts``) K cards of one host: there is no
queue, host layout or source checkout behind it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .errors import MpasPortError


def _print(text: str) -> None:
    print(text, flush=True)


def _numbers(text: str, count: int, label: str) -> list[float]:
    try:
        values = [float(part) for part in str(text).split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError(f'{label} must be {count} comma-separated numbers') from None
    if len(values) != count:
        raise argparse.ArgumentTypeError(f'{label} must be {count} comma-separated numbers')
    return values


def _latlon(text: str) -> list[float]:
    return _numbers(text, 2, 'LAT,LON')


def _latlon4(text: str) -> list[float]:
    return _numbers(text, 4, 'LAT,LON,LAT,LON')


def _check(arguments: argparse.Namespace) -> int:
    from .moving_jolt_check import run_check, table

    arms = tuple(a.strip() for a in arguments.arms.split(",") if a.strip())
    receipt = run_check(arguments.out, device=arguments.device, lts=arguments.lts,
                        rate_shift=arguments.rate_shift, arms=arms, limit=arguments.limit,
                        budget=arguments.budget, keep_meshes=arguments.keep_meshes,
                        cards=arguments.cards,
                        cases=None if arguments.cases is None else tuple(
                            c.strip() for c in arguments.cases.split(",") if c.strip()),
                        log=_print)
    _print(table(receipt))
    _print(f"receipt: {Path(arguments.out).expanduser().absolute() / 'moving-check.json'}")
    return 0 if receipt["verdict"]["passed"] else 1


def _author(arguments: argparse.Namespace) -> int:
    from .moving_author import author_case

    case = author_case(arguments.out, track=arguments.tracks or arguments.best_track,
        cycle=arguments.cycle, storm=arguments.storm, best_track=arguments.best_track is not None,
        template=arguments.template, fine_km=arguments.fine_km, background_km=arguments.background_km,
        fine_radius_km=arguments.fine_radius_km, hours=arguments.hours, source=arguments.source,
        dt_seconds=arguments.dt_seconds, physics_args=arguments.physics_arg,
        target=arguments.target, point=arguments.point, user_path=arguments.path, segment=arguments.segment,
        boundary_field=arguments.boundary_field, object_field=arguments.object_field,
        object_threshold=arguments.object_threshold, check_minutes=arguments.check_minutes,
        lead_hours=arguments.lead_hours, from_last_hour=arguments.from_last_hour,
        target_config=arguments.target_config)
    _print(json.dumps(dict(case=str(arguments.out), storm=case['storm'], authoring=case['authoring'],
                           **({'beyond_source': case['beyond_source']} if 'beyond_source' in case else {})),
                      indent=2))
    return 0


def _products(arguments: argparse.Namespace) -> int:
    from .moving_products import products
    _print(json.dumps(products(arguments.run, arguments.out, arguments.best_track), indent=2))
    return 0


def _stage(arguments: argparse.Namespace) -> int:
    from .moving_members import stage_case

    stage = stage_case(arguments.case, arguments.casedir, source=arguments.source, staged=arguments.staged,
                       best_track=arguments.best_track, ibtracs_subset=arguments.ibtracs_subset,
                       ibtracs_file=arguments.ibtracs_file, jobs=arguments.jobs, log=_print)
    _print(json.dumps({k: stage[k] for k in ("source", "cycle", "met")}, indent=2))
    return 0


def _prepare(arguments: argparse.Namespace) -> int:
    from .moving_members import prepare_case

    prepare_case(arguments.case, arguments.casedir, geog=arguments.geog, threads=arguments.threads,
                 tile_workers=arguments.tile_workers, log=_print)
    return 0


def _template(arguments: argparse.Namespace) -> int:
    from .moving_author import mesh_block
    from .moving_members import rotation_template, template_spec, _tool_env

    case = {"mesh": mesh_block(arguments.fine_km, arguments.background_km, arguments.fine_radius_km)}
    grid = rotation_template(case, arguments.out_dir, env=_tool_env(arguments.threads, None), log=_print)
    _print(json.dumps(dict(grid=str(grid), spec=template_spec(case)), indent=2))
    return 0


def _prepare_member(arguments: argparse.Namespace) -> int:
    from .moving_members import prepare_member

    if arguments.prev and arguments.stitch:
        raise MpasPortError("--prev (frozen far field) and --stitch (corridor) are two different member builders")
    prepare_member(arguments.case, arguments.folder or arguments.casedir, arguments.name, arguments.lat,
                   arguments.lon, casedir=arguments.casedir, prev=arguments.prev, stitched=arguments.stitch,
                   rotate_from=arguments.rotate_from,
                   geog=arguments.geog, threads=arguments.threads, tile_workers=arguments.tile_workers, log=_print)
    return 0


def _operator(arguments: argparse.Namespace) -> int:
    import signal
    import threading

    # Row workers supply the parallelism. Spawned interpreters inherit these
    # limits before importing NumPy/SciPy, avoiding a native pool per worker.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    from .moving_members import build_operator

    # The controller cancels the operator's entire process group. Let the
    # coordinator unwind its executor and TemporaryDirectory on SIGTERM;
    # spawned workers retain the default signal disposition. Do not replace
    # an embedding application's handler or install signals off the main thread.
    previous = signal.getsignal(signal.SIGTERM)
    owns_handler = (threading.current_thread() is threading.main_thread()
                    and previous == signal.SIG_DFL)
    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    if owns_handler:
        signal.signal(signal.SIGTERM, terminate)
    try:
        _print(json.dumps(build_operator(arguments.source, arguments.target, arguments.out)))
    finally:
        if owns_handler:
            signal.signal(signal.SIGTERM, previous)
    return 0


def _zone(arguments: argparse.Namespace) -> int:
    from .moving_members import build_zone

    _print(json.dumps(build_zone(arguments.source_grid, arguments.target_grid, arguments.source_static,
                                 arguments.target_static, arguments.source_init, arguments.target_init,
                                 arguments.out)))
    return 0


def _run(arguments: argparse.Namespace) -> int:
    from .moving_run import run_controller

    return run_controller(arguments)


def _fixed(arguments: argparse.Namespace) -> int:
    """The fixed-region control: one member, no moves, the same door flags as a moving segment."""

    from woof.output_formats import argument_values
    from .moving_case import load_case
    from .moving_members import read_stage
    from .moving_run import RUN_STATE_VARIABLES, label_argv, segment_command

    case = load_case(arguments.case)
    casedir = Path(arguments.casedir).expanduser().absolute()
    stage = read_stage(casedir)
    member = json.loads((casedir / f"{arguments.member}-case.json").read_text())
    out = Path(arguments.out).expanduser().absolute()
    out.mkdir(parents=True, exist_ok=False)
    cmd = segment_command(case, member, out, arguments.hours, arguments.history_every_minutes, arguments.engine,
                          init_source=f"{stage['source']} analysis; fixed-region control on member {member['name']}",
                          extra=[*arguments.forecast_arg, *label_argv(arguments), *argument_values(arguments)], cards=arguments.cards,
                          source_id=stage["source"])
    env = {k: v for k, v in os.environ.items() if k not in RUN_STATE_VARIABLES}
    # The shock series is recorded so a moving run can be compared against the same times.
    env.update(WOOF_HEX_MOVING_SHOCK="1", GPUWM_WRF_EXACT="0", WOOF_HEX_MESH_ROWS=member["mesh_rows"])
    if arguments.device is not None:
        from .moving_run import visible_devices_for

        env["CUDA_VISIBLE_DEVICES"] = visible_devices_for(arguments.device)
    (out / "command.json").write_text(json.dumps(dict(command=cmd, case=case["_path"], member=member,
                                                      started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                                time.gmtime())), indent=2) + "\n")
    began = time.monotonic()
    with (out / "forecast.log").open("w") as log:
        rc = subprocess.call(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=str(out))
    (out / "process-wall.txt").write_text(f"{time.monotonic() - began:.2f}\n")
    _print(json.dumps(dict(rc=rc, out=str(out), forecast=str(out / "fc")), indent=2))
    if rc:
        from .moving_run import route_refusal

        refusal = route_refusal(out)
        if refusal is not None:
            print(f"woof hex: {refusal}", file=sys.stderr)
    return rc


def _summarize(arguments: argparse.Namespace) -> int:
    from .moving_run import summarize_folder

    _print(json.dumps(summarize_folder(arguments.run), indent=1, default=float))
    return 0


def _run_arguments(r: argparse.ArgumentParser) -> None:
    from woof.output_formats import add_arguments
    add_arguments(r)
    r.add_argument('--case', required=True, help='the case file (see docs/moving-nest.md)')
    r.add_argument('--casedir', required=True,
                   help='the staged and prepared case folder (woof hex moving stage, then prepare)')
    r.add_argument('--start-member', default='start')
    r.add_argument('--members', default=None,
                   help='folder for members prepared during the run (default: <out>/members)')
    r.add_argument('--out', required=True, help='the run folder (must not exist unless --resume)')
    r.add_argument('--hours', type=float, required=True)
    r.add_argument('--check-minutes', type=float, default=None,
                   help='history and tracker interval; whole macro steps that divide the run (default: the case '
                        "tracker's placement.check_minutes, else 60)")
    r.add_argument('--device', default=None, metavar='N',
                   help="the card the segments run on: the Nth card this process can already see (a card queue's lease "
                        'holds); default: as inherited')
    r.add_argument('--cards', type=int, default=1, metavar='K',
                   help='split the region across the first K visible cards of this host (default 1)')
    r.add_argument('--engine', default=None, metavar='DIR',
                   help='woof tree whose seam files run (default: the installed engine)')
    r.add_argument('--geog', default=None, metavar='DIR', help='WPS_GEOG root for member statics')
    r.add_argument('--forecast-arg', action='append', default=[], metavar='ARG',
                   help='one extra forecast-door argument for every segment (repeatable)')
    from .products_door import add_label_arguments

    add_label_arguments(r)
    r.add_argument('--tracker-module', help='path of a staged woof/hex/storm_tracker.py (default: import it)')
    r.add_argument('--tracker-config', help='JSON TrackerConfig overrides (mode, tc, motion, placement)')
    r.add_argument('--scripted-track', help='[[model seconds, lat, lon], ...]: replaces fixes, never the policy')
    r.add_argument('--seal-every-hours', type=float, default=12.0)
    r.add_argument('--max-preparing', type=int, choices=(1, 2), default=2,
                   help='concurrent member preparations (default 2; at most 2)')
    r.add_argument('--member-retirement-evidence', metavar='DIR',
                   help='opt-in: retire inactive owned grids/init after completed collector receipts in DIR; '
                        'retain statics for later surface rendering')
    r.add_argument('--chain-min-fit', type=float, default=0.5,
                   help='deprecated compatibility option; anticipation uses the last two model fixes')
    r.add_argument('--anticipate-moves', type=int, default=3,
                   help='prepare members for this many future moves along the predicted motion (0 disables anticipation)')
    r.add_argument('--anticipate-hours', type=float, default=36.0,
                   help='do not prepare for moves predicted further ahead than this')
    r.add_argument('--prep-threads', type=int, default=_prep_cores())
    r.add_argument('--prep-tile-workers', type=int, default=_prep_cores())
    r.add_argument('--reuse-km', type=float, default=40.0,
                   help='an existing member within this distance of an anticipated centre is used instead')
    r.add_argument('--verdict-timeout', type=float, default=600.0)
    r.add_argument('--await-within-minutes', type=float, default=120.0)
    r.add_argument('--hold-for-member-seconds', type=float, default=1800.0,
                   help='when a move is decided and its member is still being prepared, hold the forecast '
                        'at that frame up to this long so the move lands on the decided frame (0: do not hold)')
    r.add_argument('--segment-timeout', type=float, default=6 * 3600)
    r.add_argument('--max-retries', type=int, default=2)
    r.add_argument('--resume', action='store_true')
    r.add_argument('--member-fallback', action=argparse.BooleanOptionalAction, default=True,
                   help='after a numerical failure, continue from the recovery carrier on another member')
    r.add_argument('--zone-carry', action='store_true',
                   help='frozen-far-field members: built from the current member, moved with an exact '
                        'carry outside the changed zone (woof.hex.moving_zone)')
    r.add_argument('--whole-domain-remap', action='store_true',
                   help='use the physical-height all-domain remap even for corridor or frozen-far-field '
                        'members; retains their geometry but disables exact far-field carry')
    r.add_argument('--prelaunch', action=argparse.BooleanOptionalAction, default=True,
                   help="start the next member's forecast process while the current one runs")
    r.add_argument('--resident', action=argparse.BooleanOptionalAction, default=True,
                   help='the forecast process continues the next segment after a move, with its state handed '
                        'over in memory and the next member prepared beside the running segment (the MPAS suite '
                        'route on one card; another route keeps one process per segment); --no-resident starts '
                        'a process for every segment')
    r.add_argument('--rotate-template', action=argparse.BooleanOptionalAction, default=None,
                   help='members prepared during the run rotate the premade family template (else the start '
                        "member's mesh) instead of generating their mesh; default: on for a concentric global "
                        'family, off for a corridor or --zone-carry (--rotate-template there is refused); '
                        '--no-rotate-template generates every member')


def _case_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--case", required=True, type=Path, help="the case file (see docs/moving-nest.md)")
    parser.add_argument("--casedir", required=True, type=Path, help="the case folder (staged inputs and members)")


def _prep_cores() -> int:
    """All CPUs available to this process, including container/affinity limits."""
    if hasattr(os, 'sched_getaffinity'):
        return max(1, len(os.sched_getaffinity(0)))
    return max(1, os.cpu_count() or 1)


def _prep_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--geog", type=Path, default=None,
                        help="WPS_GEOG root for the statics (default: rw_mpas_static's own)")
    cores = _prep_cores()
    parser.add_argument("--threads", type=int, default=cores, help="Rust tool threads")
    parser.add_argument("--tile-workers", type=int, default=cores,
                        help="static tile decode workers")


def add_moving_parser(commands) -> None:
    moving = commands.add_parser(
        "moving", help="a moving fine region that follows a storm (check, stage, prepare, run, fixed)",
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = moving.add_subparsers(dest="moving_command", required=True)

    author = sub.add_parser("case", help="author a case from tc-tracks JSON or a best-track analysis row")
    tracks = author.add_mutually_exclusive_group(required=True)
    tracks.add_argument('--tracks', type=Path, help='tc-tracks tracks.json (select --storm TRACK_ID)')
    tracks.add_argument('--best-track', type=Path, help='best-track JSON rows/row, or IBTrACS CSV with --storm SID')
    tracks.add_argument('--point', type=_latlon, help='LAT,LON: where a setup target starts (any --target)')
    tracks.add_argument('--segment', type=_latlon4,
                        help='LAT,LON,LAT,LON: a dryline or front as drawn (--target boundary)')
    tracks.add_argument('--path', type=Path,
                        help='timed points {"rows":[{"time","lat","lon"}...]} the region follows (--target path)')
    author.add_argument('--target', default='tc', choices=('tc', 'low', 'cape-shear', 'boundary', 'path', 'object'),
                        help='what the fine region follows: a tropical cyclone (tc), a surface low, the dominant '
                             'CAPE x 0-6 km shear area, a dryline/front (boundary), the user\'s path, or a '
                             'thresholded feature (object, e.g. reflectivity)')
    author.add_argument('--boundary-field', default='dewpoint', choices=('dewpoint', 'theta_e', 'temperature'),
                        help='the boundary target\'s surface field: dewpoint (dryline), theta_e or temperature (front)')
    author.add_argument('--object-field', default=None, help='the object target\'s history field (default refl10cm)')
    author.add_argument('--object-threshold', type=float, default=None, help='the object target\'s threshold')
    author.add_argument('--check-minutes', type=float, default=None,
                        help='how often the target is fixed and the region may move (default 30; 15 also works)')
    author.add_argument('--target-config', type=Path, default=None,
                        help='JSON merged into the case tracker block (per-target tuning)')
    author.add_argument('--storm', help='track identifier or IBTrACS SID; required for multiple tracks')
    author.add_argument('--cycle', required=True, help='UTC source cycle (with a track: an exact valid position)')
    lead = author.add_mutually_exclusive_group()
    lead.add_argument('--lead-hours', type=int, default=0,
                      help="start from the source's forecast at this lead: the run begins at cycle + lead")
    lead.add_argument('--from-last-hour', action='store_true',
                      help="start from the source's last forecast hour for this cycle (extend a run past its horizon)")
    author.add_argument('--out', required=True, type=Path, help='new case JSON; never overwrites a case')
    author.add_argument('--template', type=Path, help='existing case geometry/physics; event metadata is replaced')
    author.add_argument('--source', default='gfs', help='init-source row for staging')
    author.add_argument('--hours', type=float, default=12.0)
    author.add_argument('--fine-km', type=float, default=3.0)
    author.add_argument('--background-km', type=float, default=None,
                        help='background spacing (default: a power-of-two multiple of fine spacing, at least 48 km)')
    author.add_argument('--fine-radius-km', type=float, default=200.0)
    author.add_argument('--dt-seconds', type=float, help='override spacing-derived step; must align all clocks')
    author.add_argument('--physics-arg', action='append', default=None, help='forecast physics argument (repeatable)')
    author.set_defaults(handler=_author)

    check = sub.add_parser("check", help="the reduced-mesh jolt check: moving vs fixed, with and without balance")
    check.add_argument("--out", type=Path, default=Path("moving-check"), help="receipt folder (default ./moving-check)")
    check.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    check.add_argument("--lts", action="store_true",
                       help="the production full-step LTS executor with changing classes (CUDA only)")
    check.add_argument("--rate-shift", type=float, default=0.25,
                       help="LTS class boundary shift at each move, radians (default 0.25)")
    check.add_argument("--arms", default="fixed,baseline,balanced",
                       help="fixed (reference), baseline (balance off), balanced (the default path)")
    check.add_argument("--limit", type=float, default=1.5, help="largest admissible jolt ratio (default 1.5)")
    check.add_argument("--budget", type=float, default=1e-9, help="largest relative mass change per move")
    check.add_argument("--keep-meshes", action="store_true", help="keep the generated fixture meshes")
    check.add_argument("--cards", type=int, default=1, metavar="K",
                       help="with --lts: cut every rate class across K cards of this host (the multi-card "
                            "class blocks a --cards moving run uses)")
    check.add_argument("--cases", default=None,
                       help="fixture cases, comma separated: gentle (0-100 m terrain, 8 levels) and steep (a 2,700 m "
                            "plateau with an inversion, the production 55 levels). Default: both on a card, gentle "
                            "on the CPU. CPU arms run in parallel with vectorized operators")
    check.set_defaults(handler=_check)

    stage = sub.add_parser("stage", help="the case's initial-condition source -> one WPS intermediate")
    _case_arguments(stage)
    stage.add_argument("--source", default=None,
                       help="init-source row (default: the case's init_source, else gfs; see init-source --list)")
    stage.add_argument("--staged", type=Path, action="append", default=[],
                       help="directory already holding the source's files")
    stage.add_argument("--best-track", action="store_true",
                       help="also fetch the storm's IBTrACS best track (verification only)")
    stage.add_argument("--ibtracs-subset", default="last3years", help="IBTrACS list (last3years, ALL, ...)")
    stage.add_argument("--ibtracs-file", type=Path, default=None, help="a local IBTrACS v04 CSV instead")
    stage.add_argument("--jobs", type=int, default=3)
    stage.set_defaults(handler=_stage)

    prepare = sub.add_parser("prepare", help="the start member (and a corridor case's frame and template)")
    _case_arguments(prepare)
    _prep_arguments(prepare)
    prepare.set_defaults(handler=_prepare)

    template = sub.add_parser(
        "template", help="the premade rotation template of a cap family (the start member is rotated from it)",
        description="Build, or fetch from hex's prep store, the concentric cap family that `moving case` authors "
                    "from these three numbers, centred at 0N 0E. With WOOF_HEX_PREP_STORE_OUTBOX set the "
                    "result is staged there, so an image carries it and `moving prepare` rotates it to the "
                    "storm instead of generating a mesh.")
    template.add_argument("--fine-km", type=float, default=3.0)
    template.add_argument("--background-km", type=float, default=None,
                          help="background spacing (default: as `moving case` picks it)")
    template.add_argument("--fine-radius-km", type=float, default=200.0)
    template.add_argument("--out-dir", type=Path, required=True, help="folder for the template grid")
    template.add_argument("--threads", type=int, default=_prep_cores(), help="Rust tool threads")
    template.set_defaults(handler=_template)

    member = sub.add_parser("prepare-member", help="one member at a centre (run starts these itself)")
    _case_arguments(member)
    member.add_argument("--folder", type=Path, default=None, help="members folder (default: --casedir)")
    member.add_argument("--name", required=True)
    member.add_argument("--lat", type=float, required=True)
    member.add_argument("--lon", type=float, required=True)
    member.add_argument("--prev", default=None, help="build a frozen-far-field member from this member")
    member.add_argument("--stitch", action="store_true", help="stitch from the corridor frame and template")
    member.add_argument("--rotate-from", default=None, metavar="MEMBER",
                        help="rotate an existing concentric global mesh in --folder; rebuild static and init")
    _prep_arguments(member)
    member.set_defaults(handler=_prepare_member)

    run = sub.add_parser("run", help="the closed-loop moving run on one card")
    _run_arguments(run)
    run.set_defaults(handler=_run)

    fixed = sub.add_parser("fixed", help="the fixed-region control: the same member and flags, no moves")
    _case_arguments(fixed)
    from woof.output_formats import add_arguments
    add_arguments(fixed)
    fixed.add_argument("--out", type=Path, required=True)
    fixed.add_argument("--hours", type=float, required=True)
    fixed.add_argument("--member", default="start")
    fixed.add_argument("--history-every-minutes", type=float, default=60.0)
    fixed.add_argument("--device", default=None, metavar="N")
    fixed.add_argument("--cards", type=int, default=1, metavar="K",
                       help="split the region across the first K visible cards of this host (default 1)")
    fixed.add_argument("--engine", default=None, metavar="DIR")
    fixed.add_argument("--forecast-arg", action="append", default=[], metavar="ARG")
    from .products_door import add_label_arguments

    add_label_arguments(fixed)
    fixed.set_defaults(handler=_fixed)

    products = sub.add_parser('products', help='join moving-run track and intensity diagnostics across meshes')
    products.add_argument('run', type=Path)
    products.add_argument('--out', type=Path)
    products.add_argument('--best-track', type=Path, help='best-track JSON, verification only')
    products.set_defaults(handler=_products)

    summarize = sub.add_parser("summarize", help="rewrite a run's per-segment results and summary.json")
    summarize.add_argument("run", type=Path)
    summarize.set_defaults(handler=_summarize)

    operator = sub.add_parser("operator", help="conservative overlap operator between two member grids")
    operator.add_argument("source", type=Path)
    operator.add_argument("target", type=Path)
    operator.add_argument("out", type=Path)
    operator.set_defaults(handler=_operator)

    zone = sub.add_parser("zone", help="zone-carry map between two frozen-far-field members")
    for name in ("source_grid", "target_grid", "source_static", "target_static", "source_init", "target_init",
                 "out"):
        zone.add_argument(name, type=Path)
    zone.set_defaults(handler=_zone)
