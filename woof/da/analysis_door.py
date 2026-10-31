"""``woof run --wrfinput DIR --analysis BUNDLE``: a WOOF forecast that starts from an analysis.

The bundle (:mod:`woof.io.analysis_bundle`, ``docs/analysis-bundle-wrf.md``) is checked byte
for byte, its grid, vertical coordinate and static fields are checked against the run's, and
the request is rewritten into the WRF-input door's own inputs, so the forecast itself runs the
code a plain ``--wrfinput`` run does.  Four starts:

* **direct insertion** (``--iau off``, the default for a bundle with no increment.nc): the
  run's wrfinput is ``analysis.nc``;
* **IAU** (``--iau on``, the default when the bundle carries increment.nc): the run's wrfinput
  is the background (analysis minus increment, rebuilt beside the run), and the increment
  enters as a held tendency inside every RK stage over the window
  (:class:`woof.da.iau.IauForcing`, ``--iau-mode 3d``);
* **warm start** (``--restart-from WRFRST`` at the valid time): the WRF restart's state with
  the increment added at the restart instant (``--iau off``) or forced over the window;
* **cycling restart** (``--restart CHECKPOINT`` at the valid time): a WOOF checkpoint of the
  member's own run, continued with the increment added or forced.  The same command resumes an
  interrupted analysis leg: a checkpoint inside the window rebuilds the forcing from the bundle
  and integrates on, bit for bit the unbroken leg (the forcing holds no state the bundle and
  the model clock do not determine).

Around every start: the lateral boundary's first record is rebased onto the start state
(:func:`woof.da.analysis_boundary.rebase_first_record`, WRFDA's da_update_bc) whenever the
start is not the wrfinput the boundary was made from; the diagnosed pressure and density are
refreshed after any insertion; ``forecast-start.json`` names the start before the first step;
and the run receipt carries an ``analysis_bundle`` block with every check, every derived file
and the IAU receipt (the fraction applied, the steps forced).

What the door derives goes into ``<outdir>.analysis-work/<manifest digest>`` (or
``--analysis-work``): the rewritten run directory, the background, the rebased boundary, the
split of the increment, and ``door-plan.json`` (the plan the forecast worker reads).

Nothing here knows how the analysis was made.  The engine knows bundles, increments and IAU;
it never knows the producer.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping

# Once per process for each unchanged file (woof.file_digest; same digests, same change detection): the
# door hashed a member leg's derived files when it wrote them and again at every plan load and split read,
# and its checkpoint twice, about 10 s of one core per 5 km leg (single-core law, 2026-10-27).
from woof.file_digest import sha256_file as _sha

PLAN_SCHEMA = "gpuwm.analysis-door-plan/1"
RECEIPT_SCHEMA = "gpuwm.analysis-door/1"
START_SCHEMA = "gpuwm.forecast-start/1"
PLAN = "door-plan.json"
SPLIT = "iau-split.npz"
LEDGER = "applied.json"

#: The ways an increment enters (``--iau-mode``), as :data:`woof.da.iau.IAU_MODES` names
#: them.  ``oneshot`` and ``3d`` run through this door.
DOOR_MODES = ("oneshot", "3d")
#: Modes the cycle tooling has that this door does not run yet, each with the breakage that
#: running it here would cause and the named gap that closes it.
NAMED_GAP_MODES = {
    "split": ("--iau-mode split adds a fraction after each model step, and the domain-tree "
              "executor this door runs on has no after-step hook, so the fractions would never be "
              "added (named gap G12, closes when the tree "
              "executor carries an after-step hook with a resume-exact receipt)"),
    "4d": ("--iau-mode 4d forces one increment per analysis slot over a window centred on the "
           "analysis time; a v1 bundle carries one increment and the run would have to start "
           "half a window before the valid time (named gap G11; closes with slot records in increment.nc and a start at t - W/2)"),
}
IAU_FIELD_SETS = ("all", "dynamics")
#: What an IAU cold start's lateral boundary pairs with (``--iau-boundary``).
IAU_BOUNDARIES = ("background", "analysis")
DEFAULT_BOUNDARY = "background"
#: Until the IAU measurement on a cycled convective case sets it, a bundle with an increment is forced over its window with every field it carries.
DEFAULT_MODE = "3d"
DEFAULT_FIELDS = "all"
#: The WRF names the dynamics field set spreads; the rest (hydrometeor masses and numbers) go
#: in at the window start, as the cycle's ``--iau-fields dynamics`` puts them in.
DYNAMICS_WRF = ("U", "V", "W", "T", "PH", "QVAPOR")
#: The dycore core, WRF name -> DomainState attribute.
CORE_STATE = {"U": "u", "V": "v", "W": "w", "T": "thp", "PH": "php", "MU": "mup"}

_DOOR_FLAGS = (("--iau", "iau"), ("--iau-mode", "iau_mode"), ("--iau-fields", "iau_fields"),
               ("--iau-boundary", "iau_boundary"),
               ("--iau-window-seconds", "iau_window_seconds"), ("--analysis-work", "analysis_work"))


class AnalysisDoorRefusal(ValueError):
    """A ``--analysis`` request the door will not run; the message names why."""


def _refuse(message: str) -> AnalysisDoorRefusal:
    return AnalysisDoorRefusal(f"woof run --analysis: {message}")


# ---------------------------------------------------------------------------
# the flags
# ---------------------------------------------------------------------------
def add_analysis_arguments(parser, *, worker: bool = False) -> None:
    """The door's flags (stdlib only: the parser imports this module)."""

    import argparse

    parser.add_argument(
        "--analysis", type=Path, default=None, metavar="BUNDLE",
        help="WRF inputs: start from an analysis bundle (wrf-analysis-bundle/1, "
             "docs/analysis-bundle-wrf.md): a directory or its manifest.json.  Every file is "
             "checked by SHA-256, and a grid, vertical coordinate or static field other than the "
             "run's is refused by name.  The analysis (or, with --iau on, its background) becomes "
             "the run's wrfinput; with --restart or --restart-from it is added at the restart "
             "instant")
    parser.add_argument(
        "--iau", choices=("on", "off"), default=None,
        help="with --analysis: 'on' starts from the background and forces the increment over the "
             "IAU window inside every RK stage; 'off' inserts the analysis at the start.  Default: "
             "on when the bundle carries increment.nc, off otherwise")
    parser.add_argument(
        "--iau-mode", choices=DOOR_MODES + tuple(NAMED_GAP_MODES), default=None,
        help="with --analysis: how the increment enters.  oneshot = --iau off; 3d (the default with "
             "--iau on) = a held tendency over a forward window.  split and 4d are named gaps of "
             "this door and are refused by name")
    parser.add_argument(
        "--iau-fields", choices=IAU_FIELD_SETS, default=None,
        help="with --analysis and IAU: 'all' (the default) forces every field the increment "
             "carries; 'dynamics' forces winds, theta, vapour and geopotential and puts the "
             "hydrometeors in at the window start")
    parser.add_argument(
        "--iau-boundary", choices=IAU_BOUNDARIES, default=None,
        help="with --analysis and IAU on a cold start: what the lateral boundary's first record "
             "is the coupling of.  'background' (the default, measured the smaller rim "
             "disturbance) keeps the boundary of the state the run starts from; 'analysis' "
             "rebases it onto the analysis and leaves the whole boundary zone to it")
    parser.add_argument(
        "--iau-window-seconds", type=float, default=None, metavar="S",
        help="with --analysis and IAU: the window, overriding the bundle's iau.window_s")
    parser.add_argument(
        "--analysis-work", type=Path, default=None, metavar="DIR",
        help="with --analysis: where the door writes what it derives (the run directory, the "
             "background, the rebased boundary, the plan); default "
             "<outdir>.analysis-work/<manifest digest>")
    if worker:
        # The forecast worker's: the plan its parent resolved and checked.
        parser.add_argument("--analysis-plan", type=Path, default=None, help=argparse.SUPPRESS)


def request_from_args(args) -> dict[str, Any] | None:
    """The door's request from parsed arguments, or None when no door flag was given."""

    request = {"analysis": getattr(args, "analysis", None)}
    for _flag, name in _DOOR_FLAGS:
        request[name] = getattr(args, name, None)
    request["plan"] = getattr(args, "analysis_plan", None)
    if all(value is None for value in request.values()):
        return None
    return request


def refuse_unrouted(args, route: str) -> None:
    """A door flag on a route the door is not wired into.  THE BREAKAGE IT PREVENTS: the flag
    would be read by nothing, and the run would integrate an unanalysed forecast under an
    analysed name."""

    named = [flag for flag, name in (("--analysis", "analysis"),) + _DOOR_FLAGS
             if getattr(args, name, None) is not None]
    if named:
        raise _refuse(f"{', '.join(named)} start a forecast from an analysis bundle on the WRF-input "
                      f"door (--wrfinput DIR); {route} has no analysis hook, so the flag would be read "
                      "by nothing")


# ---------------------------------------------------------------------------
# the plan (parent side)
# ---------------------------------------------------------------------------
def _place(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def _atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(document, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    partial.replace(path)


def _hypsometric_opt(namelist: Path) -> int:
    from woof.fortran_namelist import parse_namelist

    values = parse_namelist(namelist).get("dynamics", {}).get("hypsometric_opt", [2])
    return int(values[0] if values else 2)


def _max_dom(namelist: Path) -> int:
    from woof.fortran_namelist import parse_namelist

    values = parse_namelist(namelist).get("domains", {}).get("max_dom", [1])
    return int(values[0] if values else 1)


def _resolve_options(request: Mapping[str, Any], bundle, *, restarting: bool) -> dict[str, Any]:
    iau = request.get("iau")
    mode = request.get("iau_mode")
    fields = request.get("iau_fields")
    window = request.get("iau_window_seconds")
    boundary = request.get("iau_boundary")
    if mode in NAMED_GAP_MODES:
        raise _refuse(NAMED_GAP_MODES[mode])
    if mode == "oneshot" and iau == "on":
        raise _refuse("--iau on with --iau-mode oneshot: oneshot inserts the whole increment at the start, "
                      "which is --iau off")
    if mode == "3d" and iau == "off":
        raise _refuse("--iau off with --iau-mode 3d: 3d forces the increment over a window, which is --iau on")
    if bundle.increment is None:
        if iau == "on" or mode == "3d":
            raise _refuse("IAU was asked for and the bundle carries no increment.nc, so there is no "
                          "background to start the window from and nothing to force")
        if restarting:
            raise _refuse("a restart adds the increment at the restart instant, and the bundle carries no "
                          "increment.nc")
        iau_on = False
    else:
        iau_on = iau == "on" or mode == "3d" or (iau is None and mode is None)
    if not iau_on:
        for flag, value in (("--iau-fields", fields), ("--iau-window-seconds", window),
                            ("--iau-boundary", boundary)):
            if value is not None:
                raise _refuse(f"{flag} shapes an IAU window and this run inserts the analysis at once")
        return {"on": False, "mode": "oneshot", "selected": "explicit" if (iau or mode) else "bundle default"}
    if window is not None and not (float(window) > 0):
        raise _refuse(f"--iau-window-seconds {window} must be positive")
    if boundary is not None and restarting:
        raise _refuse("--iau-boundary chooses a cold start's boundary; a restart reads its own boundary at "
                      "the restart instant (named gap G8b)")
    return {"on": True, "mode": "3d", "fields": fields or DEFAULT_FIELDS,
            "boundary": boundary or DEFAULT_BOUNDARY,
            "window_seconds": float(window if window is not None else bundle.iau_window_seconds),
            "window_source": "--iau-window-seconds" if window is not None else "manifest iau.window_s",
            "selected": "explicit" if (iau or mode) else "bundle default",
            "default_note": ("3d with every field until the IAU measurement on a cycled case sets the "
                             "default")}


#: G9b: how the dry-mass part of a 3d IAU (MU and its balanced geopotential share) enters the
#: leg.  ``spread``: after each model step whose interval overlaps the window, the step's share
#: of it (overlap / window, the weights summing to one exactly), pressure and density
#: re-diagnosed -- the surface-pressure increment reaches its endpoint at the window end like
#: every other field.  ``at-once``: all of it at the window start (the behaviour before G9b).
#: Breakage ``spread`` removes: the whole MU increment (interior RMS 58 Pa, range -735/+1070 Pa,
#: column interfaces moved up to 103 m, derecho cycle) went in at one instant while every other
#: field was forced over 600 s, a mass/wind imbalance shed as gravity waves at the leg start.
MASS_MODES = ("spread", "at-once")


def mass_mode() -> str:
    mode = os.environ.get("WOOF_IAU_MASS", "spread")
    if mode not in MASS_MODES:
        raise _refuse(f"WOOF_IAU_MASS={mode!r} is not one of {MASS_MODES}")
    return mode


def split_increment(increment: Mapping[str, Any], *, mode: str, fields: str, background: Any,
                    hypsometric_opt: int, coordinate: Path | None = None) -> tuple[dict, dict, dict[str, Any]]:
    """``(at_once, spread, record)``, by WRF name.

    ``background`` is the rebuilt background file (with ``coordinate``, the run's wrfinput), or
    its column already formed in memory (:func:`woof.io.analysis_bundle.column_from_arrays`,
    no ``coordinate``): the same geopotential either way.

    oneshot: everything at once.  3d: everything forced over the window, except what the RK
    forcing cannot carry or the field set puts in at once, which goes in at the window start
    with its own geopotential share, the rest of the geopotential spread:

    * ``MU`` when it moves: the RK forcing has no dry-mass slot with the mass term every
      coupled tendency would need, so the surface-pressure increment and its hydrostatically
      balanced geopotential share are carried apart from the RK forcing (``at_once``); the door
      spreads them over the same window after each model step (G9b, :data:`MASS_MODES`), or
      puts them in at the window start when ``WOOF_IAU_MASS=at-once``;
    * the hydrometeor masses and numbers under ``--iau-fields dynamics``.
    """

    import numpy as np

    record: dict[str, Any] = {"mode": mode}
    if mode == "oneshot":
        record["at_once"] = sorted(increment)
        return dict(increment), {}, record
    at_once_names = []
    if np.any(np.asarray(increment["MU"]) != 0.0):
        at_once_names.append("MU")
        record["mass_at_window_start"] = (
            "MU moves; the RK forcing has no dry-mass slot, so the surface-pressure increment and its "
            "balanced geopotential share are carried apart: spread over the window after each step "
            "(G9b, WOOF_IAU_MASS=spread, the default) unless the hydrometeors also go in at once, "
            "else at the window start")
    else:
        record["mass"] = "MU is zero everywhere: the dry mass does not move and nothing is forced for it"
    if fields == "dynamics":
        at_once_names += [name for name in sorted(increment) if name not in DYNAMICS_WRF and name != "MU"]
    at_once = {name: increment[name] for name in at_once_names}
    spread = {name: value for name, value in increment.items() if name not in at_once and name != "MU"}
    if at_once:
        from woof.io.analysis_bundle import balanced_geopotential

        if background is None:
            raise _refuse("an at-once share needs the background's column for its geopotential")
        share, balance = balanced_geopotential(background, at_once, hypsometric_opt=hypsometric_opt,
                                               coordinate=coordinate)
        at_once["PH"] = share
        from woof.da.host_analysis import binary

        spread["PH"] = binary(increment["PH"], share, subtract=True)[0]
        record["geopotential_split"] = balance
    record["at_once"] = sorted(at_once)
    record["spread"] = sorted(spread)
    return at_once, spread, record


def _write_split(path: Path, at_once: Mapping[str, Any], spread: Mapping[str, Any]) -> str:
    import numpy as np

    arrays = {f"at_once/{name}": np.asarray(value, dtype=np.float64) for name, value in at_once.items()}
    arrays.update({f"spread/{name}": np.asarray(value, dtype=np.float64) for name, value in spread.items()})
    partial = path.with_name(path.name + ".partial.npz")
    np.savez(partial, **arrays)
    partial.replace(path)
    return _sha(path)


def read_split(path: Path, sha256: str) -> tuple[dict, dict]:
    import numpy as np

    if _sha(path) != sha256:
        raise _refuse(f"{path} changed after the door wrote it; the forcing would not be the plan's")
    at_once, spread = {}, {}
    with np.load(path) as data:
        for key in data.files:
            part, name = key.split("/", 1)
            (at_once if part == "at_once" else spread)[name] = np.asarray(data[key], dtype=np.float64)
    return at_once, spread


def split_shapes(forcing_plan: Mapping[str, Any]) -> tuple[dict[str, tuple], dict[str, tuple]]:
    """``(at_once, spread)`` field shapes of a plan's split, by WRF name, with nothing decoded:
    an in-memory plan's own arrays, or the ``.npy`` headers inside ``iau-split.npz``
    (:func:`_write_split` stores them uncompressed, so only each member's header is read)."""

    import zipfile

    import numpy as np
    from numpy.lib import format as npy

    at_once, spread = {}, {}
    if isinstance(forcing_plan.get("at_once"), Mapping):
        for part, out in (("at_once", at_once), ("spread", spread)):
            for name, value in forcing_plan[part].items():
                out[name] = tuple(int(n) for n in np.shape(value))
        return at_once, spread
    with zipfile.ZipFile(Path(forcing_plan["path"])) as archive:
        for member in archive.infolist():
            key = member.filename[:-4] if member.filename.endswith(".npy") else member.filename
            part, name = key.split("/", 1)
            with archive.open(member) as handle:
                version = npy.read_magic(handle)
                header = (npy.read_array_header_1_0 if version == (1, 0) else npy.read_array_header_2_0)
                shape = header(handle)[0]
            (at_once if part == "at_once" else spread)[name] = tuple(int(n) for n in shape)
    return at_once, spread


def _run_inputs(directory: Path) -> dict[str, Path]:
    files = {"namelist.input": directory / "namelist.input", "wrfinput_d01": directory / "wrfinput_d01",
             "wrfbdy_d01": directory / "wrfbdy_d01"}
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise _refuse(f"{directory} lacks {missing}; the analysis door starts a real.exe run directory")
    return files


def _verify_derived(plan: Mapping[str, Any]) -> None:
    for name, entry in plan.get("derived", {}).items():
        path = Path(entry["path"])
        if not path.is_file():
            raise _refuse(f"the door's {name} ({path}) is gone; the run would start from files nobody checked")
        if _sha(path) != entry["sha256"]:
            raise _refuse(f"the door's {name} ({path}) changed after it was written")


def resolve(request: Mapping[str, Any], *, directory: Path, outdir: Path, run_seconds: float | None = None,
            restart: Path | None = None, restart_from: Path | None = None,
            soil_source: Path | None = None) -> dict[str, Any]:
    """Check the bundle against the run and write the plan; return the run's rewritten inputs.

    Returns ``{"directory", "soil_source", "plan", "forecast_defaults"}``: the WRF-input
    directory the run integrates, the soil source it reads, the plan path the worker loads,
    and the producer's model flags (manifest ``forecast.args``).
    """

    from woof.io.analysis_bundle import BundleRefusal, check_run, load_bundle, parse_forecast_args

    source = request.get("analysis")
    if source is None:
        named = [flag for flag, name in _DOOR_FLAGS if request.get(name) is not None]
        if named:
            raise _refuse(f"{', '.join(named)} act on an analysis bundle and no --analysis was given; they "
                          "would be accepted and never used")
        return {"directory": Path(directory), "soil_source": soil_source, "plan": None,
                "forecast_defaults": {}}
    directory = Path(directory).resolve()
    if restart is not None:
        # The worker checks the checkpoint's digest once the tree is restored (_first_application, the
        # ledger): hashed here beside the bundle checks and the plan instead of on the worker's thread.
        from woof.file_digest import prefetch
        prefetch(restart)
    try:
        bundle = load_bundle(source)
    except BundleRefusal as error:
        raise _refuse(str(error)) from None
    inputs = _run_inputs(directory)
    if _max_dom(inputs["namelist.input"]) > 1 or any(directory.glob("wrfinput_d0[2-9]")):
        raise _refuse("the run has nests; an analysis into nests and IAU carried into a nest are named gap G7, so the door starts single-domain runs and "
                      "refuses rather than start the nests from the unanalysed parent")
    restarting = restart is not None or restart_from is not None
    if restart is not None and restart_from is not None:
        raise _refuse("--restart and --restart-from both name the state to continue")
    options = _resolve_options(request, bundle, restarting=restarting)
    if options["on"] and run_seconds is not None and options["window_seconds"] > float(run_seconds) + 1e-6:
        raise _refuse(f"the IAU window is {options['window_seconds']:g} s and the run {float(run_seconds):g} s; "
                      "the forecast would end with part of the increment never applied (G10)")
    work_root = request.get("analysis_work")
    digest = bundle.manifest_sha256
    work = (Path(work_root) if work_root is not None else
            Path(outdir).expanduser().absolute().with_name(Path(outdir).name + ".analysis-work") / digest[:16])
    work = work.absolute()
    plan_path = work / PLAN
    if plan_path.is_file():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        if restart is None:
            raise _refuse(f"{work} already holds a plan for this bundle and no --restart was given; what the "
                          "door derives goes into a fresh directory, or a resume continues the run that wrote it")
        if existing.get("schema") != PLAN_SCHEMA or existing["bundle"]["manifest_sha256"] != digest:
            raise _refuse(f"{plan_path} is not this bundle's plan")
        if existing["iau"] != options:
            raise _refuse(f"{plan_path} ran with IAU {existing['iau']}; this resume asks for {options}.  A "
                          "resume integrates the leg it continues")
        _verify_derived(existing)
        existing = dict(existing)
        existing["resumed_from"] = str(Path(restart).resolve())
        return {"directory": Path(existing["run_directory"]),
                "soil_source": None if existing.get("soil_source") is None else Path(existing["soil_source"]),
                "plan": plan_path, "forecast_defaults": existing.get("forecast_defaults", {})}
    if work.exists() and any(work.iterdir()):
        raise _refuse(f"{work} exists and holds no plan for this bundle; what the door derives goes into a "
                      "fresh directory so the files of two runs cannot mix (remove it, or name another with "
                      "--analysis-work)")
    work.mkdir(parents=True, exist_ok=True)
    try:
        return _write_plan(bundle, options, request=request, directory=directory, inputs=inputs, outdir=outdir,
                           work=work, restart=restart, restart_from=restart_from, soil_source=soil_source,
                           parse_forecast_args=parse_forecast_args, check_run=check_run)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise


def _write_plan(bundle, options, *, request, directory, inputs, outdir, work, restart, restart_from,
                soil_source, parse_forecast_args, check_run) -> dict[str, Any]:
    from woof.io.analysis_bundle import BundleRefusal, read_increment, write_background
    from woof.da import analysis_boundary as wrfbdy_update

    record: dict[str, Any] = {"options": dict(options)}
    derived: dict[str, dict[str, str]] = {}
    boundary_pair_file = None
    # The increment's values as write_background read them, for the split below (one decode of the
    # increment file per leg, not two; single-core law, 2026-10-27).
    increment_read: dict[str, Any] = {}
    if restart_from is not None:
        start = "warm-start"
        run_file = Path(restart_from).resolve()
    elif restart is not None:
        start = "cycling-restart"
        run_file = inputs["wrfinput_d01"]
    else:
        start = "iau" if options["on"] else "direct-insertion"
        run_file = None
    if start in ("direct-insertion", "iau") and bundle.form != "wrfinput":
        raise _refuse(f"the bundle's analysis is in {bundle.form} form, which carries no base state for a cold "
                      "start; it starts only a cycling restart (--restart or --restart-from)")
    try:
        record["checks"] = check_run(bundle, run_file if run_file is not None else inputs["wrfinput_d01"])
    except BundleRefusal as error:
        raise _refuse(str(error)) from None
    hypsometric_opt = _hypsometric_opt(inputs["namelist.input"])
    run_directory = directory
    background = None
    wants_split = bundle.increment is not None and (options["on"] or start in ("warm-start", "cycling-restart"))
    split_job = None
    from woof.file_digest import prefetch as prefetch_digest
    if start in ("direct-insertion", "iau"):
        # Digests the boundary check and the plan read, hashed beside the writes below.
        prefetch_digest(inputs["wrfinput_d01"])
        run_directory = work / "run"
        run_directory.mkdir()
        shutil.copyfile(inputs["namelist.input"], run_directory / "namelist.input")
        if start == "iau":
            record["background"] = write_background(bundle.analysis, bundle.increment,
                                                     run_directory / "wrfinput_d01",
                                                     increment_read=increment_read)
            background = run_directory / "wrfinput_d01"
            prefetch_digest(background)
            if wants_split:
                # The split reads only the increment, the background and the run's coordinate, none of which
                # the boundary below writes: it runs beside the boundary's rebase and the derived files' digests
                # instead of after them (single-core law, 2026-10-29: the IAU leg's door ran
                # write_background, the boundary rebase and the split one after another on one thread).
                split_job = _beside(_split_to_file, increment_read, bundle, options, background,
                                    hypsometric_opt, inputs["wrfinput_d01"], work)
        else:
            _place(bundle.analysis, run_directory / "wrfinput_d01")
        start_file = run_directory / "wrfinput_d01"
        # The boundary's first record is the coupling of what it pairs with (WRFDA
        # da_update_bc when that is not the wrfinput it was made from): for a direct insertion
        # the analysis, the start itself; for IAU, by --iau-boundary, the start (the background,
        # the default) or the analysis.  Measured on a real case (12 km, an increment band that
        # reaches the western and eastern rims, 600 s window): the background boundary left a
        # 3000 m vertical-velocity difference from no-DA of at most 0.050 m/s at +15 min (rms
        # 0.0031), the analysis boundary with its zone left to it 0.095 m/s (rms 0.0046), both
        # on the rims.  Neither ramps the rim with the window (named gap G8i).
        onto = "analysis" if start == "direct-insertion" else options["boundary"]
        target = bundle.analysis if onto == "analysis" else start_file
        try:
            if wrfbdy_update.same_start(inputs["wrfinput_d01"], target):
                shutil.copyfile(inputs["wrfbdy_d01"], run_directory / "wrfbdy_d01")
                record["boundary"] = {"rebased": False, "onto": onto,
                                      "reason": f"the {onto if onto == 'analysis' else 'start'} is byte for byte "
                                                "the wrfinput the boundary was made from"}
            else:
                try:
                    record["boundary"] = {"rebased": True, "onto": onto, **wrfbdy_update.rebase_first_record(
                        inputs["wrfbdy_d01"], target, run_directory / "wrfbdy_d01")}
                except wrfbdy_update.BoundaryUpdateRefused as error:
                    raise _refuse(str(error)) from None
            prefetch_digest(run_directory / "wrfbdy_d01")
            if start == "iau" and onto == "analysis" and not wrfbdy_update.same_start(start_file, bundle.analysis):
                # The first record is the analysis's coupling, not the start's: the WRF-input door
                # holds it to the analysis by the same pair check (prepare_wrf_run
                # boundary_pair_file), and the forcing leaves the boundary zone to the boundary.
                boundary_pair_file = str(bundle.analysis)
                derived["boundary-pair"] = {"path": boundary_pair_file, "sha256": _sha(bundle.analysis)}
            for name in ("namelist.input", "wrfinput_d01", "wrfbdy_d01"):
                path = run_directory / name
                derived[f"run/{name}"] = {"path": str(path), "sha256": _sha(path)}
        except BaseException:
            # the split writes into the work directory the caller removes on a refusal: it ends first
            if split_job is not None:
                split_job.wait()
            raise
        if soil_source is None:
            soil_source = directory
    elif bundle.increment is not None:
        # A restart start: the background the increment was computed against, rebuilt so the
        # worker can say how far the restored state is from it, and (3d) so a mass increment's
        # geopotential share is integrated on its columns.
        background = work / "background.nc"
        record["background"] = write_background(bundle.analysis, bundle.increment, background,
                                                 increment_read=increment_read)
        # Its digest goes into the plan: hashed on a thread beside the split below, read back after it.
        from woof.file_digest import prefetch
        prefetch(background)
        derived["background.nc"] = {"path": str(background), "sha256": None}
    forcing = None
    if wants_split:
        if split_job is not None:
            record["split"], forcing = split_job.result()
        else:
            record["split"], forcing = _split_to_file(increment_read, bundle, options, background, hypsometric_opt,
                                                      inputs["wrfinput_d01"], work)
        derived[SPLIT] = {"path": forcing["path"], "sha256": forcing["sha256"]}
    if "background.nc" in derived and derived["background.nc"]["sha256"] is None:
        derived["background.nc"]["sha256"] = _sha(background)
    defaults = {}
    for flag, value in parse_forecast_args((bundle.manifest.get("forecast") or {}).get("args") or []):
        defaults[flag.lstrip("-").replace("-", "_")] = value
    plan = {
        "schema": PLAN_SCHEMA, "start": start, "bundle": bundle.receipt(), "valid_time": bundle.valid_time,
        "iau": options, "run_directory": str(run_directory), "original_directory": str(directory),
        "soil_source": None if soil_source is None else str(Path(soil_source).resolve()),
        "work": str(work),
        "restart": None if restart is None else str(Path(restart).resolve()),
        "restart_from": None if restart_from is None else str(Path(restart_from).resolve()),
        "hypsometric_opt": hypsometric_opt, "forcing": forcing, "derived": derived, "record": record,
        "boundary_pair_file": boundary_pair_file,
        "forecast_defaults": defaults,
    }
    _atomic_json(work / PLAN, plan)
    print(f"ANALYSIS {bundle.provenance_sentence()}: {start}"
          + (f", IAU {options['mode']} over {options['window_seconds']:g} s ({options['fields']})"
             if options["on"] else ", inserted at once")
          + f"; derived files in {work}", file=sys.stderr, flush=True)
    return {"directory": run_directory, "soil_source": None if soil_source is None else Path(soil_source),
            "plan": work / PLAN, "forecast_defaults": defaults}


def _split_to_file(increment_read, bundle, options, background, hypsometric_opt, coordinate, work):
    """``(split record, forcing plan entry)``: the increment split (:func:`split_increment`) written to the
    work directory's :data:`SPLIT`."""

    from woof.io.analysis_bundle import read_increment

    increment = increment_read or read_increment(bundle.increment)
    at_once, spread, split_record = split_increment(
        increment, mode=options["mode"], fields=options.get("fields", DEFAULT_FIELDS),
        background=background, hypsometric_opt=hypsometric_opt, coordinate=coordinate)
    split_path = work / SPLIT
    forcing = {"path": str(split_path), "sha256": _write_split(split_path, at_once, spread),
               "at_once": sorted(at_once), "spread": sorted(spread)}
    return split_record, forcing


#: The rebuilt background's fields the background match compares, WRF names.
BACKGROUND_MATCH_FIELDS = ("U", "V", "W", "PH", "MU", "QVAPOR", "T")


def _background_fields(path) -> dict[str, Any]:
    """The rebuilt background's :data:`BACKGROUND_MATCH_FIELDS` it carries, float64 first records, by WRF name."""

    from woof.io.analysis_bundle import _first_record, _open, _prefetch_unmasked

    dataset = _open(path)
    names = [name for name in BACKGROUND_MATCH_FIELDS if name in dataset.variables]
    _prefetch_unmasked(dataset, names)
    return {name: _first_record(dataset.variables[name]) for name in names}


def _door_reads(forcing_plan, background_path) -> dict[str, Any]:
    """:class:`AnalysisInitialization`'s reads of the door's files, ``{"split": ..., "background": ...}``, each
    ``(value, None)`` or ``(None, error)``: an error is raised where that read is used, as before."""

    def attempt(read):
        try:
            return read(), None
        except Exception as error:  # noqa: BLE001 - raised by AnalysisInitialization._read
            return None, error

    out: dict[str, Any] = {}
    if forcing_plan is not None:
        out["split"] = attempt(lambda: read_split(Path(forcing_plan["path"]), forcing_plan["sha256"]))
    if background_path is not None:
        out["background"] = attempt(lambda: _background_fields(background_path))
    return out


class _beside:
    """``function(*args)`` on its own thread from now; :meth:`result` joins it and returns its value or raises
    its exception, :meth:`wait` joins it and ignores either.  The door's work beside its main line."""

    def __init__(self, function, *args):
        import threading

        self._value = self._error = None

        def run():
            try:
                self._value = function(*args)
            except BaseException as error:  # noqa: BLE001 - raised to the caller at result()
                self._error = error

        self._thread = threading.Thread(target=run, name="analysis-door-beside", daemon=True)
        self._thread.start()

    def wait(self) -> None:
        self._thread.join()

    def result(self):
        self._thread.join()
        if self._error is not None:
            raise self._error
        return self._value


def hypsometric_opt_of(namelist: Path) -> int:
    """The run's hypsometric_opt as the door reads it from its namelist (the split's geopotential)."""

    return _hypsometric_opt(Path(namelist))


class _ExchangeAsBundle:
    """What :func:`_resolve_options` reads of a bundle, for an exchange."""

    def __init__(self, exchange):
        self.increment = None if exchange.increment is None else "in memory"
        self.iau_window_seconds = exchange.iau_window_seconds


#: The plan form of an analysis held in memory: the door's own keys, the forcing as arrays.
IN_MEMORY = "in-memory"


def plan_from_exchange(exchange, request: Mapping[str, Any] | None = None, *, hypsometric_opt: int,
                       coordinate: Mapping[str, Any], run_grid: Mapping[str, Any] | None = None,
                       run_static_sha256: str | None = None, run_seconds: float | None = None) -> dict[str, Any]:
    """The door's cycling-restart plan for an analysis held in memory: :func:`resolve` and
    :func:`_write_plan` (``--restart CKPT --analysis BUNDLE``) and :func:`read_split`, with nothing
    written.

    ``exchange`` is a :class:`woof.io.analysis_bundle.AnalysisExchange`; ``request`` the door's
    flags as :func:`request_from_args` names them (``iau``, ``iau_mode``, ``iau_fields``,
    ``iau_window_seconds``; ``iau_boundary`` is a cold start's and is refused, as for any
    restart); ``hypsometric_opt`` the run's namelist value; ``coordinate`` the run's wrfinput's
    coefficients (:data:`woof.io.analysis_bundle._COLUMN_COORDINATE`); ``run_grid`` and
    ``run_static_sha256`` that wrfinput's ``grid_fingerprint`` and ``static_fingerprint`` (the
    door's ``check_run``; None records that the caller checked them).  The background the
    increment was computed against is rebuilt from the exchange exactly as ``write_background``
    rebuilds ``background.nc``, its column formed exactly as ``column_from_file`` forms it, and
    the increment split by :func:`split_increment` itself, so ``forcing`` holds the words the
    door's ``iau-split.npz`` holds.
    """

    from collections import ChainMap

    import numpy as np

    from woof.io.analysis_bundle import (BundleRefusal, _background_core, check_exchange_run,
                                          column_from_arrays, parse_forecast_args)

    request = dict(request or {})
    for name in ("analysis", "analysis_work", "plan"):
        if request.get(name) is not None:
            raise _refuse(f"{name} names a file; an exchange is held in memory and its plan writes nothing")
    view = _ExchangeAsBundle(exchange)
    options = _resolve_options(request, view, restarting=True)
    if options["on"] and run_seconds is not None and options["window_seconds"] > float(run_seconds) + 1e-6:
        raise _refuse(f"the IAU window is {options['window_seconds']:g} s and the run {float(run_seconds):g} s; "
                      "the forecast would end with part of the increment never applied (G10)")
    record: dict[str, Any] = {"options": dict(options)}
    metadata = exchange.metadata
    if run_grid is not None:
        if run_static_sha256 is None:
            raise _refuse("run_grid without run_static_sha256: the door checks the run's grid and static fields "
                          "together")
        try:
            record["checks"] = check_exchange_run(metadata["grid"], metadata["static"]["sha256"], run_grid,
                                                  run_static_sha256, "the run's wrfinput")
        except BundleRefusal as error:
            raise _refuse(str(error)) from None
    else:
        record["checks"] = {"measured": False, "reason": "the caller checked the run's grid and static fields"}
    increment = exchange.increment_float64()
    state = exchange.state()
    rows = ((name, np.asarray(state[name], dtype=np.float64), np.asarray(state[name]).dtype)
            for name in sorted(increment))
    background, record["background"] = _background_core(rows, increment, exchange.increment_precision())
    column = column_from_arrays(ChainMap(background, state), coordinate)
    at_once, spread, record["split"] = split_increment(
        increment, mode=options["mode"], fields=options.get("fields", DEFAULT_FIELDS), background=column,
        hypsometric_opt=int(hypsometric_opt))
    at_once = {name: np.asarray(value, dtype=np.float64) for name, value in at_once.items()}
    spread = {name: np.asarray(value, dtype=np.float64) for name, value in spread.items()}
    defaults = {}
    for flag, value in parse_forecast_args((metadata.get("forecast") or {}).get("args") or []):
        defaults[flag.lstrip("-").replace("-", "_")] = value
    return {
        "schema": PLAN_SCHEMA, "form": IN_MEMORY, "start": "cycling-restart", "bundle": exchange.receipt(),
        "valid_time": exchange.valid_time, "iau": options, "hypsometric_opt": int(hypsometric_opt),
        "forcing": {"at_once": at_once, "spread": spread, "names": {"at_once": sorted(at_once),
                                                                     "spread": sorted(spread)}},
        "background": background, "record": record, "boundary_pair_file": None, "derived": {},
        "forecast_defaults": defaults,
    }


def load_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    if plan.get("schema") != PLAN_SCHEMA:
        raise _refuse(f"{path} is not an analysis door plan ({PLAN_SCHEMA})")
    _verify_derived(plan)
    return plan


# ---------------------------------------------------------------------------
# the forecast worker
# ---------------------------------------------------------------------------
def state_increments(increments: Mapping[str, Any], state, cfg) -> dict[str, Any]:
    """WRF-named increments as DomainState-named ones, each checked against the state's own
    field: the dycore core by :data:`CORE_STATE`, the moisture by the run's own active map."""

    import numpy as np

    from woof.ingest.wrfinput import active_moisture_map

    moisture = dict(active_moisture_map(cfg))
    out = {}
    for name, value in increments.items():
        target_name = CORE_STATE.get(name) or moisture.get(name)
        if target_name is None:
            raise _refuse(f"the increment carries {name}, which this run's physics does not carry "
                          f"(mp_physics={getattr(cfg, 'mp_physics', None)}); the analysis was made for "
                          "another configuration")
        target = getattr(state, target_name, None)
        if target is None:
            raise _refuse(f"the increment carries {name} and the state has no {target_name}")
        if tuple(np.shape(target)) != tuple(np.shape(value)):
            raise _refuse(f"the increment's {name} is {np.shape(value)}; the state's {target_name} is "
                          f"{np.shape(target)}")
        out[target_name] = value
    return out


class _StreamedDoorSteps:
    """The door's post-step acts on a root integrated out of a host store.

    THE BREAKAGE IT CLOSES.  On the tiled road the root's ``DomainState`` is
    the snapshot its store was filled from and nothing integrates it
    (``StreamedDomain.refresh_state``).  The door's per-step acts read and
    wrote that snapshot: the dry-mass share of the window (G9b) went into a
    state no step reads, so the mass increment never reached the model while
    G9b counted it applied, and the balance series (MASPT, the G4 promotion
    input) differenced a state that never moved.  Here both act on the
    compute windows the sweep steps (:class:`woof.da.iau.IauSweepLease`):
    the share is added to each window right after its step, before the
    scatter, with the resident words (:meth:`IncrementalUpdate.apply_window`),
    and each window's owned columns of the lowest level are copied into
    domain planes the balance probe reads (:class:`_PlaneSurface`).  The
    step's share is the one the resident observer computes, from the same
    integer-tick clock (``model_seconds`` of the tree runner's observer),
    and is booked once per domain step.
    """

    def __init__(self, door, *, clock, state):
        self.door = door
        self.clock = clock
        self.state = state
        self.used = False
        self.steps = 0
        self.fraction = 0.0
        self.after = None
        self.planes = None
        self.events = []
        self._read_done = None

    def begin_step(self, *, windows: int) -> None:
        door = self.door
        self.used = True
        self.fraction = 0.0
        mass = door._mass
        if mass is not None:
            clock = self.clock
            if clock is None:
                raise _refuse("a streamed root without a domain clock: the dry-mass share of the IAU window "
                              "has no step interval to be measured on (G9b)")
            # the tree runner's post-step model_seconds: the clock still holds the step's start
            before = float(door._mass_last)
            after = (clock.ticks + clock.step_ticks) / clock.tick_den
            overlap = min(after, mass.end_seconds) - max(before, float(mass.start_seconds))
            if overlap > 0 and mass.applied_fraction < 1.0:
                self.fraction = min(overlap / float(mass.window_seconds), 1.0 - mass.applied_fraction)
            self.after = after
        if door._balance is not None:
            import cupy as cp

            if self.planes is None:
                root = self.state
                with cp.cuda.Device(root.mup.device.id):
                    self.planes = {"p": cp.empty(root.mup.shape, dtype=root.p.dtype),
                                   "mup": cp.empty(root.mup.shape, dtype=root.mup.dtype),
                                   "thp": cp.empty(root.mup.shape, dtype=root.thp.dtype)}
            # the probe read last step's planes on the current stream: the windows write them after
            self._read_done = cp.cuda.Event(disable_timing=True)
            self._read_done.record()
            self.events = []

    def after_window(self, tile, spec, window_id: int, *, stream=None) -> None:
        door = self.door
        if self.fraction > 0.0:
            import cupy as cp

            from woof.da import iau

            ty, tx = slice(spec.j0 - spec.cj0, spec.j1 - spec.cj0), slice(spec.i0 - spec.ci0, spec.i1 - spec.ci0)
            door._mass.apply_window(
                tile, self.fraction, lambda name, array: iau.window_of(cp, array, spec, iau.stagger_variant(name)),
                domain_state=door._mass_state, interior=lambda mask: mask[..., ty, tx])
        if self.planes is not None:
            import cupy as cp

            current = stream if stream is not None else cp.cuda.get_current_stream()
            current.wait_event(self._read_done)
            ty, tx = slice(spec.j0 - spec.cj0, spec.j1 - spec.cj0), slice(spec.i0 - spec.ci0, spec.i1 - spec.ci0)
            fy, fx = slice(spec.j0, spec.j1), slice(spec.i0, spec.i1)
            self.planes["p"][fy, fx] = tile.p[0][ty, tx]
            self.planes["mup"][fy, fx] = tile.mup[ty, tx]
            self.planes["thp"][fy, fx] = tile.thp[0][ty, tx]
            event = cp.cuda.Event(disable_timing=True)
            event.record(current)
            self.events.append(event)

    def finish_step(self) -> None:
        door = self.door
        if door._mass is not None:
            door._mass.book(self.fraction)
            door._mass_last = self.after
        self.steps += 1


class _PlaneSurface:
    """The streamed root's lowest-level planes, as the balance probe reads a
    state (:meth:`woof.da.maspt.MasptRecorder._surface`): ``p[0]``,
    ``mup``, ``thp[0]`` from the window interiors of the last sweep
    (:class:`_StreamedDoorSteps`), ``thb`` the domain's base state."""

    def __init__(self, steps: _StreamedDoorSteps, thb):
        self._steps = steps
        self.thb = thb

    def _planes(self):
        import cupy as cp

        stream = cp.cuda.get_current_stream()
        for event in self._steps.events:
            stream.wait_event(event)
        return self._steps.planes

    @property
    def p(self):
        return self._planes()["p"][None]

    @property
    def mup(self):
        return self._planes()["mup"]

    @property
    def thp(self):
        return self._planes()["thp"][None]


class AnalysisInitialization:
    """The WRF-input door's initialization with the bundle's increment applied or forced.

    Every restore is the wrapped initialization's own; the door acts once the tree is
    restored and its clocks are open (:meth:`after_restore`, called by
    :func:`woof.prepared_domain_tree_forecast.run_prepared_tree`), and adds its block to the
    run receipt (:meth:`run_report_blocks`).
    """

    def __init__(self, base, plan: Mapping[str, Any], *, outdir: Path):
        self._base = base
        self.plan = dict(plan)
        self.outdir = Path(outdir)
        self.lateral_boundaries = getattr(base, "lateral_boundaries", None)
        self._receipt: dict[str, Any] = {"schema": RECEIPT_SCHEMA, "start": plan["start"],
                                         "bundle": plan["bundle"], "iau": plan["iau"],
                                         "plan": (None if plan.get("form") == IN_MEMORY else
                                                  str(Path(plan["work"]) / PLAN)),
                                         "record": plan["record"]}
        self._forcing = None
        self._mass = None
        self._mass_state = None
        self._mass_last = None
        self._window = None
        self._balance = None
        self._observe = None
        self._observe_streamed = None
        self._streamed = None
        self._root_grid = None
        # What apply_door_plan reads from the door's files (the increment split; on a restart start, the rebuilt
        # background's dycore fields for the background match), read on a host thread from here, beside the
        # model's construction and the restore, instead of on the stepping thread after them.  Breakage it
        # prevents (single-core law, 2026-10-29: DA member legs on woof 2.9.9
        # spent ~26 s at ~1 core before the engine started; the split's npz read and the background's decode
        # were two of the serial reads).  The same functions read the same files; a failure is raised where the
        # read used to be.
        self._reads = None
        forcing_plan = self.plan.get("forcing")
        split_file = (forcing_plan is not None and not isinstance(forcing_plan.get("at_once"), Mapping)
                      and forcing_plan.get("path") is not None)
        background_file = (self.plan.get("start") in ("warm-start", "cycling-restart")
                           and not isinstance(self.plan.get("background"), Mapping)
                           and self.plan.get("derived", {}).get("background.nc") is not None)
        if split_file or background_file:
            self._reads = _beside(_door_reads, dict(forcing_plan) if split_file else None,
                                  self.plan["derived"]["background.nc"]["path"] if background_file else None)

    def __getattr__(self, name):
        return getattr(self.__dict__["_base"], name)

    def restore_domain(self, domain, grid, bundle, **kwargs):
        return self._base.restore_domain(domain, grid, bundle, **kwargs)

    def verify_inputs(self, inputs):
        return self._base.verify_inputs(inputs)

    def domain_content_sha256(self, bundle):
        return self._base.domain_content_sha256(bundle)

    def domain_metadata(self, bundle):
        return self._base.domain_metadata(bundle)

    def device_terms(self) -> dict[str, int]:
        """The card bytes this door holds beside the restored domain through its window, by name.

        THE BREAKAGE THEY PREVENT.  The tree door prices a leg the card cannot hold resident as
        the restored domain plus tiles in what it leaves
        (:func:`woof.prepared_domain_tree_forecast._external_door_streams`), before the door
        acts.  The door then put its forcing and its dry-mass increment on the card, priced
        nowhere, so the plan spent memory that was already gone: camp-us-20230710-r n5q15
        10:00 mem003 at 13.10 GiB free planned 147x146 x12 tiles in 9.30 GiB and ran out of
        card memory on its first step (DA-CYCLE-SPEED-2026-10-27/PROGRESS.log 20:16:02Z).

        Each term is what :meth:`apply_door_plan` allocates and keeps for the leg, from the
        split's own shapes: the IAU forcing at the state's float32
        (:class:`woof.da.iau.IauForcing`), the G9b dry-mass increment in float64
        (:meth:`woof.da.iau.IncrementalUpdate._upload`) and the balance probe's three surface
        planes on the tiled road (:class:`_StreamedDoorSteps`).  An at-once insertion's words
        are released once it is in, so they hold nothing through the steps.  A resume past the
        window allocates none of it; the terms are the plan's, so it is priced as the leg that
        does.
        """
        import math

        options = self.plan.get("iau") or {}
        forcing_plan = self.plan.get("forcing")
        terms: dict[str, int] = {}
        grid = (self.plan.get("bundle") or {}).get("grid") or {}
        if grid.get("nx") and grid.get("ny"):
            terms["balance probe planes"] = 3 * 4 * int(grid["nx"]) * int(grid["ny"])
        if not options.get("on") or forcing_plan is None:
            return terms
        at_once, spread = split_shapes(forcing_plan)
        if spread:
            terms["IAU forcing (float32)"] = 4 * sum(math.prod(shape) for shape in spread.values())
        if "MU" in at_once and mass_mode() == "spread" and set(at_once) <= {"MU", "PH"}:
            terms["dry-mass increment (float64)"] = 8 * sum(math.prod(shape) for shape in at_once.values())
        return terms

    # -- the door's acts ----------------------------------------------------
    def _first_application(self, restart: Path | None) -> tuple[bool, str]:
        """Whether the analysis goes in now, at the analysis instant, and why."""

        if restart is None:
            return True, "a cold start at the analysis time"
        if self.plan.get("form") == IN_MEMORY:
            return True, "a restored background at the exchange's analysis instant; no file-backed ledger"
        ledger = Path(self.plan["work"]) / LEDGER
        source = _sha(Path(restart))
        if self.plan["start"] in ("direct-insertion", "iau"):
            return False, ("a resume of the door's own cold start at its first instant: what went in at the "
                           "window start is in the checkpoint")
        if not ledger.is_file():
            return True, "the restart is the background the analysis was computed against"
        recorded = json.loads(ledger.read_text(encoding="utf-8"))
        if recorded.get("restart_sha256") == source:
            return True, ("the restart is the same background checkpoint an earlier attempt of this leg "
                          "started from")
        return False, "the restart is this leg's own checkpoint after the analysis went in"

    #: ``continuation`` values :meth:`apply_door_plan` takes besides the door's own restore.
    CONTINUATIONS = ("in-memory",)

    def after_restore(self, *, model, nodes, start_time, run_seconds: float, restart=None) -> None:
        self.apply_door_plan(model=model, nodes=nodes, start_time=start_time, run_seconds=run_seconds,
                             restart=restart)

    def apply_door_plan(self, *, model, nodes=None, start_time, run_seconds: float, restart=None,
                        continuation: str | None = None) -> None:
        """The door's act on a model whose clocks stand where the run continues.

        The body of :meth:`after_restore`, which calls it: the door's restore and a resident
        member session run this one function.  ``continuation="in-memory"`` is the session's
        restart-equivalent continuation (the door's cycling restart without the checkpoint
        round trip): the live model at the analysis instant is the state the analysis was
        computed against, so the analysis goes in now, and there is no restart file to record
        in the ledger.  It is refused at any other instant, because a forcing rebuilt part-way
        through a window would be a resume, which the session never is.
        """
        import numpy as np

        from woof.da import iau
        from woof.io.analysis_bundle import parse_valid_time

        if continuation is not None and continuation not in self.CONTINUATIONS:
            raise _refuse(f"continuation {continuation!r}: one of {list(self.CONTINUATIONS)}")
        if continuation is not None and restart is not None:
            raise _refuse("an in-memory continuation restores nothing, so it names no restart file")
        if nodes is None:
            nodes = {int(node.cfg.grid_id): node for node in model.walk_parent_first()}
        root = nodes[min(nodes)]
        if len(nodes) != 1:
            raise _refuse("the restored tree has nests; named gap G7")
        state, cfg = root.state, root.cfg.run
        valid = parse_valid_time(self.plan["valid_time"])
        t0 = float((valid - start_time).total_seconds())
        elapsed = float(state.elapsed_seconds)
        self._receipt["window_start_seconds"] = t0
        self._receipt["restored_at_seconds"] = elapsed
        if continuation is not None:
            self._receipt["continuation"] = continuation
        if elapsed < t0 - 1e-6:
            raise _refuse(f"the run is at {elapsed:g} s and the analysis is valid at {t0:g} s; the door adds an "
                          "analysis at its own instant, not before it")
        if continuation is not None and abs(elapsed - t0) > 1e-6:
            raise _refuse(f"an in-memory continuation is at {elapsed:g} s and the analysis is valid at {t0:g} s; "
                          "the session applies an analysis at its own instant only (a forcing rebuilt part-way "
                          "through its window would be a resume)")
        options = self.plan["iau"]
        forcing_plan = self.plan.get("forcing")
        at_once = spread = {}
        if forcing_plan is not None and isinstance(forcing_plan.get("at_once"), Mapping):
            # An in-memory plan (plan_from_exchange): the split's words are held in the plan itself, the words
            # iau-split.npz would hold for the same exchange.
            at_once = {name: np.asarray(value, dtype=np.float64) for name, value in forcing_plan["at_once"].items()}
            spread = {name: np.asarray(value, dtype=np.float64) for name, value in forcing_plan["spread"].items()}
        elif forcing_plan is not None:
            at_once, spread = self._read("split", lambda: read_split(Path(forcing_plan["path"]),
                                                                     forcing_plan["sha256"]))
        mass = {}
        if options["on"] and "MU" in at_once:
            if mass_mode() == "spread" and set(at_once) <= {"MU", "PH"}:
                # G9b: the dry mass and its balanced geopotential share follow the window
                mass, at_once = dict(at_once), {}
                self._receipt["mass"] = {"mode": "spread", "fields": sorted(mass)}
            else:
                self._receipt["mass"] = {
                    "mode": "at-once",
                    "why": ("WOOF_IAU_MASS=at-once" if mass_mode() == "at-once" else
                            "other fields go in at once too; their geopotential share is one with the mass's")}
        if continuation is not None:
            first, why = True, ("the resident session's live state at the analysis instant: the background the "
                                "analysis was computed against, continued in memory")
        else:
            first, why = (self._first_application(None if restart is None else Path(restart))
                          if abs(elapsed - t0) <= 1e-6 else
                          (False, "a resume after the analysis instant: the forcing is rebuilt from the bundle"))
        self._receipt["applied_now"] = {"applied": bool(first), "why": why}
        hopt = int(getattr(cfg, "hypsometric_opt", 2))
        if options["on"] and t0 + float(options["window_seconds"]) > float(run_seconds) + 1e-6:
            raise _refuse(f"the IAU window ends at {t0 + float(options['window_seconds']):g} s and the run at "
                          f"{float(run_seconds):g} s; the forecast would end with part of the increment never "
                          "applied (G10)")
        # The balance series (woof.da.maspt): the domain-mean surface-pressure tendency of the
        # first hour from this instant, per step, with the insertion's own jump when something
        # goes in now.  It reads the state and writes nothing.
        from woof.da.maspt import MasptRecorder

        self._balance = MasptRecorder.for_config(cfg, start_seconds=elapsed, horizon_seconds=3600.0)
        self._root_grid = min(nodes)
        if self.plan["start"] in ("warm-start", "cycling-restart") and first:
            # read before anything goes in
            self._receipt["background_match"] = self._background_match(state, cfg)
        if first and at_once:
            self._balance.before_insertion(state)
        if first and at_once:
            increments = state_increments(at_once, state, cfg)
            insert = iau.IncrementalUpdate(increments, start_seconds=t0, window_seconds=1.0,
                                           hypsometric_opt=hopt)
            insert.apply_fraction(state, 1.0)
            self._receipt["at_once"] = {"fields": sorted(at_once), "state_fields": sorted(increments),
                                        "non_negative_clips": dict(insert.clipped),
                                        "rule": "added in float64 at the analysis instant, moisture clipped "
                                                "at zero, pressure and density re-diagnosed"}
            if restart is not None and self.plan.get("form") != IN_MEMORY:
                _atomic_json(Path(self.plan["work"]) / LEDGER,
                             {"restart_sha256": _sha(Path(restart)), "applied_at_seconds": t0,
                              "fields": sorted(at_once)})
        if options["on"]:
            window = float(options["window_seconds"])
            end = t0 + window
            self._window = (t0, end)
            if spread and elapsed < end - 1e-9:
                increments = state_increments(spread, state, cfg)
                nodes_ = [iau.IauNode(increments, iau.box_density(t0, window), label="analysis")]
                rows = (int(getattr(cfg, "spec_bdy_width", 1) or 1)
                        if self.plan.get("boundary_pair_file") else 1)
                forcing = iau.IauForcing(state, nodes_, boundary_rows=rows, provenance={
                    "door": RECEIPT_SCHEMA, "bundle": self.plan["bundle"]["manifest_sha256"],
                    "window_seconds": window, "window_start_seconds": t0,
                    "boundary": ("carries the analysis; the boundary zone is left to it"
                                 if rows > 1 else "the forcing leaves the specified ring to the boundary")})
                for node in forcing.nodes:
                    node.applied = iau.integrate_density(node.knots, -np.inf, elapsed)
                iau.attach(state, forcing)
                self._forcing = forcing
                self._receipt["forcing_attached_at_seconds"] = elapsed
            elif spread:
                self._receipt["forcing"] = "the restart is past the window; the increment is in the state"
            if mass and elapsed < end - 1e-9:
                self._mass = iau.IncrementalUpdate(state_increments(mass, state, cfg), start_seconds=t0,
                                                   window_seconds=window, hypsometric_opt=hopt)
                # a resume inside the window: the checkpoint holds every step's share up to it
                self._mass.applied_fraction = 0.0 if first else min(1.0, max(0.0, (elapsed - t0) / window))
                self._mass_state = state
                self._mass_last = elapsed
                self._receipt["mass"]["resumed_fraction"] = self._mass.applied_fraction
            elif mass:
                self._receipt["mass"]["note"] = "the restart is past the window; the mass increment is in the state"
        self._balance.prime(state)
        self._observe = self._balance.observer(state, start_seconds=elapsed)
        # The tiled road (a leg that does not fit the card resident streams through host memory):
        # the root's DomainState is then the snapshot its store was filled from, and nothing
        # integrates it, so the dry-mass share and the balance probe below act on the compute
        # windows the sweep steps instead (woof.da.iau.IauSweepLease).  A resident root never
        # reads the attribute.
        self._streamed = _StreamedDoorSteps(self, clock=getattr(root, "clock", None), state=state)
        setattr(state, iau.IAU_AFTER_STEP_ATTRIBUTE, self._streamed)
        self._observe_streamed = self._balance.observer(_PlaneSurface(self._streamed, state.thb),
                                                        start_seconds=elapsed)
        _atomic_json(self.outdir / "forecast-start.json", {
            "schema": START_SCHEMA, "start": self.plan["start"], "valid_time": self.plan["valid_time"],
            "grid": self.plan["bundle"]["grid"]["id"],
            "horizontal_sha256": self.plan["bundle"]["grid"]["horizontal_sha256"],
            "run_directory": self.plan.get("run_directory"),
            "analysis_bundle": self.plan["bundle"]["manifest_sha256"],
            "start_seconds": elapsed})

    def _background_match(self, state, cfg) -> dict[str, Any]:
        """How far the restored state is from the background the analysis was computed against:
        max |state - background| per dycore field (theta as WRF's T).  A record, read before
        anything goes in; a large value says the bundle was made against another state."""

        import numpy as np

        held = self.plan.get("background") if isinstance(self.plan.get("background"), Mapping) else None
        entry = self.plan.get("derived", {}).get("background.nc")
        if entry is None and held is None:
            return {"measured": False, "reason": "no background was rebuilt for this start"}
        if held is not None:
            # An in-memory plan carries the rebuilt background's words itself (plan_from_exchange).
            def background(name):
                return None if name not in held else np.asarray(held[name], dtype=np.float64)
        else:
            fields = self._read("background", lambda: _background_fields(entry["path"]))

            def background(name):
                return fields.get(name)
        out: dict[str, Any] = {"measured": True}
        for wrf_name, state_name in (("U", "u"), ("V", "v"), ("W", "w"), ("PH", "php"), ("MU", "mup"),
                                     ("QVAPOR", "qv"), ("T", "thp")):
            reference = background(wrf_name)
            if reference is None or getattr(state, state_name, None) is None:
                continue
            live = getattr(state, state_name)
            host = (live.get() if hasattr(live, "get") else np.asarray(live)).astype(np.float64)
            if wrf_name == "T":
                thb = state.thb.get() if hasattr(state.thb, "get") else np.asarray(state.thb)
                thb = np.asarray(thb, np.float64)
                host = host + (thb if thb.ndim == 3 else thb.reshape(-1)[:, None, None]) - 300.0
            out[wrf_name] = float(np.max(np.abs(host - reference)))
        return out

    def _read(self, what: str, read):
        """The ``what`` read started beside at construction, or ``read()`` here when none was started."""

        if self._reads is not None:
            held = self._reads.result().get(what)
            if held is not None:
                value, error = held
                if error is not None:
                    raise error
                return value
        return read()

    def step_observer(self, *, grid_id, model_seconds, **_event) -> None:
        """The tree runner's per-step seam: the balance series of the root domain."""

        # A streamed root's sweep already put this step's dry-mass share into its compute windows
        # and read its surface (_StreamedDoorSteps); the root's DomainState is not integrated.
        streamed = self._streamed is not None and self._streamed.used
        if self._mass is not None and int(grid_id) == int(self._root_grid) and not streamed:
            before, after = float(self._mass_last), float(model_seconds)
            self._mass_last = after
            overlap = min(after, self._mass.end_seconds) - max(before, float(self._mass.start_seconds))
            if overlap > 0 and self._mass.applied_fraction < 1.0:
                self._mass.apply_fraction(self._mass_state,
                                          min(overlap / float(self._mass.window_seconds),
                                              1.0 - self._mass.applied_fraction))
                self._mass.steps += 1
        if self._observe is not None and int(grid_id) == int(self._root_grid):
            (self._observe_streamed if streamed else self._observe)(float(model_seconds))

    def run_report_blocks(self, model) -> dict[str, Any]:
        """The receipt's ``analysis_bundle`` block; refuses a window the run left unfinished."""

        receipt = dict(self._receipt)
        if self._streamed is not None and self._streamed.used:
            receipt["road"] = {"root": "tiles", "domain_steps": int(self._streamed.steps),
                               "rule": ("the forcing folded per compute window, the dry-mass share added to "
                                        "each window after its step, the balance read from the window "
                                        "interiors (woof.da.iau.IauSweepLease)")}
        if self._balance is not None:
            receipt["balance"] = self._balance.receipt()
        if self._mass is not None:
            mass = self._mass
            receipt["mass_iau"] = mass.receipt(mode="spread", fields_wrf=self._receipt["mass"]["fields"])
            reached = float(model.root.state.elapsed_seconds) if hasattr(model, "root") else None
            if reached is not None and reached >= mass.end_seconds - 1e-6 and abs(mass.applied_fraction - 1.0) > 1e-9:
                from woof.da.iau import IauError

                # the step observer is telemetry to the tree runner (a raise there is dropped), so the
                # mass spread is checked here: a leg past the window with part of MU missing fails
                raise IauError(f"the run reached {reached:g} s, past the IAU window's end "
                               f"{mass.end_seconds:g} s, with {mass.applied_fraction!r} of the dry-mass "
                               "increment applied (G9b)")
        if self._forcing is not None:
            forcing = self._forcing
            receipt["iau_forcing"] = forcing.receipt()
            end = self._window[1]
            reached = float(model.root.state.elapsed_seconds) if hasattr(model, "root") else None
            if reached is not None and reached >= end - 1e-6 and abs(forcing.applied_fraction - 1.0) > 1e-9:
                from woof.da.iau import IauError

                raise IauError(f"the run reached {reached:g} s, past the IAU window's end {end:g} s, with "
                               f"{forcing.applied_fraction!r} of the increment applied (G10)")
        return {"analysis_bundle": receipt}

    def retire_window(self, model) -> dict[str, Any]:
        """Close this analysis on a model that integrates on (a resident member session).

        The door's own G10 check (:meth:`run_report_blocks`) at the window's end, then the
        forcing detached, so the next analysis attaches its own.  THE BREAKAGE IT PREVENTS: a
        session that attached the next analysis over a window still open would either be
        refused by :func:`woof.da.iau.attach` or, detaching early, drop part of the increment
        the files path forces in full.  Returns the receipt block the door's run receipt
        carries (``analysis_bundle``)."""

        from woof.da import iau

        state = model.root.state
        if self._forcing is not None:
            end = self._window[1]
            reached = float(state.elapsed_seconds)
            if reached < end - 1e-6:
                raise _refuse(f"the IAU window ends at {end:g} s and the model is at {reached:g} s; a window is "
                              "retired at its end, with all of its increment applied (G10)")
        receipt = self.run_report_blocks(model)["analysis_bundle"]
        if self._forcing is not None:
            if getattr(state, iau.IAU_STATE_ATTRIBUTE, None) is not self._forcing:
                raise _refuse("the state carries another forcing than this analysis's; two analyses would share "
                              "one window")
            iau.detach(state)
            receipt["retired_at_seconds"] = float(state.elapsed_seconds)
        if vars(state).get(iau.IAU_AFTER_STEP_ATTRIBUTE) is self._streamed and self._streamed is not None:
            delattr(state, iau.IAU_AFTER_STEP_ATTRIBUTE)
        return receipt


__all__ = ["AnalysisDoorRefusal", "AnalysisInitialization", "add_analysis_arguments", "hypsometric_opt_of",
           "load_plan", "plan_from_exchange", "read_split", "refuse_unrouted", "request_from_args", "resolve",
           "split_increment", "state_increments"]
