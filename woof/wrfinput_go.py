"""The go front door for stock WRF inputs, using the existing WRF runner.

Preparation, Rust IO, admission, ensembles, checkpointing and the step loop
remain owned by wrfinput_forecast. Reject changed flags that it cannot read.
"""
from pathlib import Path
from contextlib import nullcontext


def _visual_scope(value):
    from woof.verification_visuals import verification_scope
    return nullcontext() if value is None else verification_scope(value)


def validate_controls(args, *, metem=False):
    """Refuse unused controls before input IO or device admission."""
    if getattr(args, "native_shared_gpu", False):
        args.allow_shared_gpu = True
        args.no_supervise = True
        args.native_shared_gpu = False
    from woof.cli import build_parser
    from woof.go_cli import GoRefusal
    defaults = vars(build_parser(go_only=True).parse_args(["go"]))
    consumed = {
        # The CLI uses this private receipt to identify explicitly supplied flags.
        "_supplied_go_options",
        "command", "func", "ingest_preflight_handler", "interactive", "wrfinput", "outdir", "restart", "allow_shared_gpu",
        "members", "keep_member_files", "render_products", "verify_visuals",
        "history_compression", "no_memory_gate", "explain", "keep_checkpoints",
        "soil_source", "rrtmg_variant", "thompson_version", "restart_from",
        "checkpoint_at", "gpu_uuid", "health_debug",
        "no_supervise", "benchmark", "dry_run", "run_stamp", "run_seconds",
        # CLI scopes apply output formats before dispatch.
        "output_config", "output_format", "icechunk_store",
    }
    flag_inputs = getattr(args, "wrfinput", None) is not None
    if flag_inputs and getattr(args, "config", None) is not None:
        raise GoRefusal("--wrfinput reads its producing namelist.input; select exactly one input owner; remove CONFIG")
    if not flag_inputs:
        consumed.add("config")
    if metem:
        consumed.difference_update({"benchmark", "dry_run", "run_stamp", "soil_source",
                                    "rrtmg_variant", "thompson_version", "restart_from",
                                    "checkpoint_at", "gpu_uuid", "health_debug", "keep_checkpoints"})
        consumed.add("explain")
    unused = [({"render_section": "--section"}.get(name, "--" + name.replace("_", "-"))) for name, value in vars(args).items()
              if name not in consumed and value != defaults.get(name)]
    if unused:
        raise GoRefusal("WRF inputs do not consume " + ", ".join(sorted(unused))
                        + "; launching would ignore the requested setting")


def _dry_run(run, directory, outdir, *, run_seconds=None):
    import json
    from woof.wrfinput_forecast import wrfinput_window_seconds
    seconds = (run.experiment.run_seconds if run_seconds is None
               else wrfinput_window_seconds(run, run_seconds))
    print(json.dumps({"route": "native-wrf", "input": str(directory),
                      "outdir": str(outdir), "grid_ids": list(run.grid_ids),
                      "run_seconds": seconds,
                      "namelist_input": str(run.namelist_input)}, indent=2))
    return 0


def _resolve_inputs(directory, args):
    from woof.wrfinput_door import resolve_wrfinput_run
    return resolve_wrfinput_run(directory, **{
        name: getattr(args, name) for name in (
            "rrtmg_variant", "thompson_version", "restart_from")
        if getattr(args, name, None) is not None})


def _launch_flag_inputs(args):
    validate_controls(args)
    from woof.ensemble.door import request_for_inputs
    request = request_for_inputs(members=getattr(args, "members", None),
                                keep_member_files=getattr(args, "keep_member_files", None))
    from woof.core.resident_admission import memory_gate_override
    from woof.wrfinput_forecast import run_wrf_forecast
    directory = Path(args.wrfinput)
    outdir = args.outdir or directory.parent / (directory.name + "-go")
    from woof import run_stamp
    resolved = _resolve_inputs(directory, args)
    outdir = run_stamp.resolve(outdir, init=resolved.experiment.start_time,
                              enabled=args.run_stamp == "on", create=False)
    if args.dry_run:
        return _dry_run(resolved, directory, outdir, run_seconds=getattr(args, "run_seconds", None))
    from woof.go_cli import _checkpoint_retention
    with memory_gate_override(args.no_memory_gate), _visual_scope(args.verify_visuals), _checkpoint_retention(args.keep_checkpoints):
        return run_wrf_forecast(directory, outdir, restart=args.restart,
            allow_shared_gpu=args.allow_shared_gpu,
            exclusive_gpu=not args.no_supervise,
            benchmark=args.benchmark,
            **{key: getattr(args, key) for key in _GO_INPUT_CONTROLS
               if key not in {"run_seconds", "benchmark", "allow_shared_gpu"}
               and getattr(args, key, None) is not None
               and getattr(args, key) is not False},
            **({"run_seconds": args.run_seconds} if getattr(args, "run_seconds", None) is not None else {}),
            render_products=args.render_products,
            **({"ensemble_request": request.receipt()} if request is not None else {}))


def launch(args, *, observer=None):
    from woof.go_cli import GoRefusal
    from woof.core.resident_admission import memory_gate_override
    from woof.wrfinput_forecast import run_wrf_forecast
    from woof.ensemble.door import request_for_inputs
    from woof import run_stamp

    if observer is not None:
        raise GoRefusal("The WRF-input forecast reports native progress; this "
                        "launch observer cannot account for its stages")
    if getattr(args, "wrfinput", None) is not None:
        return _launch_flag_inputs(args)
    validate_controls(args)
    request = request_for_inputs(
        members=getattr(args, "members", None),
        keep_member_files=getattr(args, "keep_member_files", None))
    directory = Path(args.config)
    run = _resolve_inputs(directory, args)
    root = args.outdir or directory.with_name(directory.name + "-go")
    outdir = run_stamp.resolve(root, init=run.experiment.start_time,
                              enabled=args.run_stamp == "on", create=False)
    if args.dry_run:
        return _dry_run(run, directory, outdir, run_seconds=getattr(args, "run_seconds", None))
    from woof.go_cli import _checkpoint_retention
    with (memory_gate_override(args.no_memory_gate),
          _visual_scope(args.verify_visuals),
          _checkpoint_retention(args.keep_checkpoints)):
        return run_wrf_forecast(
            directory, outdir, restart=args.restart,
            render_products=args.render_products,
            exclusive_gpu=not args.no_supervise,
            benchmark=getattr(args, "benchmark", False),
            **{key: getattr(args, key) for key in _GO_INPUT_CONTROLS
               if key not in {"run_seconds", "benchmark", "allow_shared_gpu"}
               and getattr(args, key, None) is not None
               and getattr(args, key) is not False},
            **({"run_seconds": args.run_seconds} if getattr(args, "run_seconds", None) is not None else {}),
            allow_shared_gpu=getattr(args, "allow_shared_gpu", False),
            **({} if request is None else {"ensemble_request": request.receipt()}))


def add_go_input_arguments(parser):
    """Register only native controls absent from the composed GO parser."""
    def add(flag, *args, **kwargs):
        if flag not in parser._option_string_actions:
            parser.add_argument(flag, *args, **kwargs)
    add('--wrfinput', type=Path, metavar='DIR',
                        help='start from real.exe wrfinput_d0*, wrfbdy_d01 and namelist.input instead of CONFIG')
    add('--no-supervise', action='store_true',
                        help='WRF inputs: integrate in this process under an existing device reservation')
    add('--run-seconds', type=float, default=None,
                        help='WRF inputs: shorten the forecast within its forcing coverage')
    add('--soil-source', type=Path, default=None,
                        help='WRF inputs: original met_em and Vtable directory for soil recovery')
    add('--rrtmg-variant', choices=('rrtmg_legacy', 'rte-rrtmgp'), default=None,
                        help='WRF inputs: preserve legacy RRTMG unless explicitly overridden')
    add('--thompson-version', choices=('wrf_461', 'wrf_44'), default=None)
    add('--restart-from', type=Path, default=None, metavar='WRFRST',
                        help='WRF inputs: start from native WRF restart state')
    add('--checkpoint-at', default=None, metavar='TIMES',
                        help='WRF inputs: scheduled checkpoint times, or end')
    add('--benchmark', action='store_true',
                        help='WRF inputs: record forecast timing without rendering or post-run verification')
    add('--gpu-uuid', default=None,
                        help='WRF inputs: select the supervised worker GPU')
    add('--allow-shared-gpu', action='store_true',
                        help='WRF inputs: allow the existing priced shared-GPU admission')
    add('--health-debug', action='store_true',
                        help='WRF inputs: write detailed health diagnostics')


_GO_INPUT_CONTROLS = (
    'run_seconds', 'soil_source', 'rrtmg_variant', 'thompson_version',
    'restart_from', 'checkpoint_at', 'benchmark', 'gpu_uuid',
    'allow_shared_gpu', 'health_debug',
)


def refuse_unrouted_go_controls(args):
    """A WRF-only control must not disappear on go's CONFIG route."""
    unused = ['--' + key.replace('_', '-') for key in _GO_INPUT_CONTROLS
              if getattr(args, key, None) is not None
              and getattr(args, key) is not False]
    if unused:
        raise ValueError(', '.join(unused) + ' controls WRF-input execution; '
                         'this CONFIG route would not consume it. Add --wrfinput DIR or remove the control')



def run_wrf_go(args):
    """Compatibility name for the composed native-input owner."""
    return launch(args)
