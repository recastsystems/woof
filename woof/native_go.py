"""The stock real.exe handoff at the go front door.

The native-input runner owns import, device admission, worker supervision,
ensembles, checkpoints, history and Rust maps. This door calls that owner;
it does not translate the handoff into another prepared-data format.
"""
from pathlib import Path
import argparse
import json
import os


def run_native_go(args):
    from woof.go_cli import GoRefusal, _add_go_arguments, _checkpoint_retention
    from woof.ensemble.door import request_for_inputs
    from woof.core.resident_admission import memory_gate_override
    from woof.verification_visuals import verification_scope

    if args.config is not None:
        raise GoRefusal("go requires CONFIG or --wrfinput DIR, one input authority per launch")
    # Inspect the same parser's defaults. A new go flag must acquire an
    # owner here before this route can accept a nondefault request for it.
    parser = argparse.ArgumentParser(add_help=False)
    _add_go_arguments(parser)
    allowed = {"wrfinput", "run_seconds", "outdir", "restart", "render_products",
               "verify_visuals", "no_memory_gate", "keep_checkpoints", "members",
               "keep_member_files", "dry_run", "config", "allow_shared_gpu"}
    unsupported = [action.option_strings[0] for action in parser._actions
                   if action.option_strings and action.dest not in allowed
                   and getattr(args, action.dest, action.default) != action.default]
    if unsupported:
        raise GoRefusal("go --wrfinput does not consume " + ", ".join(unsupported)
                        + "; its domain, physics and clocks come from namelist.input")
    if args.run_seconds is not None and args.run_seconds <= 0:
        raise GoRefusal("--run-seconds must be positive")
    if args.dry_run:
        from woof.wrfinput_door import resolve_wrfinput_run
        run = resolve_wrfinput_run(args.wrfinput)
        print(json.dumps({"route": "native-wrfinput", "directory": str(run.directory),
                          "grid_ids": run.grid_ids, "run_seconds": args.run_seconds}, indent=2))
        return 0
    request = request_for_inputs(members=getattr(args, "members", None),
                                keep_member_files=getattr(args, "keep_member_files", None))
    from woof.wrfinput_forecast import run_wrf_forecast
    output = args.outdir or Path(str(args.wrfinput).rstrip("/") + "-go")
    enabled = args.verify_visuals
    if enabled is None:
        enabled = os.environ.get("WOOF_VERIFY_VISUALS", "1").lower() not in ("0", "false", "off")
    with memory_gate_override(args.no_memory_gate), verification_scope(enabled), _checkpoint_retention(args.keep_checkpoints):
        return run_wrf_forecast(args.wrfinput, output, run_seconds=args.run_seconds,
                                restart=args.restart, render_products=args.render_products,
                                exclusive_gpu=not args.allow_shared_gpu,
                                allow_shared_gpu=args.allow_shared_gpu,
                                ensemble_request=None if request is None else request.receipt())
