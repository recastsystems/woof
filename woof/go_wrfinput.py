from pathlib import Path
import os

"""Delegate the stock-directory go door to its shared execution owner."""


def launch(args):
    from woof.wrfinput_go import launch as launch_inputs
    return launch_inputs(args)


def add_arguments(parser):
    from woof.wrfinput_go import add_go_input_arguments
    add_go_input_arguments(parser)


def launch_if_selected(args):
    from pathlib import Path
    from woof.go_cli import GoRefusal
    directory = getattr(args, "wrfinput", None)
    config = getattr(args, "config", None)
    if directory is None and config is not None and Path(config).is_dir():
        directory = Path(config)
    if directory is None and config is None and getattr(args, "restart_from", None) is not None:
        directory = Path(args.restart_from).parent
        args.wrfinput = directory
    if directory is None:
        names = ("run_seconds", "checkpoint_at", "restart_from", "soil_source",
                 "allow_shared_gpu", "no_supervise")
        given = ["--" + key.replace("_", "-") for key in names
                 if getattr(args, key, None) is not None and getattr(args, key) is not False]
        if given:
            raise GoRefusal(", ".join(given) + " requires --wrfinput DIR")
        return None
    return launch(args)


def supplied_options(tokens):
    """Retain explicit default-valued flags for the native directory door."""
    import argparse
    from woof.go_cli import _add_go_arguments
    parser = argparse.ArgumentParser(add_help=False)
    _add_go_arguments(parser)
    from woof.explain import add_explain_flag
    add_explain_flag(parser)
    tokens = [str(token) for token in tokens]
    for action in parser._actions:
        action.default = argparse.SUPPRESS
    parser._defaults.clear()
    return frozenset(vars(parser.parse_args(tokens)))


def launch_if_global_directory(args, *, observer=None):
    directory = getattr(args, "config", None)
    if directory is None or not Path(directory).is_dir() or not (Path(directory) / "namelist.input").is_file():
        return None
    from woof.fortran_namelist import parse_namelist
    from woof.global_namelist import is_global, import_global_namelist
    if not is_global(parse_namelist(Path(directory)/'namelist.input')):
        return None
    from woof.go_cli import GoRefusal, _add_go_arguments, _checkpoint_retention
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    _add_go_arguments(parser)
    defaults = vars(parser.parse_args([]))
    consumed = {"config", "outdir", "render_products", "restart", "dry_run",
                "keep_checkpoints", "no_memory_gate", "verify_visuals",
                "members", "keep_member_files", "run_stamp"}
    supplied = getattr(args, "_supplied_go_options", ())
    unsupported = ["--" + key.replace("_", "-") for key, default in defaults.items()
                   if key not in consumed and
                   (key in supplied or getattr(args, key, default) != default)]
    if observer is not None:
        unsupported.append("structured go stage observer")
    if unsupported:
        raise GoRefusal("The WRF input directory already supplies its grid, physics and forcing. "
                        "This route cannot consume " + ", ".join(unsupported))
    # Honor the shared ensemble owner, including inherited run-plan
    # requests. One input directory must not silently duplicate members.
    from woof.ensemble.door import request_for_inputs
    request_for_inputs(members=getattr(args,'members',None),
                       keep_member_files=getattr(args,'keep_member_files',None))
    if getattr(args,'dry_run',False):
        import_global_namelist(Path(directory)/'namelist.input')
        print(f'go: validated native global namelist {directory}; forecast not started')
        return 0
    from woof.global_forecast import run
    output=getattr(args,'outdir',None) or Path(directory).with_name(Path(directory).name+'-go')
    from datetime import datetime
    from woof import run_stamp
    _, receipt = import_global_namelist(Path(directory)/"namelist.input")
    output = run_stamp.resolve(output, init=datetime.fromisoformat(receipt["start"]),
                               enabled=run_stamp.run_stamp_enabled(args), create=False)
    products=getattr(args,'render_products',None)
    enabled=getattr(args,'verify_visuals',None)
    if enabled is None:enabled=os.environ.get('WOOF_VERIFY_VISUALS','1').lower()not in ('0','false','off')
    if products is None and enabled:products='500mb_height_winds,850mb_temperature_height_winds'
    from woof.core.resident_admission import memory_gate_override
    from woof.verification_visuals import verification_scope
    with memory_gate_override(bool(getattr(args, "no_memory_gate", False))), \
            verification_scope(enabled), \
            _checkpoint_retention(getattr(args,'keep_checkpoints',None)):
        return run(directory,output,restart=getattr(args,'restart',None),
                   render_products=products)


def launch_if_wrf_directory(args, *, observer=None):
    status = launch_if_global_directory(args, observer=observer)
    if status is not None:
        return status
    directory = getattr(args, "config", None)
    if directory is None or not Path(directory).is_dir() or not (Path(directory)/"namelist.input").is_file():
        return None
    from woof.wrfinput_go import launch as launch_inputs
    return launch_inputs(args, observer=observer)
