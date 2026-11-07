"""Compatibility names for the shared native-input GO owner."""
import math


def validate(args):
    from woof.wrfinput_go import validate_controls
    from woof.go_cli import GoRefusal
    validate_controls(args)
    seconds = getattr(args, "run_seconds", None)
    if seconds is not None and (not math.isfinite(seconds) or seconds <= 0):
        raise GoRefusal("--run-seconds must be positive and finite")


def run(args):
    validate(args)
    from woof.wrfinput_go import launch
    return launch(args)


def add_arguments(parser):
    from woof.wrfinput_go import add_go_input_arguments
    add_go_input_arguments(parser)
