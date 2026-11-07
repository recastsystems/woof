"""CLI orchestration for the shared Rust tropical-cyclone verification tool."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import shutil
import subprocess


def run(args):
    from woof import bridges
    binary = bridges.find_artifact("WOOF_TC_VERIFY", bridges.executable_name("tc-verify"),
                                   crate_relative="tools/tc-verify")
    if not binary and bridges.fetch_on_demand("tc-verify"):
        binary = bridges.find_artifact("WOOF_TC_VERIFY", bridges.executable_name("tc-verify"),
                                       crate_relative="tools/tc-verify")
    if not binary:
        raise ValueError("tc-verify executable unavailable; install this release's bridge bundle or build tools/tc-verify")
    forward = ["--help"] if args.tc_verify_help else ["--version"] if args.tc_verify_version else args.tc_verify_args
    return subprocess.run([binary, *forward], check=False).returncode


def register_cli(subparsers):
    parser = subparsers.add_parser("tc-verify", add_help=False,
        help="score TC forecasts against observed best tracks and generate storm pages")
    parser.add_argument("-h", "--help", action="store_true", dest="tc_verify_help")
    parser.add_argument("--version", action="store_true", dest="tc_verify_version")
    parser.add_argument("tc_verify_args", nargs=argparse.REMAINDER)
    # Every token after the door name goes to the binary verbatim (its own
    # fetch/score/export grammar); only a leading --help/--version is read
    # here, and it is forwarded too.  Marked as the obs doors are.
    parser._forward_argv = True
    parser.set_defaults(func=run, handler=run)
