"""CLI orchestration for the shared Rust tropical-cyclone verification tool."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import shutil
import subprocess


def run(args):
    binary = os.environ.get("WOOF_TC_VERIFY") or shutil.which("tc-verify")
    if not binary:
        candidate = Path(__file__).parent / "bin" / "tc-verify"
        if candidate.is_file():
            binary = str(candidate)
    if not binary and args.tc_verify_help:
        # THE BREAKAGE THIS PREVENTS (hex 0.5.0 re-seal gate, 2026-11-11): the tool is not in the wheel, so
        # `woof hex tc-verify --help` stopped with rc 2 instead of saying what the command is and how to get it.
        print("usage: woof hex tc-verify [ARGS ...]\n\n"
              "Score TC forecasts against observed best tracks and generate storm pages. The arguments go to the\n"
              "tc-verify Rust tool, which is not installed here: build it from rust/tc-verify, or set WOOF_TC_VERIFY\n"
              "to its path (docs/tc-verify.md). `woof hex tc-verify --help` then prints the tool's own arguments.")
        return 0
    if not binary:
        raise ValueError("tc-verify executable unavailable; install the tc-verify Rust tool or set WOOF_TC_VERIFY (see docs/tc-verify.md)")
    forward = ["--help"] if args.tc_verify_help else ["--version"] if args.tc_verify_version else args.tc_verify_args
    return subprocess.run([binary, *forward], check=False).returncode


def register_cli(subparsers):
    parser = subparsers.add_parser("tc-verify", add_help=False,
        help="score TC forecasts against observed best tracks and generate storm pages")
    parser.add_argument("-h", "--help", action="store_true", dest="tc_verify_help")
    parser.add_argument("--version", action="store_true", dest="tc_verify_version")
    parser.add_argument("tc_verify_args", nargs=argparse.REMAINDER)
    parser.set_defaults(func=run, handler=run)
