"""A short first-use guide over the complete existing command parser."""
from __future__ import annotations

import argparse
import sys


_START = """Recast WOOF: GPU weather forecasts. woof runs WOOF-ArWen (WRF); woof hex runs WOOF-HEX-A (MPAS)
usage: woof COMMAND [OPTIONS]

Create and launch
  woof domain                   Create a configuration with guided questions
  woof go CONFIG --dry-run      Review the launch plan
  woof go CONFIG                Prepare and run your forecast
  woof sources                  List input models and acquisition routes

Existing WRF files and restarts
  woof run --wrfinput DIR       Start from wrfinput/wrfbdy and namelist.input
  woof run --met-em DIR         Start from WPS met_em and namelist.input
  woof resume CONFIG --outdir RUN_DIR   Continue from a checkpoint

Pictures
  woof render WRFOUT... --series --out DIR     Draw a run's products
  woof render --diff A_RUN B_RUN --out DIR     Run A minus run B

WOOF-HEX-A (preview), regional and global
  woof hex --help               MPAS-style unstructured-mesh model
  woof hex global --help        Global forecasts on WOOF-HEX-A

Set up or troubleshoot
  woof setup --with-geog        Install native tools, tables and geography
  woof doctor                   Check this installation and show remedies
  woof version                  Show the version and code actually running

CONFIG is your .toml file; DIR and RUN_DIR are your folders.
  woof COMMAND --help           All options for a command
  woof --help-all               List every command
"""


class AllCommandsHelp(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        parser._print_message(argparse.ArgumentParser.format_help(parser), sys.stdout)
        parser.exit()


class ForecastParser(argparse.ArgumentParser):
    """Only the top level is abbreviated; command parsers remain exhaustive."""

    def format_help(self):
        if self.prog == "woof":
            return _START
        return super().format_help()

    def _check_value(self, action, value):
        if (self.prog == "woof"
                and isinstance(action, argparse._SubParsersAction)
                and value not in action.choices):
            raise argparse.ArgumentError(
                action, f"invalid choice: {value!r}; run woof --help for common "
                "actions or woof --help-all for every command")
        return super()._check_value(action, value)
