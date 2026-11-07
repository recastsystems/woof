"""Rebuildable native reference door; model execution stays on Linux."""
import os
from pathlib import Path
import subprocess
import sys

#: Where the wheel carries tools/native_dry_reference (setup.py BuildHex stages it).
PACKAGED_REFERENCE = Path("data/native_dry_reference")


def reference_location(package: Path | None = None):
    """``(check_all.sh, tree, installed)``: the source checkout's recipe, else the wheel's copy; None if neither.

    THE BREAKAGE THIS FIXES: verify-dycore looked only for the source tree's tools/native_dry_reference/, which no
    wheel carries, so from an installed woof hex it refused ("requires the source distribution", paper B-1 rerun
    on 0.4.11) and the dycore claim could be checked only from an unpacked sdist.
    """
    package = Path(__file__).resolve().parent if package is None else Path(package)
    tree = package.parents[1]
    script = tree / "tools/native_dry_reference/check_all.sh"
    if script.is_file():
        return script, tree, False
    script = package / PACKAGED_REFERENCE / "check_all.sh"
    if script.is_file():
        # The recipe's tree argument names no src/ or tools/ here: hexcore imports from the install.
        return script, script.parent, True
    return None


def _run(args):
    found = reference_location()
    if found is None:
        print("verify-dycore cannot find tools/native_dry_reference/ in this install or a source tree",
              file=sys.stderr)
        return 2
    script, tree, installed = found
    env = dict(os.environ)
    # The recipe's Python steps import woof.hex and NumPy: the interpreter running this door has both.
    env.setdefault("PYTHON", sys.executable)
    if installed:
        # The receipt's hex_commit is a git HEAD; an install has none, and must not report the HEAD of a
        # repository that happens to hold the environment.
        ceiling = str(Path(__file__).resolve().parents[1])
        env["GIT_CEILING_DIRECTORIES"] = os.pathsep.join(
            filter(None, (ceiling, env.get("GIT_CEILING_DIRECTORIES"))))
    return subprocess.run(["bash", str(script), str(args.work_root.resolve()), str(tree),
                           str(args.jobs), args.backend, args.compiler], check=False, env=env).returncode


def add_verify_dycore_parser(commands):
    parser = commands.add_parser("verify-dycore", help="build or locate pinned native MPAS and check one dry step")
    parser.add_argument("--work-root", type=Path, default=Path(".native-dry-reference"))
    parser.add_argument("--jobs", type=int, choices=range(1,33), default=8)
    parser.add_argument("--backend", choices=("cpu","cuda","both"), default="cuda")
    parser.add_argument("--compiler", choices=("gnu","ifx"), default="gnu",
                        help="GNU with MPI/PnetCDF, or Intel oneAPI; records the actual toolchain")
    parser.set_defaults(handler=_run)
