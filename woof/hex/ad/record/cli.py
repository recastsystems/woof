# SPDX-License-Identifier: Apache-2.0
"""python -m woof.hex.ad.record {build,record,replay,retain}

  build    build the recorder libraries from source into the cache (record does this itself).
  record   record a hex trajectory (any registered mesh, the suites the sealer catalogs) and seal it into a plan package,
           then replay it strict. Example, one card, 1 h of x1.2562 hrrrv5:
               python -m woof.hex.ad.record record --mesh x1.2562 --suite hrrrv5 --init init.nc --grid grid.nc \\
                   --oro oro.nc --dt 1200 --hours 1 --out pkg --card 0 --serial
  replay   replay a sealed package strict (and optionally the continue/diagnostic replay).
  retain   the retention-list generator alone (gen_immutable.py): RECORDER_A RECORDER_B BASE OUT LABEL [RESIDUAL].
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from . import env
from . import route as routes


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra: list[str] = []
    if "--" in argv:
        extra, argv = argv[argv.index("--") + 1:], argv[:argv.index("--")]
    parser = argparse.ArgumentParser(prog="python -m woof.hex.ad.record", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="build the recorder libraries")
    b.add_argument("--force", action="store_true")

    r = sub.add_parser("record", help="record, seal and replay a trajectory (arguments after -- go to the forecast)")
    r.add_argument("--mesh", default=None, help="mesh name (default with --lam: the case's row)")
    r.add_argument("--suite", default="hrrrv5")
    r.add_argument("--init", type=Path, default=None, help="init file (default with --lam: the case's)")
    r.add_argument("--grid", type=Path, default=None, help="grid file (default with --lam: the case's)")
    r.add_argument("--oro", type=Path, default=None, help="orographic statistics file (the suite's GWD input)")
    r.add_argument("--lam", default=None, metavar="CASE",
                   help="a limited-area mesh: the regional door's work directory (or a file in it), or a LAM_* case "
                        "file; gives the static file, boundaries, start time, mesh rows and contract deck")
    r.add_argument("--fixed-free-mib", type=int, default=None,
                   help="free card memory every forecast reader sees in both recordings (default for a limited-area "
                        "mesh: the card's free memory at the start)")
    r.add_argument("--dt", type=float, default=None, help="timestep in s (default with --lam: the case's)")
    r.add_argument("--hours", type=float, required=True)
    r.add_argument("--out", type=Path, required=True, help="the sealed package directory (replaced)")
    r.add_argument("--work", type=Path, default=None, help="work directory (default OUT.work)")
    r.add_argument("--numerics", choices=("fast", "strict"), default="fast")
    r.add_argument("--card", default="0")
    r.add_argument("--card-b", default=None, help="card for recording B (default: --card)")
    r.add_argument("--cpus", default=None, help="CPU lists A:B for taskset (default: halves of this host)")
    r.add_argument("--serial", action="store_true", help="record A then B on one card")
    r.add_argument("--runner", default=None)
    r.add_argument("--retention", default=None, help="base retention list (default: the shipped list for the row)")
    r.add_argument("--pairs", default=None, help="base CIndexer pairs (default: the shipped base)")
    r.add_argument("--hold-slots", default="", help="physics slots held (default none: radiation and land live)")
    r.add_argument("--route", choices=routes.names(), default=routes.default(),
                   help="the forecast's kernel forms (data/record-routes.v1.json): derivative (default) records the "
                        "forms this release has derivative rows for; forecast records the installed forecast's own "
                        "defaults. Every form gives the same bits. A plan with a kernel kind that has no derivative "
                        "is refused when it is sealed, on either route")
    r.add_argument("--boundary-workers", type=int, default=4)
    r.add_argument("--compile-workers", type=int, default=8)
    r.add_argument("--xdg", type=Path, default=None, help="XDG cache root for the forecasts (default: the user's)")
    r.add_argument("--wait-for-card", type=float, default=0.0, metavar="MINUTES",
                   help="on a shared card, wait up to this long when hex's memory admission refuses a forward")
    r.add_argument("--no-replay", action="store_true")
    r.add_argument("--keep-work", action="store_true", help="keep the recordings after sealing")
    r.add_argument("--ranks", type=int, default=None, metavar="N",
                   help="record the forecast as N rank threads on the one card and, once sealed, cut the plan into one "
                        "plan per rank, OUT-r0 .. OUT-r<N-1> (split_plan.py): the plans a window split over N cards "
                        "runs. Default 1 (or WOOF_HEX_RECORD_RANKS): the one-card recording, unchanged")

    p = sub.add_parser("replay", help="replay a sealed package")
    p.add_argument("package", type=Path)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--card", default="0")
    p.add_argument("--runner", default=None)
    p.add_argument("--continue", dest="continue_", action="store_true", help="also the continue/diagnostic replay")

    t = sub.add_parser("retain", help="retention list from a recorded pair (gen_immutable.py)")
    t.add_argument("args", nargs="+")

    o = parser.parse_args(argv)
    if o.command == "build":
        from .build import build
        build(force=o.force)
        return 0
    if o.command == "retain":
        return subprocess.call([sys.executable, "-B", str(env.TOOLS / "gen_immutable.py"), *o.args])
    if o.command == "replay":
        from .chain import replay
        runner = env.which_runner(o.runner)
        if runner is None:
            env.defect("no replay runner: pass --runner, set HEXAD_RUNNER, or put ad-binder on PATH")
        steps = o.steps
        if steps is None:
            receipt = o.package / "record-receipt.json"
            steps = json.loads(receipt.read_text())["steps"] if receipt.is_file() else None
            if steps is None:
                env.defect(f"{o.package} has no record-receipt.json: pass --steps")
        results = {"strict": replay(o.package.resolve(), steps, runner, o.card, o.package.resolve(), "replay-strict")}
        if o.continue_:
            results["continue"] = replay(o.package.resolve(), steps, runner, o.card, o.package.resolve(),
                                         "replay-continue", continue_=True)
        print(json.dumps(results))
        ok = all(v["rc"] == 0 and not v["mismatch"] and v["boundary_hashes"] for v in results.values())
        return 0 if ok else 3
    # record
    import os
    from .chain import refuse_other_physics_mode
    refuse_other_physics_mode(extra)   # before any card or build: a recording is of the exact-mode forecast
    if o.ranks is not None:   # one word for the whole chain: every stage and tool reads the count from the environment
        if o.ranks < 1:
            env.defect(f"--ranks {o.ranks}: a count of rank threads, 1 or more")
        os.environ["WOOF_HEX_RECORD_RANKS"] = str(o.ranks)
    o.forecast_extra = extra
    o.card_b = o.card_b or o.card
    if o.cpus:
        o.cpus_a, _, o.cpus_b = o.cpus.partition(":")
        o.cpus_b = o.cpus_b or o.cpus_a
    else:  # no pinning: the process's own affinity (a container's cgroup set) is kept
        o.cpus_a = o.cpus_b = None
    if o.lam:   # a limited-area case fills what was not named (lam.py); a word given on the command line wins
        from . import lam as _lam
        try:
            o.lam = _lam.read_case(o.lam)
        except _lam.CaseError as error:
            env.defect(str(error))
        o.mesh = o.mesh or o.lam["row"]
        o.init = o.init or Path(o.lam["init"])
        o.grid = o.grid or Path(o.lam["grid"])
        o.oro = o.oro or (Path(o.lam["oro"]) if o.lam.get("oro") else None)
        o.dt = o.dt or o.lam.get("dt")
    missing = [f"--{k}" for k in ("mesh", "init", "grid", "dt") if getattr(o, k) is None]
    if missing:
        env.defect(f"record needs {', '.join(missing)} (or --lam CASE for a limited-area mesh)")
    if o.dt == int(o.dt):
        o.dt = int(o.dt)
    if o.hours == int(o.hours):
        o.hours = int(o.hours)
    from .chain import record
    record(o)
    return 0


if __name__ == "__main__":
    sys.exit(main())
