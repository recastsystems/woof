"""The differentiable-engine doors: ``tangent``, ``adjoint-test``, ``sensitivity`` and ``param-grad``.

hex is a differentiable GPU MPAS: a recorded hex forecast (a sealed plan package) is replayed by the window runner,
which carries the exact tangent and adjoint of every command next to the forward, on the GPU. These four doors put
that on the command line. Everything they need is table data (``data/ad-doors.v1.json``): the operator row (mesh,
suite, kernel set, derivative images, runner arguments), the recorded check thresholds, the state fields and the
metrics a user can name. A new row or metric is a table entry, not a code path.

* ``tangent``: the tangent-linear model over N steps from a perturbation of the initial state (a potential-temperature
  bump carried through the hex write-back); writes the tangent state at the window end (``tangent.nc``).
* ``adjoint-test``: one window that prints three numbers against the recorded thresholds: the dot test
  <w, M x> vs <M^T w, x>, the finite-difference check of the tangent and the Taylor check of the adjoint gradient,
  plus the zero-direction pass; then one more window for each further test direction of the table
  (``doors.adjoint-test.directions``: the gradient's own steepest descent, a near-ground moisture draw, a smooth
  draw), each printed with the gradient's ratio to the real model's change per step size. PASS needs every direction
  that the forecast's own roughness lets it resolve inside the bar, and at least two resolved; a direction it cannot
  resolve is printed UNRESOLVED and never counts. PASS / FAIL, exit 0 / 1 (2: refused; 3: the host is too small).
* ``sensitivity``: the adjoint of a scalar forecast metric (the mean of a named field over a box or circle and a
  level range at the window end) to the initial state; writes every field's gradient (``sensitivity.nc``) and a map
  through the Rust renderer. On a row with a gradient line (table ``rows.<row>.gradient_line``) the window runs the
  measured fast route's words; ``--objectives FILE`` runs several objectives in one command on ONE loaded program;
  an objective's terms are kept in the door's cache and taken by the next command that asks for it again.
* ``param-grad``: the gradient of the same metric to the physics constants of the row's parameter table; prints the
  table (dJ/d ln p and dJ/dp per constant).

Every door carries the strict whole-state check of its operator (the self dot and the NaN-poison proof of any
left-out memory: two windows). It belongs to the operator, not to the objective, so it runs once per plan: the first
command on a key of plan, runner, kernel set, card family, steps, row words and test direction runs it and keeps its
verdict in the door's cache; ``tangent``, ``sensitivity`` and ``param-grad`` on the same key take a passing verdict,
print one line saying so and name it in their receipt (``--self-dot`` runs it again); ``adjoint-test`` always runs it.

The runner, its derivative images and the operator rows are located through ``woof.hex.ad.runner`` when that module is
installed (``runner_binary()``, ``asset_root()``), else through ``WOOF_HEX_AD_RUNNER`` / ``WOOF_HEX_AD_ASSETS`` or
``--runner`` / ``--assets``. A plan package comes from ``--plan``, from ``--init`` through ``woof.hex.ad.recorder``
(``record(...)``), or from the row's example plan in the asset root. Every stage runs on the GPU; a missing piece is a
named refusal, never a CPU fallback.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

from .ad import gradmesh as _gm
from .ad import ranks as _ranks
from .ad import startwinds as _startwinds
from . import dot_bar as _dot_bar
from .errors import MpasPortError

TABLE_NAME = "ad-doors.v1.json"
RECORDER_ENV = "WOOF_HEX_AD_RECORDER"
DOORS = ("tangent", "adjoint-test", "sensitivity", "param-grad")
RECEIPT_SCHEMA = "gpuwm-hex.ad-door-receipt/v1"
EARTH_RADIUS_KM = 6371.0


class AdDoorError(MpasPortError):
    """A differentiable-engine door refused, with the reason named."""


class CardMemoryError(AdDoorError):
    """The runner died allocating card memory (CUDA status 2): another process took the memory it was admitted on."""


#: Exit code of a door the host is too small for (0 pass, 1 ran and did not pass, 2 any other refusal).
HOST_TOO_SMALL_EXIT = 3


class HostMemoryError(AdDoorError):
    """The host has less memory available than the row's window needs; the door stops before any download."""
    exit_code = HOST_TOO_SMALL_EXIT


class DiskSpaceError(AdDoorError):
    """The run folder's disk has less free than the window writes there; the door stops before any download."""
    exit_code = HOST_TOO_SMALL_EXIT


# --------------------------------------------------------------------------------------------------------- table
#: Card memory (MiB, by card) and host memory (MiB) held by this user's session on the command's card: counted as
#: free for that command, which attaches to the session or stops it (woof.hex.ad_session.Use._count_held). Without it
#: a later command was refused for the memory its own loaded program held.
_SESSION_HELD: dict = {"card": {}, "host": 0}


def _plan_facts(plan: Path):
    """The plan's kept facts (woof.hex.ad_plan_facts): read from its files once per plan, not once per command."""
    from .ad_plan_facts import plan_facts
    return plan_facts(plan)


def load_table(path: Path | None = None) -> dict:
    path = path or Path(__file__).resolve().parent / "data" / TABLE_NAME
    table = json.loads(Path(path).read_text(encoding="utf-8"))
    if table.get("schema") != "gpuwm-hex.ad-doors/v1":
        raise AdDoorError(f"{path}: schema {table.get('schema')!r} is not gpuwm-hex.ad-doors/v1")
    return table


def _row(table: dict, name: str | None, door: str | None = None) -> tuple[str, dict]:
    """The operator row a door runs: --row, else the door's default row, else the table's."""
    name = name or (table["doors"].get(door, {}).get("default_row") if door else None) or table["default_row"]
    rows = table["rows"]
    if name not in rows:
        raise AdDoorError(f"--row {name!r} has no row in {TABLE_NAME} or the row cache (rows: {', '.join(sorted(rows))}); "
                          "--mesh NAME|GRID makes the row for any mesh")
    if rows[name].get("template_only"):
        raise AdDoorError(f"row {name} is the template rows are made from; name the mesh with --mesh")
    if door is not None and door not in rows[name].get("doors", DOORS):
        # a template row is not a row a user names (--row refuses it: name the mesh), so it is never offered here
        able = [r for r, v in rows.items() if door in v.get("doors", DOORS) and not v.get("template_only")]
        why = rows[name].get("doors_why") or (f"its operator {rows[name]['what'].split(':')[0]!r} has no derivative "
                                               "for it")
        raise AdDoorError(f"row {name} does not run the {door} door ({why}); rows that do: "
                          f"{', '.join(able) or 'none'}")
    return name, rows[name]


def door_steps(row: dict, door: str | None, args: argparse.Namespace | None = None) -> int:
    """The window a door runs when --steps is not given: the row's ``door_steps.<door>``, else its ``default_steps``.

    THE BREAKAGE THIS PREVENTS: the shipped 120 km example's adjoint-test ran the row's 3-step (36 minute) window, over
    which the real model's change per unit step moves 1 to 6 per cent between neighbouring step sizes, more than the
    0.01 bar: a working install printed NOT PASSED (UNRESOLVED), 1 of 8 directions resolved, and exited 1 (one 96 GB card,
    15 Nov 2026). The test window is the door's own; tangent, sensitivity and param-grad keep the row's."""
    given = getattr(args, "steps", None) if args is not None else None
    if given:
        return int(given)
    own = (row.get("door_steps") or {}).get(door) if door else None
    return int(own or row["default_steps"])


def adjoint_test_taylor_h(spec_at: dict, row: dict) -> list[float]:
    """The step sizes adjoint-test reruns the real model at: the row's ``adjoint_test_taylor_h``, else the table's.

    THE BREAKAGE THIS PREVENTS: on the 120 km example the table's ladder (1, 0.3, 0.1, 0.03, 0.01) puts neighbouring
    sizes a factor of three apart, and the real model's roughness moves the change per unit step by about 1 per cent
    over such a run of three: a direction that agreed with the gradient to 1.0e-04 at a size that counts was still
    printed UNRESOLVED (1.0e-04 plus a spread of 1.04e-02 over the 0.01 bar; 15 Nov 2026). On the 3 km
    example the table's smallest size moves the measure by only 5.4 single-precision spacings of it, so the descent
    direction read 1.0044 there and the command printed UNRESOLVED on an install whose bump read inside (16 Nov
    2026); that row's ladder stops at 0.03."""
    return [float(h) for h in (row.get("adjoint_test_taylor_h") or spec_at["taylor_h"])]


# ------------------------------------------------------------------------------------------------------- runtime
def _hexad():
    """woof.hex.hexad (the wheel's runner locator), when this install has it."""
    try:
        from . import hexad
    except ImportError:
        return None
    return hexad


def _runner_api():
    """woof.hex.ad.assets (the operator rows' files), when this install has it."""
    try:
        from .ad import assets
    except ImportError:
        return None
    return assets


def resolve_runner(explicit: Path | None) -> Path:
    """The window runner binary, or a named refusal."""
    candidates: list[tuple[str, Path | None]] = [("--runner", explicit)]
    env = os.environ.get("WOOF_HEX_AD_RUNNER")
    candidates.append(("WOOF_HEX_AD_RUNNER", Path(env) if env else None))
    hexad = _hexad()
    if hexad is not None:
        try:
            candidates.append(("the installed runner (woof.hex.hexad)", Path(hexad.binary())))
        except hexad.HexadNotInstalled:
            pass
    found = shutil.which("hexad")
    candidates.append(("hexad on PATH", Path(found) if found else None))
    for source, path in candidates:
        if path is not None:
            if not path.is_file() or not os.access(path, os.X_OK):
                raise AdDoorError(f"{source} names {path}, which is not an executable runner")
            return path
    raise AdDoorError("no window runner: this install has no ad-binder (it ships in the Linux wheel's woof/hex/libexec); "
                      "name a built one with --runner or WOOF_HEX_AD_RUNNER")


def resolve_assets(explicit: Path | None, want: list[str] | None = None) -> Path:
    """The asset root that holds the operator rows (kernel sets, derivative images) and example plans. WANT (paths the
    door reads under the root) limits a first fetch to the pinned files that hold them (woof.hex.ad.assets.wanted)."""
    if explicit is not None:
        root = explicit
    else:
        api = _runner_api()
        root = None
        if os.environ.get("WOOF_HEX_AD_ASSETS"):
            root = Path(os.environ["WOOF_HEX_AD_ASSETS"])
        elif api is not None and hasattr(api, "asset_root"):
            try:   # fetched once from the release and unpacked under the hex cache (data/ad-assets.json pins)
                try:
                    root = Path(api.asset_root(want=want))
                except TypeError:   # an asset module that fetches every pinned file
                    root = Path(api.asset_root())
            except (api.AdAssetError, OSError, ValueError) as error:
                raise AdDoorError(f"differentiable-engine assets: {error}") from error
        elif (Path(__file__).resolve().parent / "data" / "ad" / "rows").is_dir():
            root = Path(__file__).resolve().parent / "data" / "ad"
    if root is None:
        raise AdDoorError("no differentiable-engine asset root (the operator rows' kernel sets and images): name it "
                          "with --assets or WOOF_HEX_AD_ASSETS")
    if not root.is_dir():
        raise AdDoorError(f"asset root {root} is not a directory")
    return root


def example_reads(row: dict, args: argparse.Namespace, line_on: bool = True) -> list[str]:
    """The files a door reads under the asset root that a pinned file other than the operator-row files may hold: the
    row's mesh files and every other row asset (a kernel set packed on its own), its gradient line's optional assets
    (an image the window uses when present), the images of the boundary-layer route the command will run, and its
    example plan only when no plan is given or recorded. The
    operator-row files are always fetched (woof.hex.ad.assets.wanted). THE BREAKAGE THIS PREVENTS: an optional image
    pinned in its own download file was never fetched, because the door asked only for mesh and plan paths, so every
    gradient-line window ran without it and only the receipt said so."""
    if line_on:   # a route's own kernel set or images stand in the row's place (route_row)
        row, _ = route_row(row, args)
    reads = [row["assets"].get("grid"), row["assets"].get("oro")]
    reads += [v for k, v in sorted(row["assets"].items()) if k not in ("grid", "oro")]
    line = row.get("gradient_line") or {}
    reads += [item.get("path") for _, item in sorted((line.get("optional_assets") or {}).items())]
    _, route, _ = boundary_layer_route(row, getattr(args, "boundary_layer", None))
    reads += [item.get("path") for _, item in sorted((route.get("images") or {}).items())]
    if line_on:
        for _, _, entry in line_routes(row, args):
            reads += [item.get("path") for _, item in sorted(((entry.get("scheme_bundle") or {}).get("images")
                                                               or {}).items())]
    if getattr(args, "plan", None) is None and getattr(args, "init", None) is None:
        reads.append(row.get("example_plan"))
    if str(getattr(args, "init", None)) == EXAMPLE_INIT:
        reads.append(row.get("example_case"))
    return [str(r) for r in reads if r and not os.path.isabs(str(r))]


#: --init example: record from the row's own example start (its example_case), as a user's own start would be
EXAMPLE_INIT = "example"


def row_assets(root: Path, row: dict, overrides: dict | None = None) -> dict[str, Path]:
    assets = {key: root / rel for key, rel in row["assets"].items()}
    return assets


#: --tempo-image: the taped TEMPO derivative images a window may run in place of the row's tempo_cubin, by name, each
#: pinned by sha256. shipped (the default) is the row's own file. f08 is the same source as the shipped split2.cubin
#: (tree/ad/phys/tempo/f08img/split2.main.cu, sha256 a15591d1..., which rebuilds the shipped ce8d3a67... byte for byte
#: with NVRTC 13.4.92) with coltest/f08_subgrid.py applied: tempo_warp_steps' PBL subgrid radii (NOAA's hrrrv5 deck,
#: config_tempo_ml_for_bl_nc) in the record entry, bl (qc_bl, cldfrac_bl) entering as constants. Built by f08img/make.sh.
#: Its file is WOOF_HEX_AD_TEMPO_F08 (a path), else <asset root>/latest/aux/split2-f08.cubin.
TEMPO_IMAGES = {"f08": "358fcde2f05c0079be3191afbb4847987dc114297f114e6551ef20753aed6a41"}
TEMPO_F08_ENV = "WOOF_HEX_AD_TEMPO_F08"


def tempo_image_word(ctx, args: argparse.Namespace) -> None:
    """--tempo-image NAME: point the row's tempo_cubin at the named, pinned image (shipped: unchanged).

    THE BREAKAGE THIS PREVENTS: every window on NOAA's hrrrv5 deck stopped at its first radiation step ("ml_bl set, bl
    present, radii launch: ... no g_tempo_bl") because the shipped taped image predates F08; the 30 km plan's window
    could not pass 27.5 minutes (18 Nov). The shipped image stays the default, so the sealed release's numbers stay."""
    name = getattr(args, "tempo_image", None) or "shipped"
    if name == "shipped":
        return
    want = TEMPO_IMAGES[name]
    if "tempo_cubin" not in ctx.assets:
        raise AdDoorError(f"--tempo-image {name}: row {ctx.row_name} names no taped TEMPO image (tempo_cubin)")
    env = os.environ.get(TEMPO_F08_ENV)
    path = Path(env) if env else Path(ctx.root) / "latest" / "aux" / "split2-f08.cubin"
    if not path.is_file():
        raise AdDoorError(f"--tempo-image {name}: {path} is missing (set {TEMPO_F08_ENV} or build it with "
                          "tree/ad/phys/tempo/f08img/make.sh). Nothing ran")
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != want:
        raise AdDoorError(f"--tempo-image {name}: {path} has sha256 {got}, not the pinned {want}. Nothing ran")
    ctx.assets = {**ctx.assets, "tempo_cubin": path}
    print(f"[woof hex] TEMPO taped image: {name} ({path}, sha256 {got[:16]}): the PBL subgrid-radii path is "
          "differentiated, bl held as constants", flush=True)


def no_plan_words(row_name: str) -> str:
    return (f"row {row_name} ships no example plan: pass --init INIT (the door records a plan package for this mesh) "
            "or --plan PACKAGE")


def refuse_steps_beyond_example(row_name: str, row: dict, steps: int, args: argparse.Namespace) -> None:
    """Refuse, before any card, session or download, a step count the row's example plan does not cover (the table's
    ``example_plan_steps``: the longest window the example is said to cover). A plan the user gives or records is
    the runner's to judge.

    THE BREAKAGE THIS PREVENTS: ``adjoint-test --steps 12`` on the 120 km example (recorded for 4 steps, a 3-step
    window) downloaded the example (2.4 GB) and then stopped in the runner with "physics step count or single-stream
    plan is invalid" (a fresh install of a 0.6.0 build). And ``--steps 0`` ran the row's default window."""
    if getattr(args, "steps", None) is not None and int(args.steps) < 1:
        raise AdDoorError(f"--steps {args.steps}: a window has at least 1 step. Nothing was downloaded")
    if getattr(args, "plan", None) is not None or getattr(args, "init", None) is not None:
        return
    most = row.get("example_plan_steps")
    if not row.get("example_plan") or most is None or 1 <= steps <= int(most):
        return
    dt = float(row.get("dt_s") or 0.0)
    span = f" ({int(most) * dt / 60:g} min of {dt:g} s steps)" if dt else ""
    raise AdDoorError(f"--steps {steps}: row {row_name}'s example plan covers windows of 1 to {int(most)} steps{span}; "
                      f"pass --steps from 1 to {int(most)}, or record a plan of your own for a longer window "
                      "(--init INIT records the steps the window needs). Nothing was downloaded")


def refuse_row_not_in_pinned_assets(row_name: str, row: dict, keys: set[str], explicit: Path | None) -> None:
    """Refuse, before anything is fetched, a row whose files the pinned downloads do not hold.

    THE BREAKAGE THIS PREVENTS: a fresh install asked for the 120 km row (--mesh x1.40962) fetched the 480 km row and
    its example plan (942 MB), which hold none of that row's files, and only then refused; with --init it went on to
    the recorder first (2026-11-11). The pins say which asset-root paths their files hold (data/ad-assets.json,
    "holds"). An asset root the user names (--assets, WOOF_HEX_AD_ASSETS) is not judged here: the window names a
    file it lacks."""
    if explicit is not None or os.environ.get("WOOF_HEX_AD_ASSETS"):
        return
    api = _runner_api()
    if api is None or not hasattr(api, "pinned_holds"):
        return
    try:
        holds = api.pinned_holds()
    except (api.AdAssetError, OSError, ValueError):
        return   # resolve_assets names a bad pin file
    if holds is None:
        return
    reads = [str(rel) for key, rel in row["assets"].items() if key in keys]
    reads.append(str((row.get("self_dot_dead") or {}).get("list") or ""))
    absent = sorted({rel for rel in reads
                     if rel and not os.path.isabs(rel) and not any(rel.startswith(h) for h in holds)})
    if absent:
        more = f" and {len(absent) - 1} more" if len(absent) > 1 else ""
        raise AdDoorError(f"row {row_name} reads {absent[0]}{more} from the asset root, and the files this install "
                          f"downloads do not hold {'them' if more else 'it'} (they hold {', '.join(holds)}), so "
                          "nothing was downloaded. Name an asset root that holds the row's files with --assets or "
                          "WOOF_HEX_AD_ASSETS")


def split_row(row: dict, split: dict) -> dict:
    """The row as its rank plans run it (table rank_splits): the split's assets in place of the row's (an
    owner-reduced kernel set), its runner words added, the variables that name the one-card package's own
    allocations dropped, and its row overrides (a left-out list of the one-card package does not describe a rank)."""
    out = dict(row, **split["row_overrides"])
    out["assets"] = {**row["assets"], **split["assets"]}
    out["window_args"] = drop_window_args(list(row["window_args"]), split.get("window_args_drop") or [])         + list(split["window_args"])
    drop = set(split["env_drop"])
    out["window_env"] = {k: v for k, v in (row.get("window_env") or {}).items() if k not in drop}
    out["asset_env"] = {k: v for k, v in (row.get("asset_env") or {}).items() if k not in drop}
    return out


def drop_window_args(argv: list[str], names: list[str]) -> list[str]:
    """ARGV without each option NAMES lists (an option and its value). THE BREAKAGE THIS PREVENTS: a made row's class
    sizes the host for one process (60 km class --host-budget-gb 320); four ranks on one host each kept it, 1,280 GB
    asked of the host together, and the rank launcher's own per-rank budget (0.8 x MemAvailable / ranks) was never
    given because the option was already there."""
    out, it = [], iter(argv)
    for a in it:
        if a in names:
            next(it, None)
            continue
        out.append(a)
    return out


def refuse_rank_pkg_not_pinned(row_name: str, pkg: str, explicit: Path | None) -> None:
    """Refuse, before anything is fetched, a rank plan package the pinned downloads do not hold (an asset root the
    user names is not judged here: the door names a rank plan it lacks). THE BREAKAGE THIS PREVENTS: a split door on
    a fresh install would fetch the one-card row and example plan (about 940 MB) and only then find no rank plans."""
    if explicit is not None or os.environ.get("WOOF_HEX_AD_ASSETS"):
        return
    api = _runner_api()
    if api is None or not hasattr(api, "pinned_holds"):
        return
    try:
        holds = api.pinned_holds()
    except (api.AdAssetError, OSError, ValueError):
        return
    if holds is None or any(str(pkg).startswith(h) for h in holds):
        return
    raise AdDoorError(f"row {row_name}: its rank plan package {pkg} is not in the files this install downloads (they "
                      f"hold {', '.join(holds)}), so nothing was downloaded. Give a rank plan package split from your "
                      "--plan with --rank-plans PREFIX, name an asset root that holds it with --assets or "
                      "WOOF_HEX_AD_ASSETS, or run on one card (--card N)")


def refuse_init_without_recorder(args: argparse.Namespace) -> None:
    """Refuse --init, before anything is fetched, on an install that cannot record.

    THE BREAKAGE THIS PREVENTS: an install with the quick-start extra only (no CUPTI) asked for --init on a card of the
    right class downloaded the row and the plan (942 MB) and only then stopped in the recorder ("CUPTI is not
    installed"; a fresh install of the 0.6.0 wheel, 2026-11-11). What the recorder lacks is known without any asset.
    """
    if getattr(args, "init", None) is None:
        return
    if getattr(args, "plan", None) is not None:
        raise AdDoorError("--plan and --init are alternatives: a plan package is either given or recorded from an init")
    if os.environ.get(RECORDER_ENV):   # a recorder named by command is the user's; it is not judged here
        return
    import contextlib
    import importlib.util
    import io
    lack = None
    if importlib.util.find_spec("woof.hex.ad.record") is None:
        lack = "woof.hex.ad.record is not in this install"
    else:
        from .ad.record import env as _record_env
        said = io.StringIO()
        try:
            with contextlib.redirect_stderr(said):
                _record_env.cupti()   # the recorder's own lookup: the CUDA toolkit wheel and CUPTI beside hexcore
        except SystemExit:
            lack = said.getvalue().strip().replace("DEFECT: woof.hex.ad.record: ", "") or "CUPTI is not installed"
    if lack:
        raise AdDoorError(f"--init needs the recorder, and this install cannot run it: {lack}. Install it with "
                          f"pip install 'recast-woof[ad]', or name a recorder command in {RECORDER_ENV}, or pass a "
                          "recorded plan package with --plan. Nothing was downloaded")


def resolve_plan(args: argparse.Namespace, root: Path, row_name: str, row: dict) -> tuple[Path, str]:
    """(plan package directory, how it was chosen)."""
    if args.plan is not None and args.init is not None:
        raise AdDoorError("--plan and --init are alternatives: a plan package is either given or recorded from an init")
    if args.plan is not None:
        plan = Path(args.plan)
        how = "given (--plan)"
    elif args.init is not None:
        plan = record_plan(args, root, row)
        how = f"recorded from {args.init} (python -m woof.hex.ad.record record)"
    elif row.get("example_plan"):
        plan = root / row["example_plan"]
        how = f"the row's example plan ({row['example_plan']})"
    else:
        raise AdDoorError(no_plan_words(row_name))
    if not (plan / "plan.json").is_file():
        raise AdDoorError(f"plan package {plan} has no plan.json")
    return Path(os.path.abspath(plan)), how   # not resolve(): the asset root names a plan by its link name


# The door records in strict numerics, and says so (the recorder log and record-receipt.json carry the label). THE
# BREAKAGE THIS PREVENTS: the recorder's own default is fast numerics; a 120 km door --init recorded fast and the pair
# could not seal (stage-2 residual allocation_first_use_not_whole_write 739, device_to_host_consumer 2,
# upload_needs_live_rule 1; a 120 km run, 2026-11-11), while every sealed 120 km plan (pk120e4) is a strict
# recording. A recorder named in WOOF_HEX_AD_RECORDER gets the label as {numerics}.
RECORD_NUMERICS = "strict"


def record_ranks_of(args: argparse.Namespace) -> int:
    """--record-ranks N (default 1). N > 1 needs --init: rank plans are made by recording, not from a given plan."""
    n = int(getattr(args, "record_ranks", None) or 1)
    if n < 1:
        raise AdDoorError(f"--record-ranks {n}: a count of rank threads, 1 or more")
    if n > 1 and getattr(args, "init", None) is None:
        raise AdDoorError(f"--record-ranks {n} records rank plans from a start: pass --init INIT")
    return n


def record_plan(args: argparse.Namespace, root: Path, row: dict) -> Path:
    """Record and seal a plan package from --init with the recorder (``python -m woof.hex.ad.record record``).
    With --record-ranks N > 1 the recorder also cuts it into OUT/plan-r<k>; the door then prints them and stops."""
    import importlib.util
    import shlex
    ranks = record_ranks_of(args)
    if str(args.init) == EXAMPLE_INIT:   # the row's example start, from the downloaded case
        if not row.get("example_case"):
            raise AdDoorError("--init example: this row ships no example start; pass --init INIT (and --case for a "
                              "limited-area mesh)")
        from .ad.record import lam as _lam
        try:
            example = _lam.read_case(root / row["example_case"])
        except _lam.CaseError as error:
            raise AdDoorError(f"--init example: {error}") from error
        args.init, args.case = Path(example["init"]), Path(example["source"])
    assets = row_assets(root, row)
    grid = Path(args.grid) if args.grid else assets.get("grid")
    oro = Path(args.oro) if getattr(args, "oro", None) else assets.get("oro")
    case = limited_area_case(args, row)
    if case is not None and (oro is None or not Path(oro).is_file()) and case.get("oro"):
        oro = Path(case["oro"])
    for what, path in (("grid", grid), ("orographic statistics", oro)):
        if path is None or not Path(path).is_file():
            raise AdDoorError(f"--init: the row's {what} file {path} is missing")
    steps = int(args.steps or row["default_steps"])
    # The recording runs one step past the window. Boundary s is the start of step s, so the window's end is
    # boundary steps + 1, and only a map that records it names the window-end field addresses (what the strict
    # self dot weighs and what an AD_DOT_DEAD list is built from). pk120e4 records 4 steps for its 3-step window.
    recorded = steps + 1
    # The hours go to the recorder at full precision. THE BREAKAGE THIS PREVENTS: six significant digits turned the
    # 480 km row's default (4 recorded steps of 1200 s, 1.3333... h) into "1.33333", and the recorder refused it
    # ("--hours 1.33333 is not a whole number of 1200 s steps"; a fresh install, tangent --init, 2026-11-11).
    hours = recorded * row["dt_s"] / 3600.0
    out = Path(args.out).resolve() / "plan"
    fill = {"mesh": row["mesh"], "suite": row["suite"], "init": str(Path(args.init).resolve()), "grid": str(grid),
            "oro": str(oro), "dt": f"{row['dt_s']:g}", "hours": repr(float(hours)), "steps": str(recorded),
            "out": str(out),
            "card": card_of(args), "numerics": RECORD_NUMERICS, "lam": case["source"] if case else "",
            "route": getattr(args, "record_route", None) or "derivative"}
    recorder = os.environ.get(RECORDER_ENV)
    if recorder:   # another recorder of the same contract (plan.json + alloc-map.json in {out}), named by command
        argv = [part.format(**fill) for part in shlex.split(recorder)]
    else:
        if importlib.util.find_spec("woof.hex.ad.record") is None:
            raise AdDoorError("--init needs the recorder (woof.hex.ad.record), which is not in this install "
                              f"(pip install 'recast-woof[ad]'), or a recorder command in {RECORDER_ENV}; or pass a "
                              "recorded plan package with --plan")
        argv = [sys.executable, "-m", "woof.hex.ad.record", "record",
                # a limited-area forecast runs on the cull's registered row (the case's mesh-rows.json), not the
                # door's name for the grid file
                "--mesh", case["row"] if case else row["mesh"], "--suite", row["suite"],
                "--init", fill["init"], "--grid", str(grid), "--oro", str(oro),
                "--dt", fill["dt"], "--hours", fill["hours"], "--out", str(out), "--card", fill["card"],
                "--numerics", RECORD_NUMERICS, "--serial", "--runner", str(resolve_runner(args.runner)),
                "--route", fill["route"]]
        if case is not None:
            argv += ["--lam", case["source"]]
        if ranks > 1:
            argv += ["--ranks", str(ranks)]
    if (out / "plan.json").is_file() and (out / "record-receipt.json").is_file() and ranks == 1:
        # a finished recording of this door's output directory is reused, never re-recorded, if it reaches the end
        if recorded_end_boundary(out) >= recorded:
            return out
        raise AdDoorError(f"the plan recorded in {out} ends before boundary {recorded} (the end of a {steps}-step "
                          "window, made by a door before the recording ran one step past the window): pass a new "
                          "--out, or --steps no larger than its recorded steps less one")
    from . import ad_radiation_seed as _radseed
    argv += forecast_words(_radseed.record_words(args), args)   # --radiation-seed, --cumulus: the forecast's words
    log = Path(args.out).resolve() / "record.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    # the forward kernel forms the row's kernel set has derivative rows for (hex picks hoisted/chunked/element forms
    # and MYNN lane widths by mesh size; a form with no row is refused in the window, i.e. a wrong gradient)
    env = {**os.environ, **{k: str(v) for k, v in (row.get("record_env") or {}).items()}}
    if ranks > 1:   # a recorder named in WOOF_HEX_AD_RECORDER reads the count from its environment
        env["WOOF_HEX_RECORD_RANKS"] = str(ranks)
    with open(log, "wb") as stream:
        rc = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, env=env).returncode
    if rc != 0:
        lines = log.read_text(errors="replace").strip().splitlines()
        defect = [x for x in lines if "DEFECT" in x][-1:] or lines[-1:]
        raise AdDoorError(f"recording from {args.init} failed (rc {rc}): {defect[0][:300] if defect else ''} "
                          f"(log {log})")
    if ranks > 1:
        plans = [Path(f"{out}-r{k}") for k in range(ranks)]
        lacking = [str(p) for p in plans if not (p / "plan.json").is_file()]
        if lacking:
            raise AdDoorError(f"--record-ranks {ranks}: the recording sealed but {len(lacking)} rank plan(s) are "
                              f"missing ({lacking[0]}); the split's log is in {out}.work/split.log")
        print(f"woof hex {getattr(args, 'command', None) or 'door'}: recorded {row['mesh']} as {ranks} rank threads "
              f"and split it: rank plans {out}-r0 .. {out}-r{ranks - 1} (the one-card plan {out}). A window split "
              f"over {ranks} cards replays them: ad-binder ranks --ranks {ranks} --cards A,B,.. --pkg {out} -- "
              "window ...", flush=True)
        raise SystemExit(0)
    return out


#: --cumulus: the cumulus schemes a recording from --init may run (MPAS config_convection_scheme values the hrrrv5
#: driver integrates). Off by default: the plan runs the suite's own deck for the mesh.
CUMULUS_SCHEMES = ("cu_ntiedtke", "cu_grell_freitas")


def forecast_words(words: list[str], args: argparse.Namespace) -> list[str]:
    """The recording's forecast words (after ``--``): the radiation seed's, plus the cumulus scheme --cumulus names.
    No --cumulus: the words unchanged, so every recording is the one it was before."""
    scheme = getattr(args, "cumulus", None)
    if not scheme:
        return list(words)
    return (list(words) or ["--"]) + ["--physics", f"config_convection_scheme={scheme}"]


def plan_scheme_holds(table: dict, row: dict, plan: Path) -> dict:
    """ROW with every scheme hold by name (``derivative_holds`` of any row of TABLE) whose launch kinds PLAN runs.
    THE BREAKAGE THIS PREVENTS: the New Tiedtke hold lived on the 12 km row only, so a plan recorded with cumulus on
    any made row (a 30 km cut, 18 Nov 2026) ran its 26 kinds as "no derivative row and no rule" and the window read
    invalid, while the recorder had sealed the plan as covered (coverage.held_by_name reads every row). A plan that
    runs none of the held kinds gets ROW back unchanged."""
    holds = dict(row.get("derivative_holds") or {})
    try:
        from .ad.record import coverage
        kinds = set(coverage.kinds_of(Path(plan) / "plan.json"))
    except Exception:  # noqa: BLE001 - an install without the recorder, or a plan it cannot read: the row's own holds
        return row
    added = False
    for other in (table.get("rows") or {}).values():
        for hold, spec in (other.get("derivative_holds") or {}).items():
            if hold not in holds and kinds & set(spec.get("kinds", ())):
                holds[hold] = spec
                added = True
    return {**row, "derivative_holds": holds} if added else row


def limited_area_case(args: argparse.Namespace, row: dict) -> dict | None:
    """The limited-area case of a recording from --init (woof.hex.ad.record.lam), or None for a global row.

    The mesh kind is data: the row made for a limited-area grid says so (mesh_kind), and its recording takes the
    regional door's prepared case: --case, else the regional work directory --init lies in. THE BREAKAGE THIS
    PREVENTS: sensitivity --mesh <limited-area grid> --init <its start> handed the recorder a grid and an init only,
    so the forecast had no boundaries, static file or contract and no regional plan could be recorded from an
    install (2026-11-13)."""
    limited = (row.get("mesh_kind") or (row.get("generated") or {}).get("mesh_kind")) == "limited-area"
    given = getattr(args, "case", None)
    if not limited and given is None:
        return None
    from .ad.record import lam as _lam
    try:
        return _lam.read_case(given if given is not None else args.init)
    except _lam.CaseError as error:
        raise AdDoorError(f"{error}. A limited-area mesh records from the regional door's prepared case: pass --case "
                          "WORKDIR (woof hex regional ... --stage mesh,intermediate,init,lbc,contract writes one), "
                          "or --init inside such a directory") from error


def forwarded_physics_mode(words: list | tuple | None) -> str | None:
    """The physics mode a list of forecast words ends on (``--physics-mode X`` or ``--physics-mode=X``; the last one
    wins, as the forecast door's own parser reads it), or None when no word names one. A word with no value gives
    ``""``, which is no mode."""
    mode = None
    words = [str(w) for w in (words or [])]
    for i, word in enumerate(words):
        if word == "--physics-mode":
            mode = words[i + 1] if i + 1 < len(words) else ""
        elif word.startswith("--physics-mode="):
            mode = word.split("=", 1)[1]
    return mode


def plan_physics_mode(plan: Path) -> str | None:
    """The physics mode a plan package was recorded in, read from its record-receipt.json: the mode its forecast words
    end on, else the receipt's own ``physics_mode``; None for a package that says neither (a package of another
    recorder). THE BREAKAGE THIS PREVENTS: the recorder appended the words after ``--`` to a forecast command that
    already said ``--physics-mode exact``, so ``record ... -- --physics-mode correct`` recorded the correct-mode
    forecast while its receipt said ``"physics_mode": "exact"``; the window then ran the exact-mode derivative rows
    on it, and the tangent-against-adjoint check passes on such a pair whatever the forward was."""
    try:
        receipt = json.loads((plan / "record-receipt.json").read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(receipt, dict):
        return None
    said = forwarded_physics_mode(receipt.get("forecast_args"))
    return said if said is not None else receipt.get("physics_mode")


def physics_mode_block(table: dict) -> dict:
    """The table's physics_mode block (the modes the derivative rows follow, the door's line, the refusals)."""
    return table.get("physics_mode") or {"linearised": ["exact"], "default": "exact", "line": None, "refused": {}}


def refuse_physics_mode(args: argparse.Namespace, door: str, table: dict) -> str:
    """The physics mode this command's linear model is of; a mode the build has no derivative rows for is refused
    here, before any card, runner or download.

    THE BREAKAGE THIS PREVENTS: ``woof hex forecast`` runs correct mode by default and every plan is recorded in
    exact mode, so a user who wants the gradient of the forecast they ran gets the exact-mode rows and no word about
    it. Where the two modes differ inside a scheme (the corrected RRFS cloud, boundary-layer and surface-layer
    schemes), the exact-mode rows on a correct-mode forecast give a gradient that is wrong in those columns while the
    tangent-against-adjoint check still passes: both sides are the same exact-mode rows."""
    block = physics_mode_block(table)
    asked = getattr(args, "physics_mode", None)
    mode = asked or block.get("default") or "exact"
    if mode not in block["linearised"]:
        why = (block.get("refused") or {}).get(mode) or (
            f"this build's tangent and adjoint are of {' and '.join(block['linearised'])} mode only")
        raise AdDoorError(f"{door}: --physics-mode {mode}: {why}")
    return mode


def refuse_plan_physics_mode(plan: Path, mode: str, table: dict) -> str | None:
    """A plan package recorded in another physics mode than the command's linear model is of is refused by name."""
    recorded = plan_physics_mode(plan)
    block = physics_mode_block(table)
    if recorded is not None and (recorded != mode or recorded not in block["linearised"]):
        raise AdDoorError(f"plan {plan} was recorded in physics mode {recorded or '(none named)'!r} (the forecast "
                          f"words of its record-receipt.json) and this command's tangent and adjoint are of {mode} "
                          f"mode: the derivative rows are of the {mode}-mode arithmetic, so they are not the "
                          "derivative of that recording. Record the plan without a --physics-mode word after --, or "
                          "with --init INIT through this door")
    return recorded


def plan_recording(plan: Path) -> dict:
    """What a plan package says about its own recording: the numerics label, whether radiation and land were recorded
    live, and the operator manifest it carries (name and sha256), read from record-receipt.json and manifest.json.
    A recording holds no derivative setting: radiation is recorded live by both recorders, and it is the row that
    holds radiation in the derivative (its refused kernel kinds). THE BREAKAGE THIS PREVENTS: the 120 km --init seal
    failure was put down to "the published recording was made with radiation frozen" (doors120, 2026-11-11), from
    the manifest's name; the receipt now states what was recorded and which manifest, so the two can be told apart."""
    said: dict = {"manifest": None}
    try:
        receipt = json.loads((plan / "record-receipt.json").read_text())
    except (OSError, ValueError):
        receipt = {}
    said["recorder_schema"] = receipt.get("schema")
    said["numerics"] = receipt.get("numerics")
    if "hold_slots" in receipt:   # the door's recorder (woof.hex.ad.record)
        held = receipt.get("hold_slots") or ""
        said["held_physics_slots"] = held
        said["radiation_and_land"] = "recorded live" if not held else f"held slots: {held}"
        said["seal_residual"] = receipt.get("residual")
    manifest = plan / "manifest.json"
    if manifest.is_file():
        try:
            name = json.loads(manifest.read_text()).get("name")
        except ValueError:
            name = None
        said["manifest"] = {"file": "manifest.json", "name": name, "sha256": _sha256_file(manifest)}
    return said


def recorded_end_boundary(plan: Path) -> int:
    """The last boundary a recorded plan's allocation map names (0 when it names none)."""
    try:
        doc = json.loads((plan / "alloc-map.json").read_text())
    except (OSError, ValueError):
        return 0
    keys = [int(k) for k in (doc.get("boundaries") or {}) if str(k).isdigit()]
    return max(keys, default=0)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


# -------------------------------------------------------------------------------------------------- field map
class FieldMap:
    """Where each named state field lives in a plan: its canonical allocation range and every byte-equal copy.

    Reads ``alloc-map.json`` in either recorded schema: the runner library's (``canonical`` + ``copies`` with byte
    offsets, ``scalars.species_slot``) or the recorder's (``allocation`` + ``copies`` ids at the same offset).

    A copy may hold fewer columns than its root: on a limited-area mesh the dycore's arrays carry MPAS's padding
    column (nCells + 1, nEdges + 1) and the physics pool's copies do not. Each copy keeps its own shape
    (``copy_shapes``), and every field is handed to the host on the mesh's own columns (``shape``); ``read_shape``
    is the shape of the copy the door reads.
    """

    def __init__(self, doc: dict, plan_sha256: str, source: Path, aliases: dict | None = None,
                 mesh: dict | None = None, plan: Path | None = None):
        self.mesh = mesh or {}
        self.found: list[dict] = []
        if doc.get("plan_sha256") != plan_sha256:
            raise AdDoorError(f"{source} belongs to plan {str(doc.get('plan_sha256'))[:12]}, not this plan "
                              f"({plan_sha256[:12]}); allocation numbers belong to one recording")
        self.doc, self.source = doc, source
        self.fields: dict[str, dict] = {}
        for name, entry in doc["fields"].items():
            if "canonical" in entry:
                canonical = entry["canonical"]
                copies = [(int(c["alloc"]), int(c["offset"])) for c in entry.get("copies", [])]
                self.fields[name] = {"alloc": int(canonical["alloc"]), "offset": int(canonical["offset"]),
                                     "shape": [int(x) for x in canonical["shape"]], "copies": copies or
                                     [(int(canonical["alloc"]), int(canonical["offset"]))],
                                     "species_slot": entry.get("species_slot"), "slot_words": entry.get("slot_words"),
                                     "copy_shapes": {}}
            elif "allocation" in entry:
                offset = int(entry.get("offset", 0))
                ids = [int(entry["allocation"])] + [int(c) for c in entry.get("copies", [])]
                # The recorder names the forward's fields.* / saved_fields.* view (the dycore's copy); the window-end
                # state is the physics pool's copy, which the forward allocates after them: the highest id.
                # A field the recorder found no pool copy of is mapped by its initial copy alone: that allocation
                # is the start state (right for a perturbation and for an initial-state gradient) and is never
                # written again, so it is NOT the field at the window end (window_end False).
                self.fields[name] = {"alloc": max(ids), "offset": offset,
                                     "shape": [int(x) for x in entry["shape"]],
                                     "copies": [(i, offset) for i in dict.fromkeys(ids)],
                                     "species_slot": None, "slot_words": None,
                                     "window_end": len(set(ids)) > 1,
                                     "copy_shapes": {int(k): [int(x) for x in v]
                                                     for k, v in (entry.get("copy_shapes") or {}).items()}}
        names = {k: k for k in self.fields}   # the doors' key -> the map's own name of the field
        if doc.get("schema") == "ad-record.alloc-map/v1":
            # the recorder's map names its fields by the forward's saved-field names (fields.rho is rho_zz in hex)
            recorded = dict(self.fields)
            self.fields = {k: v for k, v in recorded.items() if k in ("theta_m", "exner", "rho_theta", "scalars")}
            names = {k: k for k in self.fields}
            for ours, theirs in (aliases or {}).items():
                if ours != "what" and theirs in recorded:
                    self.fields[ours] = recorded[theirs]
                    names[ours] = theirs
        # Where each field is at every recorded step boundary (the recorder's boundaries[s][field]; s = 1 is t0, so
        # an N-step window ends at boundary N + 1). THE BREAKAGE THIS PREVENTS: the physics pool's copy (the highest
        # id above) is filled from the state at the START of each step, and made current only by the last recorded
        # step's closing copy. A window shorter than its recording (every door window: the door records one step
        # past it) therefore read the state one step before the window end: J of the 60 km map at 28 steps for a
        # 29-step window (302.22687187 K, = the 28-step dump to 3e-08; the 29-step dump gives 302.20619569 K;
        # f52/explain/m60c/RESULT.md (d)), and the 120 km door's 3-step window read at +24 min, not +36 (the plan's
        # step_02..step_04 each open with a device_copy of that boundary's theta_m into the pool copy).
        self.boundaries: dict[int, dict[str, tuple[int, int]]] = {}
        for key, places in (doc.get("boundaries") or {}).items():
            if not str(key).isdigit() or not isinstance(places, dict):
                continue
            here = {}
            for ours, theirs in names.items():
                place = places.get(theirs)
                if isinstance(place, dict) and "allocation" in place:
                    here[ours] = (int(place["allocation"]), int(place.get("offset", 0)))
            self.boundaries[int(key)] = here
        # the window's length in steps: set by the door before any field is read at the window end
        self.window_steps: int | None = None
        species = doc.get("species_order")
        if species and "scalars" in self.fields and not self.fields["scalars"]["species_slot"]:
            shape = self.fields["scalars"]["shape"]
            self.fields["scalars"]["species_slot"] = {name: i for i, name in enumerate(species)}
            self.fields["scalars"]["slot_words"] = int(math.prod(shape[1:]))
        if plan is not None and self.mesh:
            self._find_padded_copies(plan)

    def mesh_shape(self, shape: list) -> list:
        """A shape on the mesh's own columns: a trailing nCells + 1 (nEdges + 1) padding column dropped."""
        shape = [int(x) for x in shape]
        if shape and shape[-1] - 1 in (self.mesh.get("cells"), self.mesh.get("edges")):
            return shape[:-1] + [shape[-1] - 1]
        return shape

    def _find_padded_copies(self, plan: Path) -> None:
        """Copies a map made before the padding rule missed: an allocation whose recorded initial payload equals
        a padded root with its padding column dropped (the physics pool's copy on a limited-area mesh). A constant
        root proves nothing and is left alone, as the recorder leaves it."""
        import numpy as np
        named: dict[int, tuple[str, dict]] = {}
        for name, f in self.fields.items():
            named.setdefault(id(f), (name, f))
        todo = [(name, f) for name, f in named.values()
                if len(set(f["copies"])) == 1 and self.mesh_shape(f["shape"]) != list(f["shape"])]
        if not todo:
            return
        state = PlanState(plan)
        kept, key = self._padded_copies_file(todo, state)
        if self._take_padded_copies(kept, key, todo):
            return
        made: dict[str, dict] = {}
        for name, f in todo:
            root, off = f["copies"][0]
            shape = list(f["shape"])
            arr = np.frombuffer(state.raw(root), "<f4", count=int(math.prod(shape)), offset=off).reshape(shape)
            if arr.size == 0 or np.all(arr == arr.flat[0]):
                continue
            trim = np.ascontiguousarray(arr[..., :-1]).tobytes()
            limit = int(state.allocs[root].get("bytes") or 4 * arr.size) + 4096
            hits = [i for i, a in sorted(state.allocs.items())
                    if i != root and a.get("initial") and len(trim) <= int(a.get("bytes") or 0) <= limit
                    and state.raw(i)[:len(trim)] == trim]
            for i in hits:
                f["copies"].append((i, 0))
                f["copy_shapes"][i] = shape[:-1] + [shape[-1] - 1]
            if hits:
                f["alloc"], f["offset"], f["window_end"] = max(hits), 0, True
                self.found.append({"root": root, "copies": hits, "shape": f["copy_shapes"][hits[0]],
                                   "rule": "initial payload equals the root without its padding column"})
                made[name] = {"root": root, "hits": hits, "shape": f["copy_shapes"][hits[0]]}
            state._cache.clear()
        self.padded_copies_how = "made"
        if kept is not None:
            try:
                kept.parent.mkdir(parents=True, exist_ok=True)
                tmp = kept.with_name(kept.name + f".tmp{os.getpid()}")
                tmp.write_text(json.dumps({"schema": PADDED_COPIES_SCHEMA, "key": key, "fields": made}, indent=1) + "\n",
                               encoding="utf-8")
                os.replace(tmp, kept)
            except OSError:
                pass   # not kept: the next command searches again

    def _padded_copies_file(self, todo: list, state: "PlanState") -> tuple[Path | None, dict]:
        """Where the padded copies found on this plan are kept, and the key they belong to: the plan (its sha256),
        the map's content, every recorded initial payload as plan.json names it, the mesh's columns, the fields
        searched and the door program. THE COST THIS REMOVES: the search reads every recorded initial payload of the
        plan (300 payloads, 17.6 s on the 3 km storm example, measured on processors) on every command, before the
        command's runner starts."""
        key = {"plan_sha256": self.doc.get("plan_sha256"), "map_sha256": _json_sum(self.doc),
               "payloads_sha256": _json_sum([[i, a.get("bytes"), a.get("initial")]
                                             for i, a in sorted(state.allocs.items()) if a.get("initial")]),
               "mesh": {"cells": self.mesh.get("cells"), "edges": self.mesh.get("edges")},
               "searched": [[name, list(f["copies"][0]), list(f["shape"])] for name, f in todo],
               "door_sha256": _sha256_file(Path(__file__))}
        try:
            return objectives_dir().parent / "fieldmaps" / f"{_json_sum(key)[:32]}.json", key
        except OSError:
            return None, key

    def _take_padded_copies(self, kept: Path | None, key: dict, todo: list) -> bool:
        """The kept search result of this key laid over the map, or False (none kept, another key, unreadable)."""
        if kept is None:
            return False
        try:
            doc = json.loads(kept.read_text(encoding="utf-8"))
            if doc.get("schema") != PADDED_COPIES_SCHEMA or doc.get("key") != json.loads(json.dumps(key)):
                return False
            laid = []
            for name, f in todo:
                m = doc["fields"].get(name)
                if m:
                    laid.append((f, int(m["root"]), [int(i) for i in m["hits"]], [int(x) for x in m["shape"]]))
        except (OSError, ValueError, KeyError, TypeError):
            return False
        for f, root, hits, shape in laid:
            for i in hits:
                f["copies"].append((i, 0))
                f["copy_shapes"][i] = list(shape)
            f["alloc"], f["offset"], f["window_end"] = max(hits), 0, True
            self.found.append({"root": root, "copies": hits, "shape": list(shape),
                               "rule": "initial payload equals the root without its padding column"})
        self.padded_copies_how = "kept"
        return True

    @classmethod
    def for_plan(cls, plan: Path, root: Path, table: dict | None = None, mesh: dict | None = None) -> "FieldMap":
        plan_sha = _plan_facts(plan).sha256()
        candidates = [plan / "adjoint" / "alloc-map.json", plan / "alloc-map.json",
                      root / "plans" / (plan.name + ".adjoint") / "alloc-map.json"]
        for path in candidates:
            if path.is_file():
                doc = json.loads(path.read_text(encoding="utf-8"))
                aliases = ((table or {}).get("alloc_map_aliases") or {}).get(doc.get("schema"))
                return cls(doc, plan_sha, path, aliases, mesh=mesh, plan=plan)
        raise AdDoorError(f"plan {plan} has no allocation map (looked for {', '.join(str(p) for p in candidates)}): "
                          "the recorder writes it with every package")

    def end_boundary(self) -> int | None:
        """The recorded boundary the window ends at, when the map names it: window_steps + 1. None when the window
        reaches the recording's end (there the last step's closing copy makes the pool copy current) or the map
        names no boundaries."""
        if self.window_steps is None or not self.boundaries:
            return None
        end = int(self.window_steps) + 1
        return end if end <= max(self.boundaries) else None

    def end_place(self, key: str) -> tuple[int, int] | None:
        """(allocation, byte offset) holding the map's field ``key`` at the window end, from the boundary map; None
        when end_boundary() is None. A window that ends inside the recording at a boundary that does not name the
        field is refused: the pool copy there holds the state one step earlier."""
        end = self.end_boundary()
        if end is None:
            return None
        place = self.boundaries.get(end, {}).get(key)
        if place is None:
            raise AdDoorError(f"field {key!r}: this plan's allocation map does not say where it is at boundary {end} "
                              f"(the end of a {self.window_steps}-step window), and its other copies hold the state "
                              "one step before that; pass --steps equal to the plan's recorded steps, or a plan "
                              "whose map names that boundary")
        return place

    def at_window_end(self, table: dict, name: str) -> bool:
        """Whether this plan's map can read the field at the window end (False: only its initial copy is mapped)."""
        spec = table["fields"].get(name) or {}
        base = self.fields.get(spec.get("field"))
        end = self.end_boundary()
        if base is not None and end is not None:
            return spec.get("field") in self.boundaries.get(end, {})
        return base is not None and base.get("window_end", True)

    def resolve(self, table: dict, name: str, row: dict | None = None, window_end: bool = False) -> dict:
        """One user-facing field (``theta``, ``qv``, ``u`` ...) -> {alloc, offset, words, shape, copies}.

        ``window_end=True`` is asked by every reader of the field at the window end (a metric, a tangent output):
        a field mapped by its initial copy alone is then refused by name.

        ``alloc``/``offset`` is the copy the doors read at the window end: where the map's step boundaries put the
        field at the window's end boundary (window_steps + 1) when the window ends inside the recording; else the
        row's ``state_copy`` (the physics pool's canonical copy, current at the recording's end, or the dycore's own
        copy on a row that holds the physics). ``place`` says which."""
        spec = table["fields"].get(name)
        if spec is None:
            raise AdDoorError(f"field {name!r} is not in the table (fields: {', '.join(sorted(table['fields']))})")
        if row is not None and row.get("holds_physics") and spec.get("physics_diagnostic"):
            raise AdDoorError(f"field {name!r} is written by the physics, and this row holds the physics, so its "
                              "derivative is not carried; use a dynamics field (e.g. theta_m) or a row with the "
                              "physics live")
        base = self.fields.get(spec["field"])
        if base is None:
            raise AdDoorError(f"field {name!r} ({spec['field']}) is not in this plan's allocation map {self.source}")
        end = self.end_place(spec["field"]) if window_end else None
        if window_end and end is None and not base.get("window_end", True):
            raise AdDoorError(f"field {name!r} ({spec['field']}): this plan's allocation map names only its initial "
                              f"copy (allocation {base['alloc']}), which the forecast never writes again, so its "
                              "value at the window end cannot be read from this plan; it can be perturbed and its "
                              "initial-state gradient read (sensitivity), not used as a metric or a tangent output")
        root_shape = [int(x) for x in base["shape"]]
        own = base.get("copy_shapes") or {}
        slot = None
        if "species" in spec:
            slots = base.get("species_slot") or {}
            if spec["species"] not in slots:
                raise AdDoorError(f"species {spec['species']} has no unambiguous scalar slot in {self.source}")
            slot = int(slots[spec["species"]])

        def place(alloc: int, offset: int, root: bool = False) -> tuple[int, list]:
            """(byte offset of the field in this copy, the copy's own shape of the field). ``root``: the copy is the
            field's own view (what the boundary map names), laid out as the map's shape."""
            shape = list(root_shape if root else own.get(alloc, root_shape))
            if slot is None:
                return offset, shape
            words = (int(base.get("slot_words") or math.prod(shape[1:])) if shape == root_shape
                     else int(math.prod(shape[1:])))
            return offset + 4 * words * slot, shape[1:]

        shape = self.mesh_shape(root_shape[1:] if slot is not None else root_shape)
        words = int(math.prod(shape))
        if end is not None:
            alloc, (offset, read_shape) = end[0], place(*end, root=True)
            where = (f"boundary {self.end_boundary()} of the recorder's map (the end of a "
                     f"{self.window_steps}-step window)")
        else:
            alloc, offset = base["alloc"], base["offset"]
            where = "the map's state copy" + (" (the window ends at the recording's end)"
                                              if window_end and self.boundaries and self.window_steps else "")
            if row is not None and row.get("state_copy") == "dycore":
                others = sorted(c for c in base["copies"] if c != (alloc, offset))
                if not others:
                    raise AdDoorError(f"field {name!r} has no dycore copy in {self.source} (only {alloc}); this row "
                                      "reads the dycore's copies")
                alloc, offset = min(others)
            offset, read_shape = place(alloc, offset)
        placed = [(a, *place(a, o)) for a, o in base["copies"]]
        return {"name": name, "units": spec["units"], "grid": spec["grid"], "what": spec["what"],
                "alloc": alloc, "offset": offset, "words": words, "shape": shape, "read_shape": read_shape,
                "copies": [(a, o) for a, o, _ in placed], "copy_shapes": [s for _, _, s in placed],
                "place": where}


#: schema of a kept padded-copies search (FieldMap._find_padded_copies)
PADDED_COPIES_SCHEMA = "gpuwm-hex.ad-padded-copies/v1"


class PlanState:
    """Boundary-0 primal words of a plan package (the recorded initial payloads), read on demand."""

    def __init__(self, plan: Path):
        self.plan = plan
        self.allocs = {int(a["id"]): a for a in _plan_facts(plan).allocations()}   # kept per plan: no parse
        self._cache: dict[int, bytes] = {}

    def raw(self, alloc: int) -> bytes:
        if alloc not in self._cache:
            entry = self.allocs.get(alloc, {}).get("initial")
            if entry is None:
                raise AdDoorError(f"allocation {alloc} has no recorded initial payload in {self.plan}")
            path = self.plan / entry["path"]
            data = gzip.open(path).read() if entry.get("encoding") == "gzip" else path.read_bytes()
            self._cache[alloc] = data
        return self._cache[alloc]

    def field(self, f: dict):
        import numpy as np
        data = self.raw(f["alloc"])
        shape = f.get("read_shape", f["shape"])
        arr = np.frombuffer(data, "<f4", count=int(math.prod(shape)), offset=f["offset"]).astype(np.float64)
        return on_mesh(arr.reshape(shape), f["shape"])


def on_mesh(values, shape):
    """A copy's values on the mesh's own columns (a padding column dropped)."""
    return values[..., :shape[-1]] if values.shape[-1] != shape[-1] else values


# ------------------------------------------------------------------------------------------------------- mesh
def mesh_geometry(grid: Path, cells: int, edges: int):
    """latCell, lonCell (degrees, lon in -180..180), areaCell and latEdge/lonEdge of the row's mesh."""
    import numpy as np
    from woof.netcdf_bridge import open_dataset
    with open_dataset(grid) as ds:
        out = {name: np.asarray(ds.variables[name][:], np.float64).ravel()
               for name in ("latCell", "lonCell", "areaCell", "latEdge", "lonEdge") if name in ds.variables}
    for name in ("latCell", "lonCell", "latEdge", "lonEdge"):
        if name in out:
            out[name] = np.degrees(out[name])
            if name.startswith("lon"):
                out[name] = np.where(out[name] > 180.0, out[name] - 360.0, out[name])
    if out["latCell"].size != cells:
        raise AdDoorError(f"grid {grid} has {out['latCell'].size} cells; the row's mesh has {cells}")
    if "areaCell" not in out:
        out["areaCell"] = np.ones(cells)
    return out


def great_circle_km(lat0: float, lon0: float, lat, lon):
    import numpy as np
    a, b = np.radians(lat0), np.radians(lon0)
    la, lo = np.radians(lat), np.radians(lon)
    c = np.sin(a) * np.sin(la) + np.cos(a) * np.cos(la) * np.cos(lo - b)
    return EARTH_RADIUS_KM * np.arccos(np.clip(c, -1.0, 1.0))


def parse_levels(text: str | None, count: int) -> list[int]:
    """'a-b' or 'a,b,c' (1-based model levels) -> 0-based list; default every level."""
    if not text:
        return list(range(count))
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            out.extend(range(lo, hi + 1))
        elif part:
            out.append(int(part))
    bad = [k for k in out if not 1 <= k <= count]
    if bad or not out:
        raise AdDoorError(f"--levels {text!r}: levels run 1..{count}")
    return [k - 1 for k in dict.fromkeys(out)]


DEFAULT_AT = "20.4,-26.2"


def resolve_at(at: str | None, geometry: dict, reach_km: float = 1500.0) -> str:
    """The bump centre and metric region centre: --at when given; else 20.4,-26.2 when the mesh has a cell within
    reach_km of it (every globe); else the cell nearest the mesh's own centre (a limited-area mesh).

    THE BREAKAGE: the default 20.4N 26.2W lies thousands of km outside the 3 km storm box (40.6N 91.3W), so a plain
    3 km adjoint-test was refused ("the metric region (cells within 1500 km of 20.4N -26.2E) holds no cell of this
    mesh") and a plain 3 km tangent ran on a bump whose Gaussian is about exp(-34) everywhere in the box: a valid
    window on a perturbation of nothing (a 3 km run, 2026-11-11)."""
    import numpy as np
    if at is not None:
        return at
    lat, lon = np.asarray(geometry["latCell"], dtype=float), np.asarray(geometry["lonCell"], dtype=float)
    la0, lo0 = (float(x) for x in DEFAULT_AT.split(","))
    if great_circle_km(la0, lo0, lat, lon).min() <= reach_km:
        return DEFAULT_AT
    la, lo = np.radians(lat), np.radians(lon)
    v = np.array([np.mean(np.cos(la) * np.cos(lo)), np.mean(np.cos(la) * np.sin(lo)), np.mean(np.sin(la))])
    cla, clo = np.degrees(np.arctan2(v[2], np.hypot(v[0], v[1]))), np.degrees(np.arctan2(v[1], v[0]))
    k = int(np.argmin(great_circle_km(cla, clo, lat, lon)))
    return f"{lat[k]:.4f},{((lon[k] + 180.0) % 360.0) - 180.0:.4f}"


def select_cells(geometry: dict, box: str | None, circle: str | None):
    """Indices of the cells inside --box S,N,W,E or --circle LAT,LON,RADIUS_KM (default: every cell)."""
    import numpy as np
    lat, lon = geometry["latCell"], geometry["lonCell"]
    if box and circle:
        raise AdDoorError("--box and --circle are alternatives")
    if box:
        s, n, w, e = (float(x) for x in box.split(","))
        inlon = (lon >= w) & (lon <= e) if w <= e else (lon >= w) | (lon <= e)
        cells = np.where((lat >= s) & (lat <= n) & inlon)[0]
        where = f"box {s}..{n} N, {w}..{e} E"
    elif circle:
        la, lo, r = (float(x) for x in circle.split(","))
        cells = np.where(great_circle_km(la, lo, lat, lon) <= r)[0]
        where = f"cells within {r:g} km of {la}N {lo}E"
    else:
        cells = np.arange(lat.size)
        where = "every cell"
    if cells.size == 0:
        raise AdDoorError(f"the metric region ({where}) holds no cell of this mesh")
    return cells, where


def metric_not_offered(table: dict, name: str) -> str:
    """The reason a measure is left out of the table on purpose (table: metrics_not_offered), as a clause; else ''."""
    for key, why in (table.get("metrics_not_offered") or {}).items():
        if key in str(name).lower():
            return f": {why}"
    return ""


def metric_objective(table: dict, fmap: FieldMap, geometry: dict, name: str, levels_text: str | None,
                     box: str | None, circle: str | None, row: dict | None = None) -> dict:
    """The linear objective J = area-weighted mean of a field over a region and levels at the window end."""
    import numpy as np
    spec = table["metrics"].get(name)
    if spec is None:
        raise AdDoorError(f"--metric {name!r} is not in the table (metrics: {', '.join(sorted(table['metrics']))})"
                          f"{metric_not_offered(table, name)}")
    f = fmap.resolve(table, spec["field"], row, window_end=True)
    if f["grid"] == "edge":
        raise AdDoorError(f"--metric {name}: edge fields have no cell mean")
    nlev, ncol = f["shape"]
    levels = parse_levels(levels_text, nlev)
    cells, where = select_cells(geometry, box, circle)
    area = geometry["areaCell"][cells]
    weight = area / area.sum() / len(levels)
    base = f["offset"] // 4
    stride = f.get("read_shape", f["shape"])[-1]
    terms = [[f["alloc"], base + k * stride + int(c), float(wt)] for k in levels for c, wt in zip(cells, weight)]
    label = (f"{spec['what']} ({f['name']}, {f['units']}) over {where}, model levels "
             f"{levels_text or f'1-{nlev}'}, at the window end; area weighted")
    return {"name": label, "terms": terms, "metric": name, "field": f["name"], "units": f["units"],
            "cells": [int(c) for c in cells], "levels_1based": [k + 1 for k in levels], "allocation": f["alloc"],
            "byte_offset": f["offset"], "window_steps": fmap.window_steps, "read_at": f["place"]}


def theta_direction(table: dict, fmap: FieldMap, state: PlanState, geometry: dict, at: str,
                    amplitude: float, radius_km: float, level: int, width: float) -> dict:
    """A potential-temperature bump at fixed qv and density, through the linearised hex write-back (T0)."""
    import numpy as np
    k = table["constants"]
    rvord, rgas, rcv, p0 = k["rv"] / k["rgas"], k["rgas"], k["rgas"] / (k["cp"] - k["rgas"]), k["p0"]
    lat0, lon0 = (float(x) for x in at.split(","))
    get = {name: fmap.resolve(table, name) for name in ("theta_m", "qv", "rho_zz", "rho_p", "rtheta_p",
                                                        "rho_theta", "exner", "pressure_p")}
    thm, qv, rz, rtp, rth, ex = (state.field(get[n]) for n in ("theta_m", "qv", "rho_zz", "rtheta_p", "rho_theta",
                                                               "exner"))
    nlev, ncol = thm.shape
    if not 1 <= level <= nlev:
        raise AdDoorError(f"--level {level}: levels run 1..{nlev}")
    rtbase = rth - rtp
    zz = ex ** (1.0 / rcv) * p0 / (rgas * rth)
    hz = np.exp(-(great_circle_km(lat0, lon0, geometry["latCell"], geometry["lonCell"]) / radius_km) ** 2)
    vz = np.exp(-((np.arange(nlev) - (level - 1)) / width) ** 2)
    dth = amplitude * vz[:, None] * hz[None, :]
    keep = np.abs(dth) > BUMP_KEEP * abs(amplitude)   # by size: a cooling bump (amplitude < 0) keeps its own peak cells
    a = 1.0 + rvord * qv
    t_thm = dth * a
    t_rtp = t_thm * rz
    t_ex = rcv * ex / rth * t_rtp
    t_pp = zz * rgas * (t_ex * rtp + ex * t_rtp + rtbase * t_ex)
    terms = []
    idx = np.argwhere(keep)
    for name, values in (("theta_m", t_thm), ("rtheta_p", t_rtp), ("rho_theta", t_rtp), ("exner", t_ex),
                         ("pressure_p", t_pp)):
        for (alloc, off), shape in zip(get[name]["copies"], get[name]["copy_shapes"]):
            base, stride = off // 4, shape[-1]
            terms.extend([alloc, base + int(kk) * stride + int(c), float(values[kk, c])] for kk, c in idx)
    return refuse_zero_direction({
        "name": (f"theta bump: peak {amplitude:g} K at {lat0}N {lon0}E, Gaussian R {radius_km:g} km, centred on "
                 f"model level {level} (width {width:g}); qv and density fixed; through the write-back to "
                 "theta_m, rtheta_p, rho_theta, exner, pressure_p and every copy"),
        "terms": terms, "words_per_field": int(keep.sum()), "peak_K": amplitude})


def refuse_zero_direction(direction: dict) -> dict:
    """A perturbation with no nonzero word is refused by name before any card work (tangent and adjoint-test; the
    sensitivity metric is refused the same way by select_cells and parse_levels when it holds no cell or level).

    THE BREAKAGE: a plain 3 km tangent printed "valid True" on a bump of nothing (2026-11-11): direction.json held
    terms [] and words_per_field 0 and every output was 0. The whole-state dot of
    that run passed on the runner's own seeded direction, which checks the operator, not the direction typed."""
    terms = direction.get("terms") or []
    if not any(float(t[2]) != 0.0 for t in terms):
        what = str(direction.get("name", "the perturbation")).split(";")[0]
        raise AdDoorError(f"the perturbation is zero on every cell of this mesh ({what}: {len(terms)} words, none "
                          "nonzero), so every output would be 0 and a passing check would say nothing about it. "
                          "Put the bump on the mesh with --at LAT,LON (and --radius-km, --level), or give a "
                          "nonzero --amplitude")
    return direction


#: A bump word below this share of the bump's peak is left out of the direction (theta_direction); a cell is on the
#: bump when its Gaussian factor is above it, i.e. within about 3.03 radii of the centre.
BUMP_KEEP = 1e-4


def adjoint_test_region(args: argparse.Namespace, at: str) -> tuple[str | None, str | None]:
    """adjoint-test's metric region as select_cells takes it: --box, else --circle, else 1.25 bump radii around the
    bump. THE BREAKAGE: --box went in beside the default circle, and select_cells refuses the two together."""
    box = getattr(args, "box", None)
    circle = getattr(args, "circle", None) or (None if box else f"{at},{1.25 * (args.radius_km or 1200.0)}")
    return box, circle


def early_geometry(args: argparse.Namespace, row: dict, row_name: str) -> dict | None:
    """The row's mesh cells before the plan is fetched: from --grid, a grid path the row names outright, the asset
    root the user names, or the pinned mesh file alone (2 MB for the 12 km cut, 26 MB and 28 MB for the 120 km globe
    and the 3 km storm box; fetched once, and the run reads it again). None when none of these gives it (an install
    whose asset module fetches only whole roots, a grid that does not open here): the checks after the downloads
    still run."""
    rel = (row.get("assets") or {}).get("grid")
    if getattr(args, "grid", None):
        grid = Path(args.grid)
    elif not rel:
        return None
    elif os.path.isabs(str(rel)):
        grid = Path(rel)
    elif getattr(args, "assets", None) is not None or os.environ.get("WOOF_HEX_AD_ASSETS"):
        grid = Path(args.assets if getattr(args, "assets", None) is not None
                    else os.environ["WOOF_HEX_AD_ASSETS"]) / str(rel)
    else:
        api = _runner_api()
        if api is None or not hasattr(api, "asset_root"):
            return None
        try:
            grid = Path(api.asset_root(want=[str(rel)], rows=False)) / str(rel)
        except TypeError:   # an asset module without the mesh-only fetch
            return None
        except (api.AdAssetError, OSError, ValueError) as error:
            raise AdDoorError(f"differentiable-engine assets: {error}") from error
    if not grid.is_file():
        return None
    try:
        return mesh_geometry(grid, int(row["cells"]), int(row.get("edges") or 0))
    except AdDoorError:
        raise
    except Exception:   # a grid this install cannot open: the run's own read says why, later
        return None


def refuse_off_mesh(table: dict, row_name: str, row: dict, args: argparse.Namespace, door: str,
                    geometry: dict | None) -> None:
    """Before the plan and the row files: refuse a bump with no cell of the row's mesh under it, a bump of size 0,
    a bump level the row does not have, and a metric region that holds no cell, from the mesh file alone.

    THE BREAKAGE THIS PREVENTS: ``adjoint-test --at 10,-150`` on the 12 km cut fetched every file of the example
    (1.19 GB, 3.3 GB unpacked) and only then said "the perturbation is zero on every cell of this mesh" (a 0.6.0
    build: 26.5 s from a local folder); the mesh that answers it is a 2 MB file. The same checks after the
    downloads (theta_direction, refuse_zero_direction, select_cells) stay for a plan or grid given by hand."""
    import numpy as np
    if geometry is None:
        return
    lat, lon = np.asarray(geometry["latCell"], float), np.asarray(geometry["lonCell"], float)
    mesh = row.get("mesh") or row_name
    after = "Nothing but the row's mesh file was downloaded"
    at = None
    if door in ("tangent", "adjoint-test") and hasattr(args, "at"):
        spec = table["perturbations"]["theta"]["default"]
        at = resolve_at(args.at, geometry)
        amplitude = args.amplitude if args.amplitude is not None else spec["amplitude"]
        radius = args.radius_km if args.radius_km is not None else spec["radius_km"]
        nlev = row.get("levels")
        if nlev and not 1 <= int(args.level) <= int(nlev):
            raise AdDoorError(f"--level {args.level}: row {row_name} has model levels 1..{int(nlev)}. {after}")
        if not float(amplitude) or not float(radius):
            raise AdDoorError(f"--amplitude {amplitude:g} --radius-km {radius:g}: the bump is zero on every cell, so "
                              "every output would be 0; give a nonzero --amplitude and --radius-km. "
                              f"{after}")
        lat0, lon0 = (float(x) for x in at.split(","))
        dist = great_circle_km(lat0, lon0, lat, lon)
        if not (np.exp(-(dist / float(radius)) ** 2) > BUMP_KEEP).any():
            reach = abs(float(radius)) * float(np.sqrt(-np.log(BUMP_KEEP)))
            centre = resolve_at(None, geometry)
            raise AdDoorError(f"--at {at}: the bump (Gaussian R {float(radius):g} km, reaching about {reach:,.0f} km) "
                              f"touches no cell of row {row_name}'s mesh {mesh}: its nearest cell is "
                              f"{float(dist.min()):,.0f} km from {lat0}N {lon0}E, and the mesh's cells lie at "
                              f"{lat.min():.1f} to {lat.max():.1f} N, {lon.min():.1f} to {lon.max():.1f} E, so every "
                              f"output would be 0. Put the bump on the mesh: --at {centre} is its centre (the "
                              f"default when --at is left out), or give a larger --radius-km. {after}")
    if getattr(args, "objectives", None) is not None:
        return   # many objectives: each one's region is judged when it is made
    if door == "adjoint-test" and at is not None:
        box, circle = adjoint_test_region(args, at)
    else:
        box, circle = getattr(args, "box", None), getattr(args, "circle", None)
    if box or circle:
        try:
            select_cells(geometry, box, circle)
        except AdDoorError as error:
            raise AdDoorError(f"{error} (mesh {mesh}: cells at {lat.min():.1f} to {lat.max():.1f} N, "
                              f"{lon.min():.1f} to {lon.max():.1f} E). {after}") from error


def _plan_dt(plan: Path) -> float | None:
    """The timestep a plan package was recorded with (plan.json physics.dt_seconds), when it names one."""
    return _plan_facts(plan).dt()   # kept per plan: no parse


def _door_asset_env(table: dict | None, door: str | None) -> dict:
    """The runner files a door swaps for the row's own (param-grad: AD_SCHEMES = the parameter-carrier bundle)."""
    if not table or not door:
        return {}
    return dict(table["doors"].get(door, {}).get("asset_env") or {})


def _needed_assets(table: dict, row: dict, door: str) -> set[str]:
    """The row asset keys a door hands the runner (its flags and environment), so only those images are built."""
    keys = set(row["asset_args"].values()) | set(row["asset_env"].values())
    keys |= set(table["doors"].get(door, {}).get("asset_env", {}).values())
    return keys


# ------------------------------------------------------------------------------------------------------- window
NO_LOCAL_GPU_ENV = ("WOOF_HEX_NO_LOCAL_GPU", "GPUWM_NO_LOCAL_GPU")


def refuse_no_local_gpu() -> None:
    """A machine that declares no GPU work here (WOOF_HEX_NO_LOCAL_GPU / GPUWM_NO_LOCAL_GPU) gets a named refusal
    before the door reads or launches on any card. Without it a door read card 0 of such a machine and could have run
    there (adjoint-test refused only on size, on a 32 GB RTX 5090, 2026-11-11)."""
    for variable in NO_LOCAL_GPU_ENV:
        if os.environ.get(variable, "") not in ("", "0"):
            raise AdDoorError(f"no local GPU: {variable} is set, so this machine has declared that no GPU work happens "
                              "here and the door does not read or use a card. Unset it to run the door on this "
                              "machine's card, or run it on the machine that owns the card")



def refuse_hidden_card() -> None:
    """CUDA reads an empty CUDA_VISIBLE_DEVICES (or one that starts with -1) as "no card": a runner is not started on
    a card the caller hid. THE BREAKAGE THIS PREVENTS: the door read the empty word as card 0 and started the runner
    with CUDA_VISIBLE_DEVICES=0, so a command run with every card hidden ran its windows on card 0 (a no-card
    dry run of a 0.6.1 build: three 12 km windows on an RTX 5090; a second
    run saw "card 0 has 8875 MiB free" with the cards hidden)."""
    hidden = os.environ.get("CUDA_VISIBLE_DEVICES")
    if hidden is not None and (not hidden.strip() or hidden.strip().startswith("-")):
        raise AdDoorError(f"no card is visible to this command: CUDA_VISIBLE_DEVICES={hidden!r} names none, so the "
                          "runner is not started on any card. Unset it, or name the card to use")


def _within_visible(card: str) -> str:
    """A --card number counts inside a CUDA_VISIBLE_DEVICES a scheduler already set, as CUDA counts it.
    Without this, --card 0 on a job the scheduler gave card 7 ran on physical card 0 beside another job."""
    vis = [c.strip() for c in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if c.strip()]
    if not vis or vis[0].startswith("-"):          # unset, or -1 (no card: the tests hide the cards this way)
        return str(card)
    i = int(card)
    if not 0 <= i < len(vis):
        raise SystemExit(f"--card {card} is outside the cards this job was given (CUDA_VISIBLE_DEVICES={','.join(vis)})")
    return vis[i]


def card_of(args: argparse.Namespace) -> str:
    """--card (counted inside CUDA_VISIBLE_DEVICES when one is set), else the scheduler's first card, else 0."""
    if args.card is not None:
        return _within_visible(str(args.card))
    return os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0"


def cards_of(args: argparse.Namespace) -> list[str]:
    """--cards A,B,..., else --card, else every card of the CUDA_VISIBLE_DEVICES a scheduler set, else 0."""
    given = getattr(args, "cards", None)
    if given:
        return [_within_visible(c.strip()) for c in str(given).split(",") if c.strip()]
    if args.card is not None:
        return [_within_visible(str(args.card))]
    return [c for c in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if c] or ["0"]


def free_mib(card: str) -> int | None:
    refuse_no_local_gpu()
    refuse_hidden_card()   # a card hidden with CUDA_VISIBLE_DEVICES is not read either
    try:
        out = subprocess.run(["nvidia-smi", "-i", str(card), "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30)
        return int(out.stdout.strip().splitlines()[0]) + int(_SESSION_HELD["card"].get(str(card), 0))
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def total_mib(card: str) -> int | None:
    refuse_no_local_gpu()
    refuse_hidden_card()   # a card hidden with CUDA_VISIBLE_DEVICES is not read either
    try:
        out = subprocess.run(["nvidia-smi", "-i", str(card), "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30)
        return int(out.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def _first_card_total(args: argparse.Namespace) -> int | None:
    """The whole memory of the first card this door will use, or None where no card may be read (a machine that
    declares no GPU work, no nvidia-smi): the row is then chosen by mesh size alone, as before."""
    if any(os.environ.get(v, "") not in ("", "0") for v in NO_LOCAL_GPU_ENV):
        return None
    try:
        return total_mib(cards_of(args)[0])
    except (AdDoorError, SystemExit, ValueError, IndexError):
        return None


def compute_capability(card: str) -> str | None:
    refuse_no_local_gpu()
    refuse_hidden_card()   # a card hidden with CUDA_VISIBLE_DEVICES is not read either
    try:
        out = subprocess.run(["nvidia-smi", "-i", str(card), "--query-gpu=compute_cap", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30)
        cc = out.stdout.strip().splitlines()[0].strip()
    except (OSError, IndexError, subprocess.TimeoutExpired):
        return None
    # a card that is not there prints "No devices were found" (rc 6) on stdout: that is no card, not a capability
    major, _, minor = cc.partition(".")
    return cc if out.returncode == 0 and major.isdigit() and minor.isdigit() else None


def refuse_card_too_small(card: str, need: int) -> None:
    """A window that needs more than the whole card cannot run there; waiting cannot help, so --wait-card is not offered."""
    whole = total_mib(card)
    if whole is not None and need > whole:
        raise AdDoorError(f"card {card} has {whole} MiB in all; this row's window needs {need} MiB, more than the "
                          "whole card, so it cannot run on this card. Use a card with at least that much memory "
                          "(--need-mib N overrides the row's need, at your own risk)")


def host_available_mib(meminfo: Path = Path("/proc/meminfo")) -> int | None:
    """The host memory available to a new process now (MemAvailable), in MiB; None where the kernel does not say."""
    try:
        for line in meminfo.read_text(encoding="ascii", errors="replace").splitlines():
            if line.startswith("MemAvailable:"):
                held = int(_SESSION_HELD["host"]) if str(meminfo) == "/proc/meminfo" else 0
                return int(line.split()[1]) // 1024 + held   # the kernel writes kB (KiB)
    except (OSError, ValueError, IndexError):
        return None
    return None


def host_available_settled(wait_s: float | None = None) -> int | None:
    """The host memory available once a program this command has just stopped has given its memory back: read every
    second until two readings a second apart differ by less than 1 GiB, for at most wait_s (SESSION_RELEASE_WAIT_S).
    THE BREAKAGE THIS PREVENTS (card job of 2026-11-14, second run): a command with other words than the loaded
    program's stopped it and read 106,048 MiB available; the next command, 0.3 s later, read 114,986 MiB; both
    were refused for host memory that the stopped program was still giving back."""
    wait_s = SESSION_RELEASE_WAIT_S if wait_s is None else float(wait_s)
    last, t0 = host_available_mib(), time.monotonic()
    while last is not None and time.monotonic() - t0 < wait_s:
        time.sleep(1.0)
        now = host_available_mib()
        if now is None or now - last < 1024:
            return last if now is None else now
        last = now
    return last


def refuse_host_too_small(row_name: str, need: int | None, available: int | None, what: str | None = None,
                          smaller: str | None = None, advice: str | None = None) -> None:
    """Stop, before any download, a window the host cannot hold.

    THE BREAKAGE THIS PREVENTS: the 120 km window's runner holds about 99 GiB of host memory (adjoint-test, 3 steps,
    the default row: 103,711,332 KiB of process memory read on 2026-11-12); on a host with less the kernel ends the
    runner part way through, after the first-use download, with no sentence a user can act on. A row with no measured
    host need is not judged here."""
    if need is None or available is None or available >= need:
        return
    which = f" ({what})" if what else ""
    other = f" {smaller}" if smaller else ""
    # ``advice``: what to do, where "a host with that much" would mislead (a need that follows the host's size)
    advice = advice or "Run it on a host with at least that much free memory, or free memory here"
    raise HostMemoryError(f"this host has {available:,} MiB of memory available and row {row_name}'s window needs "
                          f"about {need:,} MiB{which}, so it would be ended part way through.{other} {advice} "
                          "(--need-host-mib N overrides the row's need, at "
                          "your own risk). Nothing but the row's mesh file was downloaded")


def with_bstore_ram_gib(argv: list[str], gib: int) -> list[str]:
    """The runner's words with its host-memory store budget set to ``gib`` (replacing one the words name)."""
    out = list(argv)
    for k, word in enumerate(out[:-1]):
        if word == "--bstore-ram-gib":
            out[k + 1] = str(int(gib))
            return out
    return out + ["--bstore-ram-gib", str(int(gib))]


def smallest_host_words(need: dict | None) -> str | None:
    """What to do about a window whose host need follows the host (store_need's host_smallest_mib): the smallest
    host it fits, or a shorter window. None for any other need."""
    least = (need or {}).get("host_smallest_mib")
    if not least:
        return None
    return (f"Run it where at least {int(least):,} MiB are available (the runner's own memory and one step "
            "boundary in host memory, the rest in the run folder), or run a shorter window (--steps)")


def answer_check_expected(row_name: str, steps: int, args: argparse.Namespace, door: str | None) -> tuple[bool, str]:
    """(whether this command will run the answer check's windows, why), judged before any download from the
    command's words and the verdicts kept here: --no-self-dot and --strict-from run none, --strict runs it, and a
    passing verdict kept by this door program for this row and window length stands for it. THE BREAKAGE THIS
    PREVENTS: a 3 km gradient that held about 152 GB of host memory was refused on a host with less than the 258 GB
    the first call's answer check held, and the sentence did not say which of the two it had judged."""
    if getattr(args, "no_self_dot", False):
        return False, "--no-self-dot: no answer check"
    if getattr(args, "strict_from", None) is not None:
        return False, "--strict-from: the answer check of a bound receipt"
    if getattr(args, "self_dot", False) or door == "adjoint-test":
        return True, "the answer check runs in this command"
    try:
        mine = _sha256_file(Path(__file__))
        for path in sorted(checks_dir().glob("*.json")):
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            key = doc.get("key") or {}
            if (doc.get("pass") is True and key.get("row") == row_name and int(key.get("steps") or -1) == int(steps)
                    and (key.get("door_code") or {}).get("ad_doors_sha256") == mine):
                return False, "an answer check's verdict for this row and window length is kept here"
    except OSError:
        pass
    return True, "the first call on a plan runs the answer check"


def window_length_words(seconds: float) -> str:
    """A window's length for a title or a line: minutes under an hour, hours from there. THE BREAKAGE THIS PREVENTS:
    the 3 km map's title read "+0.245833 h" for its 14.75-minute window."""
    return f"{seconds / 60:g} min" if seconds < 3600 else f"{seconds / 3600:g} h"


def bstore_ram_gib(ram: dict | None, available: int | None) -> int | None:
    """The gradient line's host-memory store for its checkpoints (--bstore-ram-gib): its table maximum, or the table's
    share of the host memory available now, whichever is smaller; None without the table entry, a readable host or
    at least 1 GiB."""
    if not ram or not available:
        return None
    gib = int(min(float(ram["max_gib"]), available * float(ram.get("host_share", 0.6)) / 1024.0))
    return gib if gib >= 1 else None


def store_need(row: dict, steps: int, available: int | None, line_on: bool,
               check_runs: bool = True, keep_forward: bool = False) -> dict | None:
    """What a window's step store will use on this host, from the row's measured boundary size (``boundary_mib``):
    the gradient line keeps boundary 0, the window end and ceil(sqrt(steps)) checkpoints (its checkpointed sweep),
    in host memory up to --bstore-ram-gib and in files past it; every other window keeps every step boundary, in
    host memory up to the runner's default budget (0.4 of the memory available) and in files past it. The host need
    of a gradient-line window is the line's measured memory beside the store plus the part held in memory; the disk
    need is the part in files plus the row's margin. None for a row with no measured boundary size.

    THE BREAKAGE THIS PREVENTS: the 3 km storm sensitivity told the user it needed 140,000 MiB of host memory and
    named no disk; it held 278 GiB of host memory and wrote 445 GB to its run folder (2026-11-14)."""
    bmib = row.get("boundary_mib")
    if not bmib:
        return None
    steps = max(1, int(steps))
    line = row.get("gradient_line") or {}
    if line_on:
        kept = min(steps + 1, math.ceil(math.sqrt(steps)) + 2)
        cap_gib = bstore_ram_gib(line.get("bstore_ram"), available)
        cap = (cap_gib or 0) * 1024
        how = (f"checkpointed sweep: boundary 0, the window end and {kept - 2} checkpoints of {int(bmib):,} MiB, "
               f"in host memory up to {cap_gib or 0} GiB (--bstore-ram-gib), the rest in files")
    else:
        kept = steps + 1
        cap = int(0.4 * available) if available else 0
        cap_gib = None   # the runner's default budget; window_host sets one where that does not fit the host
        how = (f"every step boundary kept: {kept} of {int(bmib):,} MiB, in host memory up to 0.4 of the memory "
               "available (the runner's default budget), the rest in files")
    store = kept * int(bmib)
    ram = min(store, cap)
    # The forward kept between gradients of one forecast (keep_forward_plan): the loaded program keeps the forward's
    # checkpoints and the window end in page-locked host memory, beside the store's budget, from its first gradient's
    # sweep on. It keeps them only when the store's checkpoints are whole in host memory (the runner's own rule).
    # THE BREAKAGE THIS PREVENTS: a command admitted on the gradient's own need whose program then held nine more
    # states (94.8 GB at 59 steps on the 3 km storm box) than the command had stated.
    kept_fwd = (kept - 1) * int(bmib) if line_on and keep_forward and ram >= store else 0
    # A short window: where the store's checkpoints and the kept forward fit together under the store's host budget
    # (--bstore-ram-gib), the loaded program's later gradient holds the kept forward a second time beside the store
    # it makes again (measured: the 12-step program held 121,796 MiB after its first gradient and 172,688 MiB after
    # the second, which took the kept forward; the 59-step program, whose two do not fit under the budget together,
    # held 213,650 and 213,687 MiB).
    # THE BREAKAGE THIS PREVENTS: a later 12-step command that attached and took the kept forward held 182,879 MiB
    # of host memory where the command had stated and been admitted on 168,733 MiB (a 0.6.0 build's card job,
    # 2026-11-14).
    kept_again = kept_fwd if kept_fwd and cap > 0 and store + kept_fwd <= cap else 0
    if kept_fwd:
        how += (f"; the forward kept for the next gradient of the same forecast: {kept - 1} more states "
                f"({kept_fwd:,} MiB) in host memory"
                + (f", counted twice on this short window ({2 * kept_fwd:,} MiB: a later gradient holds the kept "
                   "forward beside the store it makes again)" if kept_again else "")
                + " (--keep-forward off runs without them)")
    # the gradient line's host need: the call with its answer check (base_mib, the first call on a plan), or the
    # gradient alone (gradient_base_mib) when this command runs no check (answer_check_expected)
    host = host_check = host_gradient = None
    smallest = None
    wh = None if line_on else row.get("window_host")
    if wh and available:
        # A window that keeps every step boundary: the runner holds its own memory (base_mib at base_steps, growing
        # per_step_mib a step) beside the boundaries it keeps in host memory, and it keeps them there up to
        # store_share of the memory available WHEN THE WINDOW STARTS (its default budget), so the peak follows the
        # host. The need is the two together with the table's margin over the measured peaks.
        # THE BREAKAGE THIS PREVENTS: the 3 km adjoint-test (59 steps) stated and was admitted on 110,592 MiB, a
        # 3-step measure, and held 262,306 MiB and 254,885 MiB on its two card runs (2026-11-15); on a host with
        # between 110,592 MiB and about what host_smallest_mib says available it was admitted, then ended part way.
        margin, share = float(wh.get("margin", 1.1)), float(wh.get("store_share", 0.4))
        own = int(wh["base_mib"]) + int(wh.get("per_step_mib") or 0) * max(0, steps - int(wh.get("base_steps") or 0))
        host = math.ceil(margin * (own + ram))
        # A host that cannot hold the runner's default store (share of what is available) runs the window with a
        # smaller one: --bstore-ram-gib G, the largest whole GiB that fits beside the runner's own memory with the
        # margin, and never less than one boundary (the one in memory, or the copy a boundary is written to the run
        # folder from). The runner restores the same bytes from memory or from its files, so the window's numbers do
        # not move; the rest of the store goes to the run folder (the disk need below grows by it).
        # THE BREAKAGE THIS PREVENTS: the 3 km adjoint-test (59 steps) was refused on every host with less than
        # 158,958 MiB available (a 128 GB or 192 GB machine) although its own memory is 87,805 MiB: only the
        # runner's default budget for the store (0.4 of what is available) did not fit.
        one_gib = max(1, math.ceil(int(bmib) / 1024))
        if available < host:
            cap_gib = max(one_gib, int((available / margin - own) // 1024))
            ram = min(store, cap_gib * 1024)
            host = math.ceil(margin * (own + ram))
            how = (f"every step boundary kept: {kept} of {int(bmib):,} MiB, in host memory up to {cap_gib} GiB "
                   "(--bstore-ram-gib: this host is too small for the runner's default of 0.4 of the memory "
                   "available), the rest in files")
        # the smallest host the window fits: its own memory and one boundary's GiB
        smallest = min(math.ceil(margin * (own + store)), math.ceil(margin * (own + one_gib * 1024)))
    if line_on and line.get("host"):
        host_gradient = (int(line["host"].get("gradient_base_mib") or line["host"]["base_mib"]) + ram + kept_fwd
                         + kept_again)
        # the check's windows run after the loaded program has ended (its kept forward is freed by then): the call's
        # peak is the larger of the two
        host_check = max(int(line["host"]["base_mib"]) + ram, host_gradient)
        host = host_check if check_runs else host_gradient
    return {"store_mib": store, "boundaries": kept, "ram_mib": ram, "files_mib": store - ram,
            "bstore_ram_gib": cap_gib if wh and available else None,
            "disk_mib": store - ram + int(row.get("disk_margin_mib") or 0), "host_mib": host, "how": how,
            "line": bool(line_on), "host_check_mib": host_check, "host_gradient_mib": host_gradient,
            "keep_forward_mib": kept_fwd, "keep_forward_again_mib": kept_again, "host_smallest_mib": smallest,
            "host_for": (None if host is None else
                         f"{steps} steps: the runner's own memory and the {ram // int(bmib)} of {kept} step "
                         "boundaries it keeps in host memory on a host of this size"
                         + (f" (--bstore-ram-gib {cap_gib}, the other {kept - ram // int(bmib)} in the run folder)"
                            if wh and available and cap_gib is not None else "") if not line_on
                         else "the gradient with its answer check" if check_runs
                         else "the gradient alone with its forward kept for the next one, no answer check in this "
                              "command" if kept_fwd
                         else "the gradient alone, no answer check in this command")}


def keep_forward_plan(row: dict, args: argparse.Namespace, line_on: bool, session: Any = None) -> dict:
    """Whether this command's loaded program keeps its forward between gradients of one forecast (the table's
    gradient_line.boundary_layer.routes.<route>.keep_forward): on where the route names it, unless --keep-forward
    off, and only in a command whose gradients run on a program that stays loaded (several targets in one command, or
    a session), because a program that serves one gradient and ends has no later gradient to keep it for."""
    name, route, _ = boundary_layer_route(row, getattr(args, "boundary_layer", None))
    kf = route.get("keep_forward") if line_on else None
    if not kf:
        return {"on": False, "offered": False, "route": name,
                "why": route.get("keep_forward_off") or "this row and route name no kept forward"}
    if getattr(args, "keep_forward", None) == "off":
        return {"on": False, "offered": True, "route": name, "why": "--keep-forward off"}
    if getattr(args, "objectives", None) is None and getattr(session, "mode", None) != "use":
        return {"on": False, "offered": True, "route": name,
                "why": "one target on a program that ends with this command: no later gradient to keep it for"}
    return {"on": True, "offered": True, "route": name, "why": None, "env": {k: str(v) for k, v in kf["env"].items()}}


def say_forward_not_kept(ctx: Any, why: str | None) -> None:
    """The command's one line on a forward that is not kept between its gradients, and why. THE BREAKAGE THIS
    PREVENTS: a session command on a host too small for the kept forward printed the line twice, once where the host
    memory is judged and once where the gradient line's words are made; and one Python session object printed it
    again for every gradient. The line is said once for a command (the command's context remembers it)."""
    if getattr(ctx, "forward_not_kept_said", False):
        return
    ctx.forward_not_kept_said = True
    print(f"woof hex {getattr(ctx, 'door', None) or 'sensitivity'}: forward not kept between gradients: {why}",
          flush=True)


def keep_forward_words(doc: dict, plan: dict | None) -> str | None:
    """What the loaded program did with the forward of one gradient, in the door's words, from the runner's own
    receipt (window.json keep_forward); None for a command that keeps no forward."""
    if not (plan or {}).get("on"):
        return None
    kf = doc.get("keep_forward")
    if not isinstance(kf, dict):
        return "forward: ran (the runner's receipt names no kept forward for this gradient)"
    if kf.get("used"):
        return ("forward: KEPT, not run again: this gradient starts from the start of the gradient before, so it "
                f"took that forward ({float(kf.get('forward_step_loop_s_not_run') or 0.0):.1f} s of forward steps "
                "not run) and went straight to the backward sweep")
    first = bool((doc.get("loaded_program") or {}).get("first"))
    why = ("this is its program's first gradient" if first else
           "the program held no forward of this start (another start than the gradient before's, by the program's "
           "own fingerprint of the start; or none was kept)")
    after = ("it is kept for the next gradient of the same forecast" if kf.get("keeping")
             else f"it is not kept ({kf.get('why_not') or 'the runner did not say why'})")
    return f"forward: ran its own ({why}); {after}"


def route_window_warning(route: dict, steps: int, dt_s: float | None) -> str | None:
    """The one-line warning (never a refusal) of a route asked for a window longer than the length it was measured
    for (table routes.<route>.window_limit_s); None inside it."""
    limit = route.get("window_limit_s")
    if not limit or not dt_s or int(steps) * float(dt_s) <= float(limit):
        return None
    return (f"{route.get('window_limit_warning') or 'this window is longer than the route was measured for'} (asked: "
            f"{window_length_words(int(steps) * float(dt_s))})")


def _free_disk_mib(folder: Path) -> int | None:
    """Free space of the disk ``folder`` is (or would be made) on, in MiB."""
    probe = Path(os.path.abspath(folder))
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        return int(shutil.disk_usage(probe).free // (1 << 20))
    except OSError:
        return None


def refuse_disk_too_small(row_name: str, need: dict | None, folder: Path, free: int | None = None) -> int | None:
    """Stop, before any download, a window whose run folder's disk cannot take the files it writes there; returns
    the free MiB read. A row with no measured boundary size is not judged here."""
    free = _free_disk_mib(folder) if free is None else free
    if need is None or free is None or free >= int(need["disk_mib"]):
        return free
    raise DiskSpaceError(f"the run folder {folder} is on a disk with {free:,} MiB free and row {row_name}'s window "
                         f"writes about {int(need['disk_mib']):,} MiB there ({need['how']}), so it would stop part "
                         "way through: give --out on a disk with that much free, or free space there. Nothing but "
                         "the row's mesh file was downloaded")


def host_peak_mib() -> int:
    """The largest resident memory of this door or of any runner it waited for, in MiB (getrusage ru_maxrss, KiB on
    Linux; for the children it is the largest single child, which is the runner).

    ``resource`` is POSIX only and imported here: at the top of the module it stopped ``--help`` of every door,
    and every import of this module, on Windows (``No module named 'resource'``)."""
    import resource

    peak = max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    return int(peak) // 1024


#: Seconds a command's own window waits for the card after the command ended its session's program (see
#: Context.window): the program releases its card memory a few seconds after it is told to end.
SESSION_RELEASE_WAIT_S = 120.0


#: Seconds a later window of one command waits for the host memory of the window before it (window_host_admission).
WINDOW_RELEASE_WAIT_S = 30.0


def admit_card(card: str, need: int, wait_s: float, archs: list[str] | None = None) -> dict:
    """Refuse a card the row's images cannot load on; wait (up to wait_s) for need MiB free, else refuse."""
    if archs:
        cc = compute_capability(card)
        if cc is not None and cc not in archs:
            from .ad import rows as _rows
            shipped = [_rows.arch_of(a) for a in archs]
            kinds = ", ".join(f"{_rows.CARD_FAMILIES[a][0]} ({_rows.CARD_FAMILIES[a][1]})" if a in _rows.CARD_FAMILIES
                              else "another family" for a in shipped)
            raise AdDoorError(f"card {card} is {_rows.family_words(_rows.arch_of(cc))}; this row's derivative images "
                              f"are built for compute capability {', '.join(archs)} only ({kinds}): run it on such a "
                              "card")
    refuse_card_too_small(card, need)
    started = time.monotonic()
    while True:
        free = free_mib(card)
        if free is None:
            raise AdDoorError(f"cannot read card {card}'s free memory (nvidia-smi); the window runs on the GPU only")
        if free >= need:
            return {"card": card, "free_mib": free, "need_mib": need, "waited_s": round(time.monotonic() - started, 1)}
        if time.monotonic() - started >= wait_s:
            raise AdDoorError(f"card {card} has {free} MiB free; this row's window needs {need} MiB. Free the card or "
                              "pass --wait-card SECONDS to wait for it")
        time.sleep(15)


def process_card_mib(pid: int) -> dict[str, int]:
    """Card memory one process holds now, per card (bus id -> MiB), from nvidia-smi's compute-app list."""
    refuse_no_local_gpu()
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_bus_id,pid,used_memory",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return {}
    held: dict[str, int] = {}
    for line in out.stdout.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 3 and parts[1] == str(pid) and parts[2].isdigit():
            held[parts[0]] = held.get(parts[0], 0) + int(parts[2])
    return held


def window_validity(out: Path, doc: dict, argv: list[str], streams: tuple | None = None) -> dict:
    """What the window says about itself, beyond its numbers: every launch kind it ran as identity that the row's
    refuse list does not name (a derivative that silently skips a kernel is not a derivative), and how many step
    boundaries of the replayed forecast equal the recording. ``streams``: the logs to read (default out/stdout and
    out/stderr; a rank's are out/r<k>.stdout and .stderr)."""
    listed = set()
    if "--refuse" in argv:
        listed = {x for x in argv[argv.index("--refuse") + 1].split(",") if x}
    kinds = set()
    for key in (doc.get("refused") or {}):
        name = key[len("REFUSED:"):] if key.startswith("REFUSED:") else key
        kinds.add(name.split(":")[0].strip())
    beyond = sorted(k for k in kinds if k not in listed)
    steps = []
    for stream in (streams or ("stdout", "stderr")):
        path = out / stream
        if not path.is_file():
            continue
        for line in path.read_text(errors="replace").splitlines():
            if line.startswith("{") and '"window_tangent_step"' in line:
                try:
                    steps.append(json.loads(line))
                except ValueError:
                    pass
    checked = [s for s in steps if s.get("reference_check")]
    equal = sum(1 for s in checked if s.get("boundary_hashes") == "equal")
    ok = not beyond and equal == len(checked)
    out = {"refused_kinds": len(kinds), "refused_by_row_list": len(kinds) - len(beyond),
           "refused_beyond_row_list": beyond, "boundaries_checked": len(checked), "boundaries_equal": equal,
           "valid": ok,
           "what": "valid: no launch kind ran as identity beyond the row's refuse list, and every checked step "
                   "boundary of the replayed forecast equals the recording"}
    # The runner tallies launch kinds (commands_by_rule, refused) only while a tangent forward runs (window.rs: the
    # label count is on for the tangent pass and off before the reverse). A window with --no-tangent 1 (sensitivity,
    # param-grad) runs the refused kinds as identity in its reverse all the same, and an unlisted refusal still stops
    # it, but its window.json reads refused {}, refuse_list_kinds 0 and complete_operator true. THE BREAKAGE THIS
    # PREVENTS: a 120 km sensitivity receipt (2026-11-11) read 0 refused kinds and a complete operator
    # beside a tangent receipt with 64 on the same --refuse list. Such a window's count is "not counted", not 0.
    if "--no-tangent" in argv and argv[argv.index("--no-tangent") + 1] == "1":
        out.update({"refused_kinds": None, "refused_by_row_list": None, "refused_counted": False,
                    "refused_listed": len(listed), "runner_complete_operator_counted": False,
                    "refused_why": "no tangent forward ran in this window, and the runner counts launch kinds only "
                                   "in a tangent forward: the refuse list was passed and applied (refused_listed "
                                   "names given), the kinds that ran as identity were not counted, and the runner's "
                                   "complete_operator true says only that nothing was counted"})
    return out


def window_command(table: dict, row: dict, assets: dict[str, Path], door: str, runner: Path, steps: int,
                   out: Path, extra: list[str], env_overrides: dict | None = None,
                   door_args_key: str = "window_args", extra_env: dict | None = None,
                   spec_door: str | None = None, replace_args: dict | None = None) -> tuple[list[str], dict]:
    """The runner's line and environment for one window. ``replace_args``: flag -> value set in place of the row's
    value of that flag (or added when the line has none): the gradient line's tape and scratch sizes."""
    door_spec = table["doors"][spec_door or door]   # spec_door: a window another door's table entry defines
    for key in sorted(set(row["asset_args"].values()) | set(row["asset_env"].values()) | set(
            door_spec.get("asset_env", {}).values()) | set((env_overrides or {}).values())):
        if key not in assets:
            raise AdDoorError(f"row asset {key} is not named by this row: the {spec_door or door} door hands it to "
                              "the runner")
        if not assets[key].exists():
            raise AdDoorError(f"row asset {key} ({assets[key]}) is missing from the asset root")
    argv = [str(runner), "window", "--config", "plan.json", "--steps", str(steps), "--out", str(out),
            "--bstore", str(out / "bstore")]
    for flag, key in row["asset_args"].items():
        argv += [flag, str(assets[key])]
    argv += list(row["window_args"]) + list(door_spec[door_args_key]) + list(extra)
    for flag, value in (replace_args or {}).items():
        if flag in argv[2:]:
            at = argv.index(flag, 2)
            argv[at + 1] = str(value)
        else:
            argv += [flag, str(value)]
    live_max = row.get("fct_live_max_s")
    made = row.get("generated") or {}
    if (live_max is None and made.get("from_row") and "--fct-frozen" in argv[2:]
            and ((made.get("window_class") or {}).get("mesh_kind") == "limited-area")):
        # a made limited-area row runs the limiter live up to its template's fct_live_max_s, also when it (and a plan
        # recorded with it) was made before the template said so. THE BREAKAGE THIS PREVENTS: the 6 km made row's
        # 29-step whole-state self-dot read +5.47 roundings with the limiter held (water-vapour unit 9.19e-6 off) and
        # passes with it live (1.6e-8), and a plan recorded before 0.6.2 would keep holding it
        live_max = (load_table()["rows"].get(made["from_row"]) or {}).get("fct_live_max_s")
        if live_max is not None:
            argv[argv.index("--fct-frozen", 2) + 1] = "0"
    if live_max is not None and "--fct-frozen" in argv[2:] and steps * float(row.get("dt_s") or 0) > float(live_max):
        # THE BREAKAGE THIS PREVENTS: the limiter's exact derivative grows where the forecast oscillates from step to
        # step (measured on the 3 km example past about 30 minutes), so a row that runs it live holds it on longer windows.
        at = argv.index("--fct-frozen", 2)
        argv[at + 1] = "1"
    refuse = assets.get("refuse")
    if refuse is not None:
        # A row that names a refuse list needs the file (an empty file when it holds no launch kind by name).
        # THE BREAKAGE THIS PREVENTS: an asset root put together by hand without the file ran its window with no
        # --refuse. The runner then holds nothing by name: of the 64 kinds the made rows' list holds (radiation and
        # its cloud inputs), 50 have a derivative row in the kernel set and 2 in the scheme bundle, so they run live,
        # an operator no check has seen, while the receipt still carries the row's words "radiation is held", an
        # empty refused list and no refused kind beyond it, and reads valid when the strict dot passes.
        if not refuse.is_file():
            raise AdDoorError(f"row asset refuse ({refuse}) is missing from the asset root: the row holds launch "
                              "kinds by the names in that file (an empty file when it holds none), and its window "
                              "does not run without it")
        names = refuse.read_text(encoding="utf-8").strip()
    else:
        names = ""
    # A row that holds a scheme's derivative by name (derivative_holds) adds those launch kinds to the refuse list.
    # THE BREAKAGE THIS PREVENTS: a 12 km plan recorded with New Tiedtke on ran the cumulus kinds as "no derivative
    # row and no rule", beyond the row's list, so the window read invalid instead of saying the scheme is held.
    held = [k for h in (row.get("derivative_holds") or {}).values() for k in h.get("kinds", [])]
    listed = [x for x in names.split(",") if x]
    names = ",".join(listed + [k for k in held if k not in listed])
    if names:
        argv += ["--refuse", names]
    env = dict(row.get("window_env", {}))
    for var, key in {**row["asset_env"], **door_spec.get("asset_env", {}), **(env_overrides or {})}.items():
        env[var] = str(assets[key])
    env.update({k: str(v) for k, v in (extra_env or {}).items()})
    return inner_model_words(argv, env)


#: inner-model (0.6.2): the inner loop's linear model is one word with three values (18 Nov 2026):
#: full (the default: the row's own recorded derivative, the window unchanged bit for bit), reduced (the full
#: derivative with pieces left out by name; INNER_MODEL_KEEP_ENV lists the pieces kept) and simple (the purpose-built
#: form: the boundary layer as a linear diffusion with MYNN's own diffusivities from the trajectory held, everything
#: else but the dynamics held at the trajectory's tendencies). Carried to window_command in the route environment.
INNER_MODEL_ENV = "AD_IM_MODEL"
INNER_MODEL_KEEP_ENV = "AD_IM_KEEP"
INNER_MODELS = ("full", "reduced", "simple")
INNER_MODEL_LINES = {
    "reduced": "linear model: reduced (--inner-model reduced): the row's own derivative of the dynamics and of the "
               "pieces kept ({keep}); every piece left out ({out}) held at the trajectory's tendencies. It is an "
               "inner loop's linear model, not the forecast's derivative",
    "simple": "linear model: simple (--inner-model simple): the dynamics' exact derivative; the boundary layer as a "
              "linear diffusion with MYNN's own diffusivities (el sqrt(qke) sh, sm) and the surface coefficients of "
              "the trajectory held; cloud and rain, radiation, gravity-wave drag, land and sea ice held at the "
              "trajectory's tendencies. It is an inner loop's linear model, not the forecast's derivative",
}
INNER_MODEL_LINE = INNER_MODEL_LINES["simple"]
#: The launch kinds each piece holds by name when it is left out (the dynamics cannot be left out; the surface layer
#: is held on every route of this row already). Microphysics is TEMPO (--hold tempo); the boundary layer is held by
#: the inner-model route with no mixing (AD_IM_BL=held).
INNER_MODEL_PIECES = {
    "radiation": ("rlw_cldprmc_coalesced,rlw_inatm,rlw_rtrn_accum_rows,rlw_rtrn_final,rlw_rtrn_march_coalesced,"
                  "rlw_rtrn_prol_gpoints,rlw_rtrn_secdiff,rlw_setcoef,rlw_taut,rlw_taumol_batched,rsw_cldprmc_b,"
                  "rsw_inatm_layers_b,rsw_laysolfr_b,rsw_post_b,rsw_setcoef_b,rsw_sfluxzen_b,rsw_spc_accum_b,"
                  "rsw_spcvmc_gpt_b,rsw_taumol_b,mpas_lw_mcica_gpoint,mpas_lw_mcica_seeds,mpas_lw_outputs,"
                  "mpas_lw_prepare,mpas_sw_aerosol_columns,mpas_sw_aerosol_finish,mpas_sw_aerosol_levels,"
                  "mpas_sw_diffuse,mpas_sw_driver_times,mpas_sw_gt_aod,mpas_sw_mcica_gpoint,mpas_sw_mcica_seeds,"
                  "mpas_sw_outputs,mpas_sw_prep,mpas_sw_radconst,cloud_condensate,cloud_from").split(","),
    "drag": ["ugwp_oro_run", "ugwp_oro_dx", "ugwp_oro_diff"],
    "land": ["oracle_land_ruc_land", "oracle_drivers_driver_forcing"],
    "ice": ["oracle_ice_ruc_ice", "oracle_drivers_correct_tsk_over_seaice", "oracle_drivers_seaice_finish",
            "oracle_drivers_seaice_prepare"],
    "microphysics": [],
    "boundary-layer": [],
}
#: reduced's default kept list: tonight's measured form (dynamics + boundary layer + surface layer; MEASURED
#: Table 4). The dynamics and the surface layer are not in the list (always as the row has them).
INNER_MODEL_KEEP_DEFAULT = ("boundary-layer",)
#: The launch kinds the simplified inner model holds by name (radiation and its cloud inputs; drag, land, sea ice and
#: their drivers), beside TEMPO (--hold tempo) and the boundary layer (AD_IM_BL=fixedk).
INNER_MODEL_HELD = [k for p in ("radiation", "drag", "land", "ice") for k in INNER_MODEL_PIECES[p]]


def inner_model_keep(text: str | None) -> tuple[str, ...]:
    """The pieces reduced keeps, from --inner-model-keep (comma list; empty keeps only the dynamics)."""
    if text is None:
        return INNER_MODEL_KEEP_DEFAULT
    keep = tuple(x.strip() for x in text.split(",") if x.strip())
    bad = [x for x in keep if x not in INNER_MODEL_PIECES]
    if bad:
        raise AdDoorError(f"--inner-model-keep {','.join(bad)}: the pieces are {', '.join(INNER_MODEL_PIECES)} "
                          "(the dynamics and the surface layer are always kept)")
    return keep


def _hold_words(argv: list[str], hold_tempo: bool, refuse: list[str]) -> list[str]:
    out, skip, hold = [], 0, []
    for i, w in enumerate(argv):
        if skip:
            skip -= 1
            continue
        if hold_tempo and i >= 2 and w in ("--tempo", "--tempo-arena-gib", "--tempo-cubin", "--hold"):
            if w == "--hold" and i + 1 < len(argv):
                hold += [x for x in argv[i + 1].split(",") if x]
            skip = 1
            continue
        out.append(w)
    if hold_tempo:
        out += ["--hold", ",".join(hold + [x for x in ("tempo",) if x not in hold])]
    if refuse:
        if "--refuse" in out[2:]:
            at = out.index("--refuse", 2)
            have = [x for x in out[at + 1].split(",") if x]
            out[at + 1] = ",".join(have + [k for k in refuse if k not in have])
        else:
            out += ["--refuse", ",".join(refuse)]
    return out


def inner_model_words(argv: list[str], env: dict) -> tuple[list[str], dict]:
    """The runner's line and environment for the inner loop's linear model (``--inner-model full|reduced|simple``).
    full (the default, or the word absent) returns its arguments untouched, so every default window is the one it was
    before."""
    model = env.get(INNER_MODEL_ENV, "full")
    if model == "reduced":
        keep = inner_model_keep(env.get(INNER_MODEL_KEEP_ENV))
        env = {k: v for k, v in env.items() if k not in (INNER_MODEL_ENV, INNER_MODEL_KEEP_ENV)}
        left = [p for p in INNER_MODEL_PIECES if p not in keep]
        if "boundary-layer" in left:
            env = {k: v for k, v in env.items() if not k.startswith("AD_P3_DIRECT")}
            env["AD_IM_BL"] = "held"
        refuse = [k for p in left for k in INNER_MODEL_PIECES[p]]
        return _hold_words(argv, "microphysics" in left, refuse), env
    if model != "simple":
        return argv, env
    env = {k: v for k, v in env.items() if not k.startswith("AD_P3_DIRECT")}   # the tape-free MYNN route is not run
    env["AD_IM_BL"] = "fixedk"
    return _hold_words(argv, True, list(INNER_MODEL_HELD)), env


def library_path() -> str:
    """Folders holding this install's CuPy CUB/Thrust modules and NVIDIA wheel libraries, for the runner's pinned-library
    lookup (a plan records the recording machine's absolute paths; ``HEXAD_LIBRARY_PATH`` points the runner at the
    same libraries here). A caller's own ``HEXAD_LIBRARY_PATH`` comes first."""
    import importlib.util
    dirs = [d for d in os.environ.get("HEXAD_LIBRARY_PATH", "").split(":") if d]
    cupy = importlib.util.find_spec("cupy")
    if cupy is not None and cupy.submodule_search_locations:
        dirs += [str(Path(loc) / "cuda") for loc in cupy.submodule_search_locations]
    nvidia = importlib.util.find_spec("nvidia")
    if nvidia is not None and nvidia.submodule_search_locations:
        for loc in nvidia.submodule_search_locations:
            dirs += sorted(str(p) for p in Path(loc).glob("*/lib") if p.is_dir())
    return ":".join(dict.fromkeys(dirs))


#: The folder name a packed plan gives the CUDA libraries it replays. A plan records each library operation (CuPy's CUB
#: and Thrust modules, cuBLAS) by path and sha256, and the runner resolves a relative path against the plan folder.
#: THE BREAKAGE THIS PREVENTS: the pinned 120 km example names them "<site-packages>/..." and nothing bound that name,
#: so on a plain install every door stopped in the runner ("cannot resolve cuBLAS library") after the downloads.
PLAN_SITE_PACKAGES = "<site-packages>"


def install_site_packages() -> Path:
    """The folder of this install that holds cupy/ and nvidia/ (where the plan's replayed libraries live)."""
    import importlib.util
    import sysconfig
    cupy = importlib.util.find_spec("cupy")
    if cupy is not None and cupy.submodule_search_locations:
        return Path(list(cupy.submodule_search_locations)[0]).parent
    return Path(sysconfig.get_paths()["platlib"])


def plan_libraries(table: dict, row: dict) -> list[dict]:
    """The libraries the row's example plan replays, from the table (path under site-packages, sha256, where from)."""
    return list((table.get("example_plan_libraries") or {}).get(row.get("example_plan") or "", []))


def _file_sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def refuse_plan_libraries_missing(table: dict, row_name: str, row: dict, site: Path | None = None) -> None:
    """Before any download: the example plan replays these exact library files; say which one this install lacks."""
    libraries = plan_libraries(table, row)
    if not libraries:
        return
    site = install_site_packages() if site is None else site
    for lib in libraries:
        path = site / lib["path"]
        if not path.is_file():
            state = "is not in this install"
        elif _file_sha256(path) != lib["sha256"]:
            state = "is another build here"
        else:
            continue
        need = (table.get("example_plan_install") or {}).get(row.get("example_plan") or "", "")
        raise AdDoorError(f"row {row_name}: its example plan replays {lib['path']} ({lib.get('from', 'a CUDA library')}) "
                          f"byte for byte, and that file {state} ({path}). {need} Nothing was downloaded")


def bind_plan_libraries(plan: Path, site: Path | None = None) -> None:
    """Point the plan's library folder name at this install (a link in the plan folder; see PLAN_SITE_PACKAGES)."""
    site = install_site_packages() if site is None else site
    link = plan / PLAN_SITE_PACKAGES
    try:
        if link.is_symlink():
            if Path(os.readlink(link)) == site:
                return
            link.unlink()
        elif link.exists():
            return   # a real folder the user put there
        link.symlink_to(site, target_is_directory=True)
    except OSError as error:
        with open(plan / "plan.json", "rb") as stream:   # only a plan that names the folder needs the link
            needed = any(b'"' + PLAN_SITE_PACKAGES.encode() + b"/" in block
                         for block in iter(lambda: stream.read(1 << 24), b""))
        if needed:
            raise AdDoorError(f"plan package {plan} names its libraries under {PLAN_SITE_PACKAGES}/ and the door could "
                              f"not make that link there ({error}); copy the plan to a folder you can write") from error


def _child_pids(pid: int) -> list[int]:
    """The processes a runner started (the rank launcher's ranks), from /proc; empty where /proc does not say."""
    try:
        return [int(x) for x in Path(f"/proc/{pid}/task/{pid}/children").read_text().split()]
    except (OSError, ValueError):
        return []


def ranks_validity(out: Path, doc: dict, argv: list[str], n: int) -> dict:
    """window_validity of every rank (its receipt out/r<k>/window.json, its logs out/r<k>.stdout and .stderr, as the
    launcher writes them), joined: the split window is valid only if every rank is."""
    rows = []
    for k in range(n):
        own = json.loads((out / f"r{k}" / "window.json").read_text())
        rows.append(window_validity(out, own, argv, streams=(out / f"r{k}.stdout", out / f"r{k}.stderr")))
    joined = dict(rows[0])
    for key in ("refused_kinds", "refused_by_row_list", "boundaries_checked", "boundaries_equal"):
        values = [r.get(key) for r in rows]
        joined[key] = None if any(v is None for v in values) else sum(values)
    joined["refused_beyond_row_list"] = sorted({x for r in rows for x in r["refused_beyond_row_list"]})
    joined["valid"] = all(r["valid"] for r in rows)
    joined["ranks"] = rows
    joined["what"] = "every rank valid: " + rows[0]["what"]
    return joined


#: Seconds a window's runner may write nothing before the door says so (WOOF_HEX_AD_QUIET_S overrides); then again
#: every QUIET_AGAIN_S while it stays quiet. The runner writes a line per launch it times, every few seconds at most on
#: the shipped rows.
QUIET_S = 120.0
QUIET_AGAIN_S = 300.0


def card_busy_percent(card: str) -> int | None:
    """The card's own busy reading (nvidia-smi utilization.gpu), or None when it cannot be read."""
    try:
        out = subprocess.run(["nvidia-smi", "-i", str(card).split(",")[0], "--query-gpu=utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30)
        return int(out.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def last_runner_step(stderr: Path) -> str:
    """Where the runner's log stops: the pass and step of its last line that names them, and that line's event."""
    try:
        with open(stderr, "rb") as f:
            f.seek(max(0, f.seek(0, os.SEEK_END) - 262144))
            lines = f.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return "no log yet"
    last_event = None
    for line in reversed(lines):
        try:
            doc = json.loads(line)
        except ValueError:
            if line.strip() and last_event is None:
                last_event = line.strip()[:120]
            continue
        if not isinstance(doc, dict):
            continue
        last_event = last_event or doc.get("event") or doc.get("at")
        if doc.get("step") is not None:
            return (f"pass {doc.get('pass') or '?'}, step {doc['step']} ({doc.get('event') or doc.get('at')}"
                    f"{': ' + str(doc['function']) if doc.get('function') else ''})")
        if doc.get("at"):
            return f"{doc['at']} (event {doc.get('event')})"
    return str(last_event or "no line yet")


class QuietWatch:
    """Says, with the time, when a window's runner has written nothing for QUIET_S seconds: where its log stopped,
    how busy the card is and the memory the runner holds. It does not stop the runner.

    THE BREAKAGE THIS ANSWERS: a 0.6.0 build's first adjoint-test on the 12 km cut stopped writing in its third
    window (backward step 12), held the card and kept a processor busy for ten minutes until the caller's own time
    limit ended it; the door printed nothing in that time, and a rerun passed the same window in 73 s. A user saw a
    command that had gone silent, with no word of where or for how long."""

    def __init__(self, stderr: Path, name: str, card: str):
        self.stderr, self.name, self.card = stderr, name, card
        try:
            self.quiet_s = float(os.environ.get("WOOF_HEX_AD_QUIET_S") or QUIET_S)
        except ValueError:
            self.quiet_s = QUIET_S
        self.size, self.since, self.said_at = -1, time.monotonic(), None
        self.notes: list[dict] = []

    def check(self, pid: int, elapsed: float) -> None:
        try:
            size = self.stderr.stat().st_size
        except OSError:
            size = 0
        now = time.monotonic()
        if size != self.size:
            if self.said_at is not None:
                print(f"woof hex: window {self.name}: the runner writes again, {elapsed:.0f} s into the window "
                      f"(it was quiet for {now - self.since:.0f} s)", file=sys.stderr, flush=True)
            self.size, self.since, self.said_at = size, now, None
            return
        quiet = now - self.since
        if quiet < self.quiet_s or (self.said_at is not None and now - self.said_at < QUIET_AGAIN_S):
            return
        self.said_at = now
        busy = card_busy_percent(self.card)
        held = sum(process_card_mib(pid).values())
        where = last_runner_step(self.stderr)
        self.notes.append({"elapsed_s": round(elapsed, 1), "quiet_s": round(quiet, 1), "last": where,
                           "card_busy_percent": busy, "runner_card_mib": held})
        print(f"woof hex: window {self.name}: the runner (pid {pid}) has written nothing for {quiet:.0f} s, "
              f"{elapsed:.0f} s into the window; its log stops at {where}; card {self.card}: "
              f"{'busy reading n/a' if busy is None else f'{busy}% busy'}, {held:,} MiB held by the runner. Still "
              "waiting (this is not the usual pace: the runner writes a line every few seconds); Ctrl-C stops the "
              "command", file=sys.stderr, flush=True)


def refuse_runner_without_plan_rules(runner: Path, plan: Path) -> None:
    """Refuse, before the card is taken, a window whose runner lacks a window rule the plan's sealed coverage relies on
    (woof.hex.ad.record.coverage.RUNNER_RULE_KERNELS names the breakage: 0.6.2 plans on the 0.6.1 runner stopped at
    their first model-top damping launch with "no derivative row and no rule"). Plans sealed before the coverage
    file named its rule kinds carry no "rule_kinds" and pass."""
    cov = Path(plan) / "derivative-coverage.json"
    if not cov.is_file() or not runner.is_file():
        return
    try:
        kinds = json.loads(cov.read_text()).get("rule_kinds") or []
    except (OSError, ValueError):
        return
    from .ad.record import coverage as _coverage
    lacks = _coverage.runner_lacks(kinds, runner)
    if lacks:
        raise AdDoorError(f"the window runner {runner} has no window rule for {', '.join(lacks)}, which this plan "
                          "launches (rules of 0.6.2): name a runner built from 0.6.2 or later with --runner or "
                          "WOOF_HEX_AD_RUNNER")


def run_window(argv: list[str], env: dict, plan: Path, out: Path, card: str, ranks: int | None = None,
               exchange_label: str | None = None) -> dict:
    """One runner process with the plan package as cwd; returns window.json (or names the failure). With ``ranks``
    the process is the rank launcher (ad-binder ranks): its N ranks write out/r<k>, and the receipt returned is
    the one the door composes from theirs (woof.hex.ad.ranks.compose_window_doc)."""
    refuse_no_local_gpu()
    refuse_runner_without_plan_rules(Path(argv[0]), plan)
    from .ad import rows as _rows
    _rows.forget_plan_docs()   # the parsed plan is not held in this process while the runner holds the host's memory
    out.mkdir(parents=True, exist_ok=True)
    (out / "bstore").mkdir(exist_ok=True)
    refuse_hidden_card()
    refuse_planted_env("window", before_download=False)
    child = {**os.environ, **env, "CUDA_VISIBLE_DEVICES": str(card), "HEXAD_LIBRARY_PATH": library_path()}
    (out / "cmd").write_text(" ".join(f"{k}={v}" for k, v in sorted(env.items())) + " " + " ".join(argv[1:]) + "\n")
    started = time.monotonic()
    peak: dict[str, int] = {}
    quiet = QuietWatch(out / "stderr", out.name, card)
    with open(out / "stdout", "wb") as so, open(out / "stderr", "wb") as se:
        proc = subprocess.Popen(argv, cwd=plan, env=child, stdout=so, stderr=se)
        while True:   # the card memory this process holds, read beside the run (nvidia-smi, every 2 s)
            try:
                rc = proc.wait(timeout=2.0)
                break
            except subprocess.TimeoutExpired:
                quiet.check(proc.pid, time.monotonic() - started)
                if ranks:   # the ranks are the launcher's children, each on its own card
                    now: dict[str, int] = {}
                    for pid in _child_pids(proc.pid):
                        for bus, mib in process_card_mib(pid).items():
                            now[bus] = now.get(bus, 0) + mib
                else:
                    now = process_card_mib(proc.pid)
                for bus, mib in now.items():
                    peak[bus] = max(peak.get(bus, 0), mib)
    wall = time.monotonic() - started
    shutil.rmtree(out / "bstore", ignore_errors=True)
    receipt = out / "window.json"
    if rc != 0:
        logs = [out / "stderr"] + ([out / f"r{k}.stderr" for k in range(ranks)] if ranks else [])
        lines = [x for p in logs if p.is_file() for x in p.read_text(errors="replace").strip().splitlines()]
        tail = [x[:300] for x in lines if not x.startswith("{")][-2:] or [x[:300] for x in lines[-1:]]
        kind = CardMemoryError if any("CUDA status 2" in x for x in lines if not x.startswith("{")) else AdDoorError
        raise kind(f"window runner exited {rc} after {wall:.0f} s: {' | '.join(tail)} (full log {out / 'stderr'}"
                   + (f", ranks' logs {out}/r<k>.stderr" if ranks else "") + ")")
    if ranks:
        from .ad import ranks as _ranks_mod
        try:
            doc = _ranks_mod.compose_window_doc(out, ranks, exchange_label)
        except _ranks_mod.RankSplitError as error:
            raise AdDoorError(str(error)) from error
    else:
        if not receipt.is_file():
            raise AdDoorError(f"window runner exited 0 but wrote no {receipt}")
        doc = json.loads(receipt.read_text())
    doc.setdefault("door_wall_s", wall)
    doc["door_card_peak_mib"] = max(peak.values(), default=0) or None   # the fullest card
    doc["door_card_peaks_mib"] = peak
    if quiet.notes:   # the spells the runner wrote nothing for QUIET_S or more, as the door said them
        doc["door_quiet"] = quiet.notes
    doc["door_validity"] = ranks_validity(out, doc, argv, ranks) if ranks else window_validity(out, doc, argv)
    substituted = [json.loads(x) for x in (out / "stderr").read_text(errors="replace").splitlines()
                   if x.startswith("{") and '"library_substituted"' in x]
    if substituted:   # another build of a pinned library; the replay's boundary hashes held, or the run stopped
        doc["libraries_substituted"] = substituted
    return doc


# ------------------------------------------------------------------------------------------------------ outputs
def mesh_coordinates(geometry: dict | None, grids=("cell", "interface", "edge")) -> dict:
    """The latitude and longitude of the mesh's cell centres (and edge midpoints, when the mesh file has them), in
    degrees, as ``write_mesh_fields`` coordinates: {name: (values, units, long_name, standard_name, grid)}.
    THE BREAKAGE THIS PREVENTS: sensitivity.nc held the gradient on cell and edge numbers only, so a user had to find
    the row's mesh file in the cache to place it (a 0.6.0 build, 2026-11-15)."""
    import numpy as np
    if not geometry:
        return {}
    out = {}
    for grid, place in (("cell", "Cell"), ("edge", "Edge")):
        if grid not in grids and not (grid == "cell" and "interface" in grids):
            continue
        lat, lon = geometry.get(f"lat{place}"), geometry.get(f"lon{place}")
        if lat is None or lon is None:
            continue
        what = "cell centre" if grid == "cell" else "edge midpoint"
        out[f"lat{place}"] = (np.asarray(lat, np.float64), "degrees_north", f"latitude of each {what}", "latitude",
                              grid)
        out[f"lon{place}"] = (np.asarray(lon, np.float64), "degrees_east", f"longitude of each {what} (-180 to 180)",
                              "longitude", grid)
    return out


def write_mesh_fields(path: Path, fields: dict, attrs: dict, coords: dict | None = None) -> None:
    """A classic NetCDF of (nVertLevels|nVertLevelsP1, nCells|nEdges) float32 fields, through woof's Rust writer.
    ``coords`` (``mesh_coordinates``) adds latCell/lonCell (and latEdge/lonEdge) in degrees, and each field names
    the pair on its own grid in its ``coordinates`` attribute."""
    import numpy as np
    from woof.io.nc_writer_bridge import ClassicSchema
    schema = ClassicSchema()
    dims: dict[tuple[str, int], Any] = {}
    horizontal_of = {"cell": "nCells", "interface": "nCells", "edge": "nEdges"}

    def dim(name: str, size: int):
        key = (name, size)
        if key not in dims:
            dims[key] = schema.def_dim(name, size)
        return dims[key]

    for key, value in attrs.items():
        schema.put_global_attr(key, str(value))
    payloads = []
    pairs = {}   # horizontal dimension -> "lonX latX", for the fields' coordinates attribute
    for name, (data, units, what, standard, grid) in (coords or {}).items():
        data = np.ascontiguousarray(data, dtype=np.float64)
        number = schema.def_var(name, "f8", (dim(horizontal_of[grid], data.shape[0]),))
        schema.put_var_attr(number, "units", units)
        schema.put_var_attr(number, "long_name", what)
        schema.put_var_attr(number, "standard_name", standard)
        pairs.setdefault(horizontal_of[grid], []).append(name)
        payloads.append((number, data))
    for name, (data, units, what, grid) in fields.items():
        data = np.ascontiguousarray(data, dtype=np.float32)
        horizontal = horizontal_of[grid]
        vertical = "nVertLevelsP1" if grid == "interface" else "nVertLevels"
        number = schema.def_var(name, "f4", (dim(vertical, data.shape[0]), dim(horizontal, data.shape[1])))
        schema.put_var_attr(number, "units", units)
        schema.put_var_attr(number, "long_name", what)
        if len(pairs.get(horizontal, [])) == 2:
            schema.put_var_attr(number, "coordinates", " ".join(sorted(pairs[horizontal], reverse=True)))
        payloads.append((number, data))
    with schema.create(path) as writer:
        for number, data in payloads:
            writer.write_var(number, data)


def write_cf_plane(path: Path, cp, lat, lon, valid, variable: str, units: str, plane, source: str) -> None:
    """One CF lat-lon plane for the Rust renderer's --cf mode, through woof's Rust NetCDF writer."""
    import numpy as np
    from woof.io.nc_writer_bridge import ClassicSchema
    schema = ClassicSchema()
    t = schema.def_dim("time", 1)
    y = schema.def_dim("lat", lat.shape[0])
    x = schema.def_dim("lon", lat.shape[1])
    schema.put_global_attr("Conventions", "CF-1.8")
    schema.put_global_attr("source", source)
    payloads = []

    def var(name, unit, data, dims, standard=None):
        number = schema.def_var(name, "f4", dims)
        schema.put_var_attr(number, "units", unit)
        if standard:
            schema.put_var_attr(number, "standard_name", standard)
        payloads.append((number, np.ascontiguousarray(cp.asnumpy(data), dtype=np.float32)))

    var("time", "seconds since " + valid.strftime("%Y-%m-%d %H:%M:%S"), cp.asarray([0]), (t,), "time")
    var("lat", "degrees_north", lat[:, 0], (y,), "latitude")
    var("lon", "degrees_east", lon[0], (x,), "longitude")
    var(variable, units, plane[None], (t, y, x))
    with schema.create(path) as writer:
        for number, payload in payloads:
            writer.write_var(number, payload)


def map_scale(values) -> dict:
    """The top of a symmetric colour scale for a cell field: its peak, unless a few cells stand far above the rest.

    Then the scale tops at the 99th percentile of |value| over the cells where the field is (at least 1e-3 of the
    peak; 200 cells or more), so the field's structure is drawn and the extreme cells saturate. THE BREAKAGE THIS
    PREVENTS: one cell 4 to 50 times the field sets the scale and the map is blank but for that cell. The peak and
    the number of cells above the scale are returned for the receipt; nothing is removed from the data."""
    import numpy as np
    mag = np.abs(np.asarray(values, np.float64)).ravel()
    peak = float(mag.max()) if mag.size else 0.0
    if not peak > 0.0:
        return {"top": 1.0, "peak": 0.0, "cells_above_scale": 0, "rule": "empty field"}
    where = mag[mag >= 1e-3 * peak]
    robust = float(np.percentile(where, 99.0)) if where.size >= 200 else peak
    if robust < 0.5 * peak:
        return {"top": robust, "peak": peak, "cells_above_scale": int((mag > robust).sum()),
                "rule": "99th percentile of |value| over the cells within 1e-3 of the peak; cells above it saturate"}
    return {"top": peak, "peak": peak, "cells_above_scale": 0, "rule": "the peak"}


def map_choice(objective: dict, g_theta, g_qv, units: str, asked_level: int | None) -> dict:
    """Which start field's gradient the sensitivity map draws, at which level, and the words that say so.

    A measure of water vapour (qv) draws dJ/dqv; a measure of theta or theta_m draws dJ/dtheta; a measure of any
    other field (cloud water, rain, pressure, density, vertical velocity) draws dJ/dtheta and the line says so (its
    own start field's gradient is in sensitivity.nc). The level is --map-level, else the level that carries most of
    the drawn field (map_level); the line names the measure's own levels beside it.
    THE BREAKAGE THIS PREVENTS: the map of a --metric qv measure over levels 1-5 showed dJ/dtheta at level 12, and
    nothing on the screen said which field or why that level (a 0.6.0 build, 2026-11-15)."""
    field = objective.get("field") or objective.get("metric") or "theta"
    if field == "qv":
        variable, values, start, shown = "dJ_dqv", g_qv, "initial water vapour (qv)", f"{units}/(kg/kg)"
    else:
        variable, values, start, shown = "dJ_dtheta", g_theta, "initial theta", f"{units}/K"
    level = int(asked_level) if asked_level else map_level(values)
    levels = objective.get("levels_1based") or []
    span = (f"{levels[0]}-{levels[-1]}" if levels == list(range(levels[0], levels[-1] + 1)) and len(levels) > 1
            else ",".join(str(k) for k in levels)) if levels else "all"
    picked = "--map-level" if asked_level else f"the level that carries most of {variable.replace('_', '/', 1)}"
    if field in ("theta", "theta_m"):
        # the words every theta measure has printed since 0.5 (README and FIRST-HOUR show them)
        said = f"map: level {level}, the level that carries most of the field; --map-level draws another"
    else:
        said = (f"map: {variable.replace('_', '/', 1)} at model level {level} ({picked}); the measure is of {field} at "
                f"model levels {span}"
                + ("" if field == "qv" else f", whose own start field's gradient is in sensitivity.nc, not drawn")
                + ("; --map-level draws another" if not asked_level else ""))
    return {"variable": variable, "values": values, "level": level, "start": start, "units": shown, "said": said,
            "measure_field": field, "measure_levels": span}


def map_level(field) -> int:
    """The 1-based model level a (levels, cells) field is drawn at: the level that carries the most of it (largest
    root mean square), not the level of one extreme cell."""
    import numpy as np
    return int(np.argmax(np.sqrt((np.asarray(field, np.float64) ** 2).mean(axis=1)))) + 1


MAP_LAM_PROBE_DEG = 3.0   # a 1-degree point farther than this from every cell: the mesh is limited-area
MAP_LAM_MASK = 0.75       # a point farther than this many of its nearest cell's neighbour spacings: outside the mesh
MAP_LAM_MAX_POINTS = 4_000_000


def _unit_vectors(xp, lat_deg, lon_deg, axis):
    a, b = xp.deg2rad(xp.asarray(lat_deg, xp.float64)), xp.deg2rad(xp.asarray(lon_deg, xp.float64))
    return xp.stack((xp.cos(a) * xp.cos(b), xp.cos(a) * xp.sin(b), xp.sin(a)), axis=axis)


def _nearest(xp, target, source, exclude_self: bool = False):
    """Index of, and angle in degrees to, the nearest source column for every target row (chunked dot products)."""
    n = target.shape[0]
    index = xp.empty(n, xp.int64)
    best = xp.empty(n, xp.float64)
    for at in range(0, n, 256):
        dots = target[at:at + 256] @ source
        if exclude_self:
            rows = xp.arange(dots.shape[0])
            dots[rows, rows + at] = -2.0
        index[at:at + 256] = xp.argmax(dots, axis=1)
        best[at:at + 256] = xp.max(dots, axis=1)
    return index, xp.degrees(xp.arccos(xp.clip(best, -1.0, 1.0)))


def map_grid(xp, geometry: dict) -> dict:
    """The lat-lon grid a cell field is drawn on, and the cell each grid point takes its value from.

    A global mesh keeps the 1-degree whole-globe grid it always had (``outside`` and ``bounds`` None). A limited-area
    mesh (some 1-degree point lies more than MAP_LAM_PROBE_DEG from every cell) is drawn on a grid at the mesh's own
    cell spacing (the median distance from a cell to its nearest neighbour) over the mesh's own extent, the view is
    the mesh, and every point farther than MAP_LAM_MASK neighbour spacings from its nearest cell is masked.
    THE BREAKAGE THIS PREVENTS: the 3 km storm box (about 600 km across) was drawn as about 8 x 10 one-degree squares,
    and its field was spread over the whole globe (values of about 1e-9 K/K in the southern hemisphere, where the box
    has no cell; door.out "CF grid=181x360")."""
    import numpy as np
    from .s2s_products import product_grid
    lat = np.asarray(geometry["latCell"], np.float64)
    lon = np.asarray(geometry["lonCell"], np.float64)
    yy, xx, index = product_grid(xp, np.radians(lat), np.radians(lon), 1)
    source = _unit_vectors(xp, lat, lon, 0)
    probe = _unit_vectors(xp, yy.ravel(), xx.ravel(), 1)
    far = float(xp.degrees(xp.arccos(xp.clip((probe * source[:, index].T).sum(axis=1), -1.0, 1.0))).max())
    if far <= MAP_LAM_PROBE_DEG:
        return {"kind": "global", "ylats": yy, "xlons": xx, "index": index, "outside": None, "bounds": None,
                "step_deg": 1.0, "masked_points": 0}
    _, spacing = _nearest(xp, source.T, source, exclude_self=True)
    step = float(xp.median(spacing))
    lon0 = float(np.degrees(np.arctan2(np.sin(np.radians(lon)).mean(), np.cos(np.radians(lon)).mean())))
    rel = (lon - lon0 + 180.0) % 360.0 - 180.0
    south, north = float(lat.min()), float(lat.max())
    west, east = lon0 + float(rel.min()), lon0 + float(rel.max())
    slon = step / max(math.cos(math.radians(0.5 * (south + north))), 0.1)
    while ((north - south) / step + 3) * ((east - west) / slon + 3) > MAP_LAM_MAX_POINTS:
        step, slon = step * 1.25, slon * 1.25
    ylat = xp.arange(math.floor((north - south) / step) + 3, dtype=xp.float64) * step + (south - step)
    xlon = xp.arange(math.floor((east - west) / slon) + 3, dtype=xp.float64) * slon + (west - slon)
    gy, gx = xp.meshgrid(ylat, xlon, indexing="ij")
    near, angle = _nearest(xp, _unit_vectors(xp, gy.ravel(), gx.ravel(), 1), source)
    outside = angle > MAP_LAM_MASK * spacing[near]
    return {"kind": "limited-area", "ylats": gy.astype(xp.float32), "xlons": gx.astype(xp.float32), "index": near,
            "outside": outside, "bounds": f"{west:.2f},{east:.2f},{south:.2f},{north:.2f}", "step_deg": step,
            "masked_points": int(outside.sum())}


def render_cell_map(out: Path, geometry: dict, values, *, variable: str, title: str, units: str, valid: str) -> dict:
    """A cell field drawn as a global map: GPU nearest-cell remap, Rust NetCDF plane, rw_wrfbatch CF renderer."""
    import cupy as cp
    from woof.rustwx import run_cf_renderer
    from datetime import datetime
    from .ocean_products import physical_renderer
    renderer = physical_renderer()
    grid = map_grid(cp, geometry)
    ylats, xlons, index = grid["ylats"], grid["xlons"], grid["index"]
    plane = cp.asarray(values, cp.float32)[index].reshape(ylats.shape)
    if grid["outside"] is not None:   # a limited-area mesh: no value where the mesh has no cell
        plane = cp.where(grid["outside"].reshape(ylats.shape), cp.float32(cp.nan), plane)
    chosen = map_scale(values)
    peak = chosen["top"]
    exponent = math.floor(math.log10(peak))
    scale = 10.0 ** exponent
    top = math.ceil(peak / scale * 10.0) / 10.0
    levels = [round(top * (i - 5.5) / 5.5, 6) for i in range(12)]   # 12 levels, 11 intervals, one colour each
    colors = [[5, 48, 97], [33, 102, 172], [67, 147, 195], [146, 197, 222], [209, 229, 240],
              [247, 247, 247], [253, 219, 199], [244, 165, 130], [214, 96, 77], [178, 24, 43], [103, 0, 31]]
    shown = f"{units} x 1e{exponent}" if exponent else units
    table = out / f"cf-{variable}.json"
    table.write_text(json.dumps([{"names": [variable], "standard_names": [], "product": variable, "title": title,
                                  "default_units": shown, "display_units": shown, "unit_factors": {shown: 1},
                                  "scale": {"levels": levels, "colors": colors}}], indent=1))
    data = out / f"{variable}.nc"
    when = datetime.strptime(valid, "%Y-%m-%dT%H:%M:%S")
    drawn = cp.clip(plane / scale, -0.999 * top, 0.999 * top) if chosen["cells_above_scale"] else plane / scale
    write_cf_plane(data, cp, ylats, xlons, when, variable, shown, drawn, "woof hex sensitivity")
    images = out / f"map-{variable}"
    receipt = out / f"{variable}.render.json"
    if grid["bounds"] is not None:   # a limited-area mesh: the view is the mesh
        bounds = grid["bounds"]
    else:   # a global mesh: the whole globe.
        # THE BREAKAGE THIS PREVENTS: the view used to be cropped to where the field is at least 5 % of its peak, padded
        # 15 degrees. A compact field (a 120 km bump over Iowa, zero elsewhere) gave the view -119.4,-61.7,15.2,66.6,
        # which the Rust renderer (rw_wrfbatch --cf, rustwx c5f33d58) draws in its geographic projection as one flat
        # colour with no field in it, whatever the field: a 120 km orography field is flat in that view too, while the
        # whole globe and a Lambert view of the same plane draw it.
        # A field that spans the globe was drawn on the whole globe already, and still is, byte for byte.
        bounds = "-180,180,-89,89"
    rc = run_cf_renderer(renderer, data, variable=variable, out_dir=images, valid=valid + "Z",
                         bounds=bounds, table=table, size=(1100, 700), receipt=receipt)
    pictures = sorted(str(p.resolve()) for p in images.rglob("*.png"))
    if rc != 0 or not pictures:
        raise AdDoorError(f"the Rust renderer failed for {variable}: rc={rc}, images={pictures}")
    return {"variable": variable, "images": pictures, "plane": str(data), "table": str(table), "bounds": bounds,
            "render_receipt": str(receipt), "scale": f"1e{exponent}", "renderer": str(renderer),
            "grid": {"kind": grid["kind"], "step_deg": grid["step_deg"], "shape": list(ylats.shape),
                     "masked_points": grid["masked_points"]},
            "scale_top": top * scale, "scale_rule": chosen["rule"], "field_peak": chosen["peak"],
            "cells_above_scale": chosen["cells_above_scale"]}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def _init_xtime(path: str | None):
    """The start time an MPAS init file carries (its first xtime), or None."""
    from datetime import datetime
    if not path or not Path(path).is_file():
        return None
    try:
        import numpy as np
        from woof.netcdf_bridge import open_dataset
        with open_dataset(Path(path)) as ds:
            raw = np.asarray(ds.variables["xtime"][:]).reshape(-1)
        text = (b"".join(raw.astype("S1")) if raw.dtype.kind in "SU" and raw.dtype.itemsize == 1 else raw[0])
        text = text.decode() if isinstance(text, bytes) else str(text)
        return datetime.strptime(text.strip("\0 ")[:19], "%Y-%m-%d_%H:%M:%S")
    except (OSError, KeyError, ValueError, IndexError):
        return None


def _valid_time(fmap: FieldMap, steps: int, dt: float) -> str:
    """The window end as YYYY-MM-DDTHH:MM:SS, from the recorded init's start time when the map names it."""
    from datetime import datetime, timedelta
    start = None
    init = fmap.doc.get("init")
    if init:
        name = Path(init).parts
        for part in name:
            try:
                start = datetime.strptime(part, "%Y-%m-%d_%H.%M.%S")
            except ValueError:
                continue
    if start is None:   # a recorder-made plan names its window start (or its init file) beside the allocation map
        for base in (fmap.source.parent, fmap.source.parent.parent):
            try:
                doc = json.loads((base / "from-state.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            try:
                start = datetime.strptime(doc["window_start"], "%Y-%m-%d_%H:%M:%S")
            except (ValueError, KeyError):
                start = _init_xtime(doc.get("init"))
            if start is not None:
                break
    if start is None:   # the door recorder (woof.hex.ad.record) names its init file in record-receipt.json.
        # THE BREAKAGE THIS PREVENTS: sensitivity on a plan the door itself recorded from --init refused with "the plan
        # names no start time" (120 km, 2026-11-11), because only the earlier recorder's
        # from-state.json was read.
        for base in (fmap.source.parent, fmap.source.parent.parent):
            try:
                doc = json.loads((base / "record-receipt.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(doc, dict) and doc.get("init"):
                start = _init_xtime(doc["init"])
            if start is not None:
                break
    if start is None:   # the plan itself names the clock its recording read before step 1 (physics.state_xtime).
        # THE BREAKAGE THIS PREVENTS: a plain 3 km sensitivity on a rented box refused "the plan names no start time"
        # (2026-11-11): the package's from-state.json and record-receipt.json name the init file by a path on the
        # recording machine that is not on the machine that replays it, while plan.json carries the start time.
        # The date is one of the plan's kept facts (woof.hex.ad_plan_facts): read with the plan's one parse, not by
        # parsing plan.json again for every answer.
        for base in (fmap.source.parent, fmap.source.parent.parent):
            if not (base / "plan.json").is_file():
                continue
            try:
                stamp = _plan_facts(base).state_xtime()
                start = datetime.strptime(str(stamp).strip("\0 ")[:19], "%Y-%m-%d_%H:%M:%S")
            except (OSError, ValueError, KeyError, TypeError):
                continue
            break
    if start is None:
        raise AdDoorError(f"the plan names no start time (no init path in {fmap.source}, no from-state.json, "
                          "record-receipt.json or plan.json state_xtime beside it): the map would carry a wrong "
                          "date; pass --no-map")
    return (start + timedelta(seconds=steps * dt)).strftime("%Y-%m-%dT%H:%M:%S")


# ------------------------------------------------------------------------------------------- the gradient's mesh
GRADIENT_MESH_DEFAULT = "auto"


def _native_mesh(args: argparse.Namespace, table: dict, door: str) -> dict | None:
    """The mesh this door's row is on, and its grid file when known, from --mesh, --row or the door's default row,
    without making a row or asking a card. None when it cannot be told (the door then names its own refusal)."""
    from .ad import rows as _rows
    if getattr(args, "mesh", None):
        try:
            found = _rows.find_mesh(args.mesh, Path(args.grid) if getattr(args, "grid", None) else None,
                                    Path(args.oro) if getattr(args, "oro", None) else None)
        except _rows.AdRowError:
            return None
        return {"mesh": found["mesh"], "grid": found["grid"], "row": None}
    every = {**_rows.cached_rows(), **table["rows"]}
    name = getattr(args, "row", None) or table["doors"].get(door, {}).get("default_row") or table["default_row"]
    row = every.get(name)
    if not row:
        return None
    grid = (row.get("assets") or {}).get("grid")
    if getattr(args, "grid", None):
        grid = args.grid
    return {"mesh": row.get("mesh"), "grid": Path(grid) if grid and os.path.isabs(str(grid)) else None, "row": name,
            "grid_rel": str(grid) if grid and not os.path.isabs(str(grid)) else None,
            "example_plan": row.get("example_plan"), "example_plan_start": row.get("example_plan_start")}


def gradient_route(args: argparse.Namespace, door: str) -> dict:
    """Which mesh this door takes the gradient on (--gradient-mesh native|auto|NAME), from the table's
    ``gradient_meshes`` section. ``native``: the mesh itself, today's route unchanged. ``auto``: the mesh's own default
    in the table (a door with no coarse route, or a mesh with no companion, takes the mesh itself). A companion name:
    the coarse route, or a refusal by name, before any card, download or row is made."""
    asked = getattr(args, "gradient_mesh", None) or GRADIENT_MESH_DEFAULT
    if asked == "native":
        return {"carried": False, "asked": asked, "why": "asked for the gradient on the mesh itself (--gradient-mesh "
                                                         "native)"}
    table = load_table()
    section = table.get("gradient_meshes") or {}
    carried_doors = list(section.get("doors_carried") or [])
    native = _native_mesh(args, table, door)
    key, entry = _gm.entry_for(section, (native or {}).get("mesh"))
    if asked == "auto":
        if door not in carried_doors:
            return {"carried": False, "asked": asked, "why": f"auto: the {door} door takes the gradient on the mesh "
                                                             "itself (it has no coarse route)"}
        if entry is None:
            return {"carried": False, "asked": asked, "why": "auto: the table names no companion mesh for this mesh"}
        choice = entry.get("default", "native")
        if choice == "native":
            return {"carried": False, "asked": asked, "why": f"auto: the table's default for mesh {key} is native ("
                                                             f"{entry.get('default_why') or 'no reason given'})"}
        name, how = choice, f"auto: the table's default for mesh {key} is {choice}"
    else:
        name, how = asked, f"asked for (--gradient-mesh {asked})"
        if door not in carried_doors:
            why = (section.get("doors_refused") or {}).get(door) or "this door has no coarse route"
            raise AdDoorError(f"--gradient-mesh {asked}: the {door} door is not carried between meshes: {why}. "
                              "--gradient-mesh native runs it on the mesh itself. Nothing was downloaded")
    if native is None or not native.get("mesh"):
        raise AdDoorError(f"--gradient-mesh {name}: the door cannot tell which mesh the row is on; name it with --mesh "
                          "NAME --grid FILE. Nothing was downloaded")
    if entry is None:
        have = ", ".join(sorted(section.get("by_mesh") or {})) or "none"
        raise AdDoorError(f"--gradient-mesh {name}: no companion row for mesh {native['mesh']} (the table names "
                          f"companion meshes for: {have}); --gradient-mesh native takes the gradient on the mesh "
                          "itself. Nothing was downloaded")
    ckey, comp = _gm.companion(entry, name)
    if comp is None:
        raise AdDoorError(f"--gradient-mesh {name}: mesh {key} has no companion of that name (its companions: "
                          f"{', '.join(sorted(entry.get('companions') or {})) or 'none'}; or native). Nothing was "
                          "downloaded")
    try:
        return _carried_route(args, door, table, key, entry, ckey, comp, native, asked, how)
    except _gm.GradientMeshError as error:
        raise AdDoorError(f"--gradient-mesh {name}: {error}") from error


def refuse_other_start(native: dict, ckey: str, comp: dict, plan: Any, gplan: Any) -> None:
    """Refuse, before any download, the coarse route on a row whose example forecast starts at another time than the
    companion's shipped plan: with no plan given the native window is the example's, and the gradient made on the
    companion would be of another forecast. THE BREAKAGE THIS PREVENTS: sensitivity on the 3 km storm example (HRRR
    2023-03-31 18Z) with --gradient-mesh 6km gave the gradient of the 2024-05-21 15Z forecast, said in one line only
    (2026-11-14). A plan given with --plan is judged against the companion's after download (same start or refused);
    a mesh with no example plan takes the companion's start and says so."""
    if plan is not None or gplan is not None or not native.get("example_plan"):
        return
    ex, co = native.get("example_plan_start"), comp.get("plan_start")
    if ex and co and ex != co:
        raise _gm.GradientMeshError(
            f"the {ckey} route carries a plan that starts at {co.replace('T', ' ')} and this row's example forecast "
            f"starts at {ex.replace('T', ' ')}, so its gradient would be of another forecast: run the example on its "
            f"own mesh (leave out --gradient-mesh), or give --plan with a recording of {co.replace('T', ' ')} on this "
            f"mesh for the {ckey} route. Nothing was downloaded")


def _carried_route(args: argparse.Namespace, door: str, table: dict, key: str, entry: dict, ckey: str, comp: dict, native: dict,
                   asked: str, how: str) -> dict:
    """The coarse route's whole setup, before any card: the target against the shown list, the companion's files
    (asset root or --gradient-plan), its plan's start against the native one, the map checked, and the words of the
    same command on the companion mesh."""
    if getattr(args, "init", None) is not None:
        raise _gm.GradientMeshError("--init records a plan on the native mesh, and the coarse route replays the "
                                    "companion mesh's plan: pass --gradient-plan PACKAGE (a recording of the same "
                                    "start on the companion mesh), or leave both out for the companion's own plan")
    nat = entry.get("native") or {}
    native_plan = Path(args.plan) if getattr(args, "plan", None) else None
    native_dt = (_plan_dt(native_plan) if native_plan else None) or float(nat["dt_s"])
    # the native row's own default window: a shipped row's, else the template a made row is made from
    shipped = table["rows"].get(native.get("row") or "") or {}
    template = table["rows"].get((table.get("suites") or {}).get(getattr(args, "suite", None) or "", {}).get(
        "template_row") or "") or {}
    native_steps = int(args.steps or nat.get("default_steps") or shipped.get("default_steps")
                       or template.get("default_steps") or 1)
    window_s = native_steps * native_dt
    target = {"metric": getattr(args, "metric", None), "box": getattr(args, "box", None),
              "circle": getattr(args, "circle", None), "levels": getattr(args, "levels", None), "window_s": window_s}
    _gm.shown_target(comp, target, check_start=False)   # table only: a target not shown downloads nothing
    files = {k: comp.get(k) for k in ("grid", "oro") if comp.get(k)}
    files["map"] = comp["map"]["file"]
    if not native.get("grid") and native.get("grid_rel"):
        # the native row's grid is a downloaded file (the storm example's mesh): fetched with the companion's files.
        # THE BREAKAGE THIS PREVENTS: sensitivity --row <storm row> --gradient-mesh 6km from a fresh install stopped
        # at "the native mesh's grid file (None) is needed", though the downloads hold it (2026-11-14)
        files["native_grid"] = native["grid_rel"]
    gplan = getattr(args, "gradient_plan", None)
    refuse_other_start(native, ckey, comp, getattr(args, "plan", None), gplan)
    if gplan is None and comp.get("plan"):
        files["plan"] = comp["plan"]
    refuse_row_not_in_pinned_assets(f"{key} companion {ckey}", {"assets": files}, set(files), args.assets)
    if gplan is None and not comp.get("plan"):
        raise _gm.GradientMeshError(f"companion {ckey} ships no plan: pass --gradient-plan PACKAGE (a recording of "
                                    f"the same start on mesh {comp['mesh']})")
    root = resolve_assets(args.assets, want=[str(v) for v in files.values()])   # the files this route reads only
    coarse_plan = Path(os.path.abspath(gplan if gplan is not None else root / comp["plan"]))
    if not (coarse_plan / "plan.json").is_file():
        raise _gm.GradientMeshError(f"the companion's plan package {coarse_plan} has no plan.json"
                                    + ("" if gplan is not None else " (pass --gradient-plan PACKAGE)"))
    start_native, start_coarse = _gm.plan_start(native_plan), _gm.plan_start(coarse_plan)
    if native_plan is not None and start_native != start_coarse:
        raise _gm.GradientMeshError(f"the native plan starts at {start_native} and the companion's plan at "
                                    f"{start_coarse}: a gradient of another start is not this window's")
    target["start"] = start_coarse
    shown = _gm.shown_target(comp, target, check_start=True)
    start_note = None
    if native_plan is None:   # no native plan given: the window is the companion plan's, and the receipt says so
        start_note = (f"no native plan was given, so the gradient is of the {ckey} plan's start, {start_coarse}"
                      + (f"; the row's example plan ({native['example_plan']}) is not used: the {ckey} route was "
                         "shown on that start only" if native.get("example_plan") else ""))
        print(f"woof hex {door}: {start_note}", flush=True)
    grid = root / comp["grid"] if comp.get("grid") and (root / comp["grid"]).is_file() else None
    oro = root / comp["oro"] if comp.get("oro") and (root / comp["oro"]).is_file() else None
    native_grid = (Path(native["grid"]) if native.get("grid")
                   else root / native["grid_rel"] if native.get("grid_rel") else None)
    if native_grid is None or not native_grid.is_file():
        raise _gm.GradientMeshError(f"the native mesh's grid file ({native_grid}) is needed to carry the gradient "
                                    "to it: pass --grid FILE")
    if nat.get("levels") and comp.get("levels") and int(nat["levels"]) != int(comp["levels"]):
        raise _gm.GradientMeshError(f"the companion mesh has {comp['levels']} levels and the native mesh "
                                    f"{nat['levels']}: the map carries a field level by level")
    op = _gm.load_map(root / comp["map"]["file"], rows=int(nat["cells"]), cols=int(comp["cells"]),
                      sha256=comp["map"].get("sha256"))
    coarse_dt = _plan_dt(coarse_plan) or float(comp["dt_s"])
    steps = _gm.coarse_steps(window_s, coarse_dt)
    cargs = argparse.Namespace(**vars(args))
    cargs.row, cargs.mesh, cargs.grid, cargs.oro = None, comp["mesh"], grid, oro
    cargs.plan, cargs.init, cargs.dt, cargs.steps = coarse_plan, None, None, steps
    cargs.assets = root if args.assets is None and not os.environ.get("WOOF_HEX_AD_ASSETS") else args.assets
    cargs.out = Path(args.out) / f"made-on-{ckey}"
    cargs.gradient_mesh = "native"
    if hasattr(cargs, "no_map"):
        cargs.no_map, cargs.map_level = True, None   # the map is drawn once, on the native mesh
    return {"carried": True, "asked": asked, "how": how, "name": ckey, "native_key": key, "entry": entry,
            "companion": comp, "shown": shown, "map": op, "coarse_args": cargs, "door": door,
            "native": {"mesh": key, "cells": int(nat["cells"]), "levels": int(nat.get("levels") or 0),
                       "grid": str(native_grid), "plan": str(native_plan) if native_plan else None,
                       "plan_start": start_native, "steps": native_steps, "dt_s": native_dt, "window_s": window_s,
                       "start_note": start_note},
            "coarse": {"mesh": comp["mesh"], "cells": int(comp["cells"]), "plan": str(coarse_plan),
                       "plan_start": start_coarse, "steps": steps, "dt_s": coarse_dt, "window_s": steps * coarse_dt}}


def carried_block(route: dict, ctx: "Context | None" = None) -> dict:
    """The receipt's gradient_mesh block of a carried gradient: what was asked, the mesh it was made on, the native
    mesh it was carried to, how, and where its answer check was made."""
    comp, op = route["companion"], route["map"]
    made = dict(route["coarse"], name=route["name"])
    if ctx is not None:
        made.update(row=ctx.row_name, steps=ctx.steps, dt_s=ctx.row["dt_s"], window_s=ctx.steps * ctx.row["dt_s"])
    return {"asked": route["asked"], "how": route["how"], "carried": True,
            "made_on": made, "carried_to": dict(route["native"]),
            "method": comp["method"],
            "map": {k: op[k] for k in ("path", "sha256", "rows", "cols", "nnz")},
            "answer_check": f"made on the coarse operator (mesh {comp['mesh']}"
                            + (f", row {ctx.row_name}" if ctx is not None else "")
                            + "): the strict check in this receipt is that operator's. No check of the native "
                              "operator's gradient was made, and the carried field is not one",
            "shown_on": _gm.describe(route["shown"]),
            "scope": comp.get("scope")}


# -------------------------------------------------------------------------------------------------------- doors
class Context:
    def __init__(self, args: argparse.Namespace, door: str):
        from .ad import rows as _rows
        self.args, self.door = args, door
        refuse_strict_words(args)   # opposite strict-check words: refused before any card or download
        self.table = load_table()
        # the mode the linear model is of: a mode with no derivative rows is refused before any card or download
        self.physics_mode = refuse_physics_mode(args, door, self.table)
        # the row cache: rows made on demand earlier (another mesh, another card) answer --row like shipped rows
        self.table["rows"] = {**_rows.cached_rows(), **self.table["rows"]}
        try:
            if getattr(args, "mesh", None):
                if args.row:
                    raise AdDoorError("--row and --mesh are alternatives: a row is named or made for a mesh")
                mesh = _rows.find_mesh(args.mesh, Path(args.grid) if args.grid else None,
                                       Path(args.oro) if getattr(args, "oro", None) else None)
                name, row = _rows.make_row(self.table, args.suite, mesh, init=args.init, dt=args.dt,
                                             own_case=bool(args.init or getattr(args, "plan", None)),
                                             door=door, card_mib=_first_card_total(args),
                                             ranks=record_ranks_of(args))
                self.table["rows"][name] = row
                self.row_name, self.row = _row(self.table, name, door)
            else:
                self.row_name, self.row = _row(self.table, args.row, door)
            # a row from the cache (--mesh or --row) asks the card for at least what its class in the table asks
            self.row = _rows.with_class_floor(self.table, self.row)
            refuse_steps_beyond_example(self.row_name, self.row, door_steps(self.row, door, args), args)
            refuse_no_local_gpu()   # after the table's own (card-free) refusals, before any card, runner or asset
            self.runner = resolve_runner(args.runner)
            self.steps = door_steps(self.row, door, args)
            # a made row's live default boundary layer runs up to its template's length, the held closure past it; the
            # timestep is the given plan's when it names one (the made row follows the recording, below)
            try:
                plan_dt = _plan_dt(Path(args.plan)) if (getattr(args, "plan", None) and self.row.get("generated")) else None
            except Exception:   # an unreadable plan is refused, with its own words, where it is resolved
                plan_dt = None
            boundary_layer_for_window(args, self.row, self.steps, plan_dt or self.row.get("dt_s"), door)
            want = int(self.row.get("cards") or 1)
            have = cards_of(args)
            if len(have) < want:
                raise AdDoorError(f"row {self.row_name}: its window runs on {want} cards (the physics columns are "
                                  f"sharded over them and the step tape is streamed); pass --cards A,B,... with "
                                  f"{want} cards (given: {','.join(have)})")
            # A row with a rank split (table rank_splits) and --cards naming N > 1 cards runs its window as N ranks,
            # one a card (ad-binder ranks); every other row, and every one-card door, runs as before.
            self.split = None
            split = _ranks.split_of(self.table, self.row_name)
            if split is None and getattr(args, "rank_plans", None) is not None and len(have) > 1:
                split = _ranks.made_split(self.table, self.row)   # a made row whose rank plans the user gives
            if split is not None and getattr(args, "cards", None) and len(have) > 1:
                self.split = _ranks.choose(split, self.row_name, len(have), door)
                if self.split.get("experimental"):   # table rank_splits.<row>.experimental: said before it runs
                    print(f"woof hex {door}: EXPERIMENTAL: {self.split['experimental']['line']}", flush=True)
                self.row = split_row(self.row, self.split)
                self.cards = list(have)
            else:
                # a row that runs on one card or more takes the cards the user named (--cards A,B), up to its cards_max
                self.cards = have[:max(want, min(len(have), int(self.row.get("cards_max") or want)))]
            self.card = self.cards[0]
            # the named routes this command runs (boundary layer, cloud derivative; routes_on): a route that names its
            # own kernel set or images runs them in the row's place, in every window of the command, on every route
            # door, so adjoint-test checks the linear model sensitivity computes
            self.route_assets = []
            self.routes_on = routes_on(self.row, args, door, self.split)
            if self.routes_on:
                self.row, self.route_assets = route_row(self.row, args)
            # THE ORDER OF THE CHECKS: what the command asked for comes first, the machine's state now comes last.
            # First everything the user can only fix by changing the command, the install or the card itself (the
            # card's family and whole size, a row the pinned downloads lack, no plan, --init without the recorder,
            # an unknown metric, the wrong Python or a missing extra, a bump or region off the mesh); then what is
            # free on this machine at this moment (card memory, host memory, disk, /dev/shm).
            # THE BREAKAGE THIS PREVENTS (a 0.6.0 build on a shared card, 2026-11-15): with a
            # busy card (25 GB free) and a busy host (27 GB available) a Python 3.12 install, an install without the
            # ad extra and a bump off the mesh were each first told to free the card or the host; the user frees a
            # card, runs again, and only then learns the install or the command was wrong.
            cc = self.check_cards(state=False)   # readable, a family the row has images for, not smaller than the need
            self.arch = _rows.arch_of(cc) if cc else None
            # Every refusal that needs no asset root comes before it: a card class the row has no images for, a row
            # whose files the pinned downloads do not hold, a row with no plan to differentiate, and --init on an
            # install without the recorder.
            needed = _needed_assets(self.table, self.row, door)
            if self.arch:
                _rows.refuse_arch_without_images(self.row_name, self.row, self.arch, needed=needed)
                if self.routes_on:
                    refuse_route_card_family(self.row_name, self.row, getattr(args, "boundary_layer", None),
                                             self.arch, self.card)
                    refuse_routes_card_family(self.row_name, self.row, args, self.arch, self.card)
            refuse_row_not_in_pinned_assets(self.row_name, self.row, needed | {"refuse"}, args.assets)
            if self.split and getattr(args, "rank_plans", None) is None:
                refuse_rank_pkg_not_pinned(self.row_name, self.split["pkg"], args.assets)
            if (getattr(args, "plan", None) is None and getattr(args, "init", None) is None
                    and not self.row.get("example_plan")):
                raise AdDoorError(no_plan_words(self.row_name))
            refuse_init_without_recorder(args)   # --init without the recorder: known before any download
            metric = getattr(args, "metric", None)   # a name the table can judge is judged before any download
            if metric is not None and metric not in self.table["metrics"]:
                raise AdDoorError(f"--metric {metric!r} is not in the table (metrics: "
                                  f"{', '.join(sorted(self.table['metrics']))})"
                                  f"{metric_not_offered(self.table, metric)}. Nothing was downloaded")
            if getattr(args, "plan", None) is None and getattr(args, "init", None) is None:
                refuse_plan_libraries_missing(self.table, self.row_name, self.row)
            # the bump and the metric region on the row's mesh alone (a few MB), before the plan and the row files
            refuse_off_mesh(self.table, self.row_name, self.row, args, door,
                            early_geometry(args, self.row, self.row_name))
            # The machine's state, last. The session first: a command that keeps none stops the program found loaded
            # on its card, and only a command that passed every check above does that.
            from . import ad_session as _session   # the session on this card: this command's first line
            self.session = _session.open_for(self, defer=True)   # its line: once the checks below pass
            self.check_cards(state=True)   # the need is free now (or --wait-card waits for it at the window)
            # after this command stopped the program found loaded: the memory that program is giving back
            self.host_available = (host_available_settled() if getattr(self.session, "stopped", None)
                                   else host_available_mib())
            # what the window's step store will use here (host memory and the run folder's disk), from the row's
            # measured boundary size and the route this door will run; stated before it runs, refused if short
            line_on = bool(self.row.get("gradient_line")
                           and (self.table["doors"].get(door) or {}).get("gradient_line")
                           and not getattr(args, "checked_forward", False) and not self.split)
            self.line_on = line_on
            self.check_expected, self.check_expected_why = answer_check_expected(self.row_name, self.steps, args,
                                                                                 door)
            self.keep_forward = keep_forward_plan(self.row, args, line_on, self.session)
            self.store_need = store_need(self.row, self.steps, self.host_available, line_on, self.check_expected,
                                         keep_forward=self.keep_forward["on"])
            if self.keep_forward["on"]:
                without = store_need(self.row, self.steps, self.host_available, line_on, self.check_expected) or {}
                sn = self.store_need or {}
                why = None
                if not sn.get("keep_forward_mib"):
                    why = ("the window's checkpoints do not all fit in host memory here, and the program keeps a "
                           "forward only when they do")
                elif (not getattr(args, "need_host_mib", None) and self.host_available is not None
                      and sn.get("host_mib") and without.get("host_mib")
                      and int(sn["host_mib"]) > self.host_available >= int(without["host_mib"])):
                    why = (f"this host has {self.host_available:,} MiB available: enough for the gradient "
                           f"({int(without['host_mib']):,} MiB) and not for the gradient with its forward kept "
                           f"({int(sn['host_mib']):,} MiB)")
                if why:
                    self.keep_forward = {**self.keep_forward, "on": False, "why": why}
                    self.store_need = without or self.store_need
                    say_forward_not_kept(self, why)
            sn = self.store_need or {}
            every = bool(sn.get("host_for")) and sn.get("line") is False   # a window that keeps every step boundary
            refuse_host_too_small(
                self.row_name, self.need_host_mib(), self.host_available,
                what=(sn["host_for"] if every else
                      f"{sn['host_for']}: {self.check_expected_why}" if sn.get("host_for") else None),
                advice=(smallest_host_words(sn) if every and not getattr(args, "need_host_mib", None) else None),
                smaller=(None if every else
                         f"The gradient alone needs about {int(sn['host_gradient_mib']):,} MiB: --no-self-dot runs "
                         "it without its answer check (the receipt then reads not checked)."
                         if sn.get("host_for") and self.check_expected and not getattr(args, "need_host_mib", None)
                         and self.host_available is not None and sn.get("host_gradient_mib")
                         and self.host_available >= int(sn["host_gradient_mib"]) else None))
            self.disk_free = refuse_disk_too_small(self.row_name, self.store_need, Path(args.out))
            if self.split:
                _ranks.refuse_shm(self.row_name, self.split["shm_mib"], _ranks.shm_free_mib())
            self.check_card_words()   # a class budget the card or host cannot give: refused before any download
            self.session.said()   # the card, host-memory and disk checks passed: the session's line is true now
            print(self.needs_line(), flush=True)
            self.root = resolve_assets(args.assets, want=example_reads(self.row, args, self.routes_on))
            self.assets = row_assets(self.root, self.row)
            tempo_image_word(self, args)   # --tempo-image f08: the taped TEMPO image with the PBL radii path
            self.route_env, self.route_block = {}, None
            if self.routes_on:   # each file a route names is the pinned build; a scheme bundle a route lays images over
                check_route_assets(self)
                if not line_on:   # a gradient line carries the route's words itself (gradient_line_words)
                    self.route_env, self.route_block = route_window_words(self, door)
            model = getattr(args, "inner_model", "full")
            if model != "full":   # inner-model: full (the row's own derivative) unless another is asked for by name
                if door not in ROUTE_DOORS:
                    raise AdDoorError(f"--inner-model {model} runs on {', '.join(ROUTE_DOORS)}, not on {door}")
                words = {INNER_MODEL_ENV: model}
                line = INNER_MODEL_LINES[model]
                if model == "reduced":
                    keep = inner_model_keep(getattr(args, "inner_model_keep", None))
                    words[INNER_MODEL_KEEP_ENV] = ",".join(keep)
                    line = line.format(keep=", ".join(("dynamics", "surface layer") + keep),
                                       out=", ".join(p for p in INNER_MODEL_PIECES if p not in keep) or "none")
                self.route_env = {**self.route_env, **words}
                print(f"woof hex {door}: {line}", flush=True)
            self.out = Path(args.out).resolve()
            self.out.mkdir(parents=True, exist_ok=True)
            self.plan, self.plan_how = resolve_plan(args, self.root, self.row_name, self.row)
            self.row = plan_scheme_holds(self.table, self.row, self.plan)   # a scheme the plan runs held by name
            self.plan_physics_mode = refuse_plan_physics_mode(self.plan, self.physics_mode, self.table)
            from . import ad_radiation_seed as _radseed   # the plan's cloud-sample seed rule; the other is refused
            self.radiation_seed = _radseed.refuse_plan_seed(self.plan, getattr(args, "radiation_seed", None), door)
            said = physics_mode_block(self.table).get("line")
            if said:   # every door says which forecast its linear model is of, before the window
                print(f"woof hex {door}: {said}", flush=True)
            for hold in (self.row.get("derivative_holds") or {}).values():   # a scheme held by name, said on every door
                print(f"woof hex {door}: {hold['what']}", flush=True)
            bind_plan_libraries(self.plan)
            recorded = _plan_facts(self.plan).arch()
            if self.arch and recorded and recorded != self.arch:
                raise AdDoorError(f"plan {self.plan} was recorded on an {recorded} card (its forward kernel images are "
                                  f"that card's code) and card {self.card} is {self.arch}: record the plan on this card "
                                  "with --init INIT (or run on an " + recorded + " card)")
            recorded_dt = _plan_dt(self.plan)
            if self.row.get("generated") and recorded_dt and recorded_dt != float(self.row["dt_s"]):
                # a given plan carries its own timestep; the made row follows the recording, not the default rule
                default_words = f"dt {float(self.row['dt_s']):g} s"
                self.row = dict(self.row, dt_s=recorded_dt,
                                what=self.row["what"].replace(default_words, f"dt {recorded_dt:g} s (the given plan's; "
                                                              f"the row default is {float(self.row['dt_s']):g} s)", 1))
            # the card part of the row: the derivative images for this card's arch (built once, then cached)
            self.arch_assets = (_rows.ensure_arch_images(self.table, self.row_name, self.row, self.arch, self.root,
                                                         needed=_needed_assets(self.table, self.row, door))
                                if self.arch else {})
            self.assets.update(self.arch_assets)
            # the mesh part: a kernel set pinned to this mesh's derived topology tables
            self.kset_assets = _rows.ensure_kset(self.table, self.row_name, self.row, self.plan, self.root)
            self.assets.update(self.kset_assets)
            self.rank_prefix, self.rank_maps, self.rank_plan_check = None, None, None
            if self.split:   # the rank plans and their maps back to the global mesh, before any card work
                given = getattr(args, "rank_plans", None)
                self.rank_prefix = Path(os.path.abspath(given if given is not None else self.root / self.split["pkg"]))
                _ranks.refuse_rank_plans_missing(self.rank_prefix, self.split["ranks"], self.split["gather"])
                self.rank_maps = [_ranks.load_gather(d, self.split["gather"])
                                  for d in _ranks.rank_dirs(self.rank_prefix, self.split["ranks"])]
                self.rank_plan_check = _ranks.refuse_other_plan(self.rank_maps, self.plan)
                if self.split.get("ownerreduce"):   # the row's own kernel set plus the owner-reduced damping row
                    from .ad import ownerreduce as _own
                    try:
                        built = _own.ensure(Path(self.assets["kernels"]),
                                            self.root / self.split["ownerreduce"]["selected"],
                                            _rows.rows_cache(), cubins=Path(self.assets["cubins"]))
                    except _own.OwnerReduceError as error:
                        raise AdDoorError(f"row {self.row_name} on {self.split['ranks']} ranks: {error}") from error
                    self.assets.update(built)
                    self.kset_assets = {**self.kset_assets, **built}
        except _rows.AdRowError as error:
            self._session_refused()
            raise AdDoorError(str(error)) from error
        except _ranks.RankSplitError as error:
            self._session_refused()
            raise AdDoorError(str(error)) from error
        except BaseException:
            self._session_refused()
            raise
        self.fmap = FieldMap.for_plan(self.plan, self.root, self.table,
                                      mesh={"cells": int(self.row["cells"]), "edges": int(self.row["edges"])})
        self.fmap.window_steps = self.steps   # every window-end read is placed at boundary steps + 1
        grid = Path(args.grid) if args.grid else self.assets.get("grid")
        if grid is None or not grid.is_file():
            raise AdDoorError(f"mesh grid {grid} is missing: pass --grid (the row's mesh grid file)")
        self.grid = grid
        self.geometry = mesh_geometry(grid, self.row["cells"], self.row["edges"])
        self.started = time.time()
        from .ad_plan_facts import process_age_s
        age = process_age_s()   # what the command spent before its own clock (wall_s) started: on the receipt
        self.before_clock_s = None if age is None else round(age, 1)
        self.stages: list[dict] = []
        self.windows: list[dict] = []

    def _session_refused(self) -> None:
        """The command stopped before its checks passed: the session's held line is dropped (ad_session.Use.refused)."""
        use = getattr(self, "session", None)
        if use is not None and hasattr(use, "refused"):
            use.refused()

    def need_mib(self) -> int:
        """Free card memory the window is admitted on: --need-mib, else the row's measured (or made) need."""
        given = getattr(self.args, "need_mib", None)
        if given:
            return int(given)
        if getattr(self, "split", None):   # a rank's window on its own card
            return int(self.split["need_free_mib_rank"])
        need = int(self.row["need_free_mib"])
        if getattr(self, "routes_on", getattr(self, "door", None) == "sensitivity"
                   and not getattr(self.args, "checked_forward", False)):
            # the boundary-layer route may hold more than the row's other windows (its own measure)
            _, route, _ = boundary_layer_route(self.row, getattr(self.args, "boundary_layer", None))
            need = max(need, int(route.get("need_free_mib") or 0))
        return need

    def row_what(self) -> str:
        """What the operator this door ran is: the row's words and, for param-grad, the words of the bundle it runs in
        place of the row's (the table's param_bundle.instead_of_row), so a receipt never names a hold its run lacked."""
        own = (self.row.get("param_bundle") or {}).get("instead_of_row") if getattr(self, "door", None) == "param-grad" else None
        return f"{self.row['what']}. IN param-grad: {own}" if own else self.row["what"]

    def need_host_mib(self) -> int | None:
        """Host memory the window needs: --need-host-mib, else this door's measured need on the row
        (need_host_mib_by_door), else the row's (need_host_mib), else None (not measured). The doors differ: through
        the door at 120 km, adjoint-test (its finite-difference forwards) held about 26 GiB more host memory than the
        bare window."""
        given = getattr(self.args, "need_host_mib", None)
        if given:
            return int(given)
        split = getattr(self, "split", None)
        if split:   # every rank runs on this host: N times a rank's need
            return int(split["need_host_mib_rank"]) * int(split["ranks"]) if split.get("need_host_mib_rank") else None
        store = getattr(self, "store_need", None)
        if store and store.get("host_mib"):   # the gradient line: its memory beside the store plus the store in memory
            # a window that keeps every step boundary: the largest need any window of this command was judged on
            return max(int(store["host_mib"]), int(getattr(self, "host_need_seen", 0) or 0))
        by_door = self.row.get("need_host_mib_by_door") or {}
        need = by_door.get(getattr(self, "door", None)) or self.row.get("need_host_mib")
        return int(need) if need else None

    def needs_line(self) -> str:
        """One line, printed before the window runs: the card and host memory the row needs, what is free here, and
        the minutes the door took on the reference card where the table records them."""
        free = free_mib(self.card)
        host = self.need_host_mib()
        words = [f"woof hex {self.door}: row {self.row_name} needs {self.need_mib():,} MiB of card memory "
                 f"(card {self.card}: {'unknown' if free is None else format(free, ',')} MiB free)"]
        avail = getattr(self, "host_available", None)
        here = "unknown" if avail is None else format(avail, ",")
        store = getattr(self, "store_need", None)
        which = ("" if not (store and store.get("host_for")) or getattr(self.args, "need_host_mib", None) else
                 f"; {store['host_for']}" if store.get("line") is False else
                 f"; {store['host_for']}: {getattr(self, 'check_expected_why', '')}")
        words.append(f"{host:,} MiB of host memory ({here} MiB available{which})" if host else
                     f"host memory not measured for this row ({here} MiB available)")
        if store:
            free = getattr(self, "disk_free", None)
            words.append(f"{int(store['disk_mib']):,} MiB of disk in the run folder "
                         f"({'unknown' if free is None else format(free, ',')} MiB free there; {store['how']})")
        ref = (self.row.get("reference_minutes") or {}).get(self.door)
        if ref is not None:
            words.append(f"about {float(ref):g} minutes on {self.row.get('reference_card') or 'the reference card'}")
        if self.row.get("card_words"):   # what the class will put on the runner's line on this card, as of now
            from .ad import rows as _rows
            try:
                _, filled = _rows.fill_card_words([], self.row["card_words"], free, avail)
            except _rows.AdRowError:
                filled = {}
            cls = (self.row.get("generated") or {}).get("window_class") or {}
            words.append(f"window class {cls.get('from_recipe')} on card{'s' if len(self.cards) > 1 else ''} "
                         f"{','.join(self.cards)}, words from this card and host: "
                         + (", ".join(f"{k.strip('{}')} {v}" for k, v in sorted(filled.items())) or "not readable"))
        split = getattr(self, "split", None)
        if split:
            shm = _ranks.shm_free_mib()
            words.append(f"split over {split['ranks']} ranks, one on each of cards {','.join(self.cards)} (each card "
                         f"{split['need_free_mib_rank']:,} MiB; /dev/shm "
                         + (f"{split['shm_mib']:,} MiB" if split.get("shm_mib") else "not measured")
                         + f", {'unknown' if shm is None else format(shm, ',')} MiB free; exchange {split['exchange']})")
        return "; ".join(words)

    def check_cards(self, state: bool | None = None) -> str:
        """The card rules of the window's admission, applied before the asset root is resolved, so a door that would
        be refused at the window is refused before it downloads the row and plan (about 940 MB on first use): every
        card is readable, no card is smaller than its need, and without --wait-card the need is free now. The window
        is admitted again when it starts (memory can change meanwhile). Returns the first card's compute capability.

        ``state``: False runs only what no waiting changes (the card is readable, its family has images, it is not
        smaller than the need in all); True runs only the card's free memory now; None (a caller with one call) runs
        both. The door's context runs the first half before its install and mesh checks and the second after them."""
        shard_need = int(self.row.get("need_free_mib_shard") or 0)
        first = None
        for index, card in enumerate(self.cards):
            need = self.need_mib() if index == 0 or getattr(self, "split", None) else shard_need
            if state is True:   # the free memory alone: the card was read and sized by the first half
                if index == 0 and getattr(getattr(self, "session", None), "stopped", None):
                    admit_card(card, need, max(float(self.args.wait_card or 0.0), SESSION_RELEASE_WAIT_S))
                elif not self.args.wait_card:
                    admit_card(card, need, 0.0)
                continue
            cc = compute_capability(card)
            if cc is None:
                raise AdDoorError(f"cannot read card {card} (nvidia-smi): the window runs on the GPU only, so the door "
                                  "stops before it fetches or opens the row's assets")
            first = first or cc
            if index == 0:   # a card family the row has no images for: said before its memory (no size helps it)
                from .ad import rows as _rows
                needed = (_needed_assets(self.table, self.row, self.door) if getattr(self, "table", None)
                          else None)
                _rows.refuse_arch_without_images(getattr(self, "row_name", "?"), self.row, _rows.arch_of(cc),
                                                 needed=needed, card=card, need_mib=self.need_mib())
            if getattr(self, "split", None) and cc != first:   # the ranks share one kernel set and one plan's images
                raise AdDoorError(f"card {card} has compute capability {cc} and card {self.cards[0]} has {first}: a "
                                  "split window runs one kernel set on every card, so give cards of one family")
            if state is False:
                refuse_card_too_small(card, need)
                continue
            if index == 0 and getattr(getattr(self, "session", None), "stopped", None):
                # THE BREAKAGE THIS PREVENTS: this command has just stopped the program found loaded on the card (it
                # was started with other words, or this command keeps no session); that program gives the card back
                # a few seconds after it ends (3 s on the card job of 2026-11-14), and a command asking at once was
                # refused for the memory its own stop was freeing. It waits for that memory, as Context.window does.
                admit_card(card, need, max(float(self.args.wait_card or 0.0), SESSION_RELEASE_WAIT_S))
            elif self.args.wait_card:
                refuse_card_too_small(card, need)
            else:
                admit_card(card, need, 0.0)
        return first

    def window_host_admission(self, sub: str) -> dict | None:
        """A window that keeps every step boundary (the row's window_host), judged when it starts: the runner sizes
        its store from the memory available at that moment, so the need of each window of a command is read then.
        Returns what it read (the receipt keeps it with the window's admission); None for every other window, for a
        split window and with --need-host-mib.

        THE BREAKAGE THIS PREVENTS: the three windows of the 3 km adjoint-test start 13 to 33 minutes apart; on a
        shared host the memory available moved between them (2026-11-15: 304 GB at the first window, 469 GB at the
        second), so the need stated at the command's start described the first window only, and a window that
        starts with less than it needs is ended part way through with no sentence."""
        sn = getattr(self, "store_need", None) or {}
        if (sn.get("line") is not False or not sn.get("host_for") or getattr(self.args, "need_host_mib", None)
                or getattr(self, "split", None)):
            return None
        available = host_available_mib()
        now = store_need(self.row, self.steps, available, False) or {}
        if not now.get("host_mib"):
            return None
        if available is not None and available < int(now["host_mib"]):
            # A window of this command has just ended: its runner's page-locked memory can take seconds to come back
            # (host_available_settled's own breakage). A reading that is short is read again once it has settled, so
            # a later window is never refused for the memory the window before it is still giving back.
            available = host_available_settled(WINDOW_RELEASE_WAIT_S)
            now = store_need(self.row, self.steps, available, False) or now
        refuse_host_too_small(self.row_name, int(now["host_mib"]), available,
                              what=f"its {sub} window, {now['host_for']}", advice=smallest_host_words(now))
        self.host_need_seen = max(int(getattr(self, "host_need_seen", 0) or 0), int(now["host_mib"]))
        got = {"window": sub, "available_mib": available, "need_mib": int(now["host_mib"]),
               "store_in_memory_mib": int(now["ram_mib"])}
        if now.get("bstore_ram_gib") is not None:
            got["bstore_ram_gib"] = int(now["bstore_ram_gib"])
        return got

    def class_label(self) -> str | None:
        return ((self.row.get("generated") or {}).get("window_class") or {}).get("from_recipe")

    def check_card_words(self) -> None:
        """A class whose words follow the card and host (card_words): the card's free memory and the host's available
        memory now give each word at least its smallest budget, or the door stops here, before the asset root (with
        --wait-card the card words are filled and checked at the window's admission instead). The window fills them
        again when it starts."""
        words = self.row.get("card_words")
        if not words:
            return
        from .ad import rows as _rows
        if self.args.wait_card:
            words = {k: v for k, v in words.items() if v.get("from") != "card_free_mib"}
        _rows.fill_card_words([], words, free_mib(self.card), self.host_available, card=self.card,
                              label=self.class_label())

    def stage(self, name: str, t0: float, parallelism: str) -> None:
        self.stages.append({"stage": name, "wall_s": round(time.monotonic() - t0, 3), "parallelism": parallelism})

    def window_line(self, extra: list[str], sub: str = "window", env_overrides: dict | None = None,
                    door_args_key: str = "window_args", extra_env: dict | None = None,
                    spec_door: str | None = None, replace_args: dict | None = None) -> tuple[list[str], dict]:
        """The runner's line and environment of a window of this door (no card is asked)."""
        argv, env = window_command(self.table, self.row, self.assets, self.door, self.runner, self.steps,
                                   self.out / sub, extra, env_overrides, door_args_key,
                                   {**getattr(self, "route_env", {}), **(extra_env or {}),
                                    **getattr(self, "plant_env", {})}, spec_door, replace_args)
        payloads = self.plan / "recording" / "payloads"
        if "AD_BINDER_REFERENCE_PAYLOADS" not in env and payloads.is_dir():
            env["AD_BINDER_REFERENCE_PAYLOADS"] = str(payloads)
        return argv, env

    def window(self, extra: list[str], sub: str = "window", env_overrides: dict | None = None,
               door_args_key: str = "window_args", extra_env: dict | None = None,
               spec_door: str | None = None, replace_args: dict | None = None,
               unchecked_forward: dict | None = None) -> dict:
        """One runner window. On a shared card another process can take the memory the window was admitted on;
        with --wait-card the door then waits for the memory again and reruns the window (each attempt is in the
        receipt), at most three times. Without --wait-card the memory failure is the door's answer.
        ``unchecked_forward``: the gradient line's table entry when this window runs it (its forward boundaries are
        not compared with the recording; see gradient_line_validity)."""
        program = getattr(self, "session_program", None)
        release_wait = 0.0
        if program is not None:   # a window of this command's own needs the card the session's program holds
            program.end(f"this command's {sub} window needed the card")
            # THE BREAKAGE THIS PREVENTS (card job of 2026-11-14): a later command that had ATTACHED ran its gradient,
            # ended the loaded program for its answer check and was refused "card 0 has 13210 MiB free; this row's
            # window needs 96000 MiB": the program it had just ended gave the card back 3 s later. The window is
            # admitted with a wait for the memory that program is releasing.
            release_wait = SESSION_RELEASE_WAIT_S
        # every window of a route door runs the route's words (route_window_words): the test windows, the answer
        # check and the poison proof all check the linear model the command names
        argv, env = window_command(self.table, self.row, self.assets, self.door, self.runner, self.steps,
                                   self.out / sub, extra, env_overrides, door_args_key,
                                   {**getattr(self, "route_env", {}), **(extra_env or {}),
                                    **getattr(self, "plant_env", {})}, spec_door, replace_args)
        # The plan's own recorded payloads: the runner reads one only when a boundary word check mismatches, to read
        # device addresses its address list misses (it replaces no model bytes). Without it the 3 km storm box
        # stopped at step 1 ("state hash mismatch ... first 102"); its recipe carries it.
        split = getattr(self, "split", None)
        if split:   # N rank processes, one a card: each replays its own rank plan; the door gathers their answers
            argv = _ranks.ranks_argv(argv, split["ranks"], self.cards, self.rank_prefix)
            env = {**env, **split["exchange_env"]}
            payloads = _ranks.rank_dirs(self.rank_prefix, 1)[0] / "recording" / "payloads"
            if "AD_BINDER_REFERENCE_PAYLOADS" not in env and payloads.is_dir():   # each rank its own ({rank})
                env["AD_BINDER_REFERENCE_PAYLOADS"] = f"{self.rank_prefix}-r{{rank}}/recording/payloads"
        else:
            payloads = self.plan / "recording" / "payloads"
            if "AD_BINDER_REFERENCE_PAYLOADS" not in env and payloads.is_dir():
                env["AD_BINDER_REFERENCE_PAYLOADS"] = str(payloads)
        attempts = []
        while True:
            archs = None if self.arch_assets else self.row.get("compute_capability")
            host_now = self.window_host_admission(sub)
            admission = admit_card(self.card, self.need_mib(), max(float(self.args.wait_card), release_wait), archs)
            if host_now:
                admission["host"] = host_now
            shard_need = self.need_mib() if split else int(self.row.get("need_free_mib_shard") or 0)
            admission["shard_cards"] = [admit_card(c, shard_need, float(self.args.wait_card), archs)
                                        for c in self.cards[1:]]
            run_argv, run_env = argv, env
            if host_now and host_now.get("bstore_ram_gib") is not None:   # a host too small for the default store
                run_argv = with_bstore_ram_gib(run_argv, host_now["bstore_ram_gib"])
                print(f"woof hex {self.door}: its {sub} window keeps {host_now['bstore_ram_gib']} GiB of step "
                      f"boundaries in host memory (this host has {host_now['available_mib']:,} MiB available, too "
                      "little for the runner's default of 0.4 of it) and writes the rest to the run folder: the "
                      "same numbers, more disk, and more time writing and reading the files", flush=True)
            if self.row.get("card_words"):   # the class's words that follow the card it was admitted on, and the host
                from .ad import rows as _rows
                try:
                    run_argv, filled = _rows.fill_card_words(argv, self.row["card_words"], admission.get("free_mib"),
                                                             host_available_mib(), card=self.card,
                                                             label=self.class_label())
                except _rows.AdRowError as error:
                    raise AdDoorError(str(error)) from error
                admission["card_words"] = filled
            # physics columns on the other cards (--cards); not under ranks: each rank sees its one card only
            # (CUDA_VISIBLE_DEVICES), so shard ordinals 0..N-1 would name cards the rank cannot see
            if len(self.cards) > 1 and self.row.get("shard_env") and not split:
                ordinals = ",".join(str(i) for i in range(len(self.cards)))
                run_env = {**env, **{k: v.replace("{ordinals}", ordinals) for k, v in self.row["shard_env"].items()}}
            t0 = time.monotonic()
            try:
                if split:
                    doc = run_window(run_argv, run_env, self.plan, self.out / sub, ",".join(self.cards),
                                     ranks=split["ranks"], exchange_label=split.get("exchange_runner_label"))
                else:
                    doc = run_window(run_argv, run_env, self.plan, self.out / sub, ",".join(self.cards))
            except CardMemoryError as error:
                attempts.append({"admission": admission, "failed": str(error), "wall_s": round(time.monotonic() - t0, 1)})
                if not self.args.wait_card or len(attempts) >= 3:
                    raise
                print(f"woof hex {self.door}: card {self.card} ran out of memory under another process "
                      f"(attempt {len(attempts)}); waiting for {self.need_mib()} MiB and rerunning",
                      file=sys.stderr)
                continue
            break
        self.stage(f"window ({sub})", t0, "GPU (runner)")
        if split:
            self.window_docs = [*getattr(self, "window_docs", []), {"window": sub, "ranks": doc.get("ranks")}]
        if unchecked_forward is not None:
            doc["door_validity"] = gradient_line_validity(doc.get("door_validity") or {}, unchecked_forward)
        doc["_admission"] = admission
        self.memory_retries = attempts
        self.windows.append({"window": sub, "card_peak_mib": doc.get("door_card_peak_mib"),
                             "wall_s": round(doc.get("door_wall_s") or 0.0, 1), **(doc.get("door_validity") or {})})
        return doc

    def window_loaded(self, program: "LoadedProgram", extra: list[str], name: str, words: dict,
                      line: dict) -> dict:
        """One objective's gradient-line window on the call's loaded program: the first starts it (admitted on the
        card as any window), each later one is a request to it. Its validity is the gradient line's."""
        argv, env = self.window_line(extra, sub="window", extra_env=words["env"], replace_args=words["replace"])
        out = self.out / "window"
        t0 = time.monotonic()
        first = program.proc is None
        if first:
            archs = None if self.arch_assets else self.row.get("compute_capability")
            admission = admit_card(self.card, self.need_mib(), float(self.args.wait_card), archs)
            if self.row.get("card_words"):
                from .ad import rows as _rows
                try:
                    argv, filled = _rows.fill_card_words(argv, self.row["card_words"], admission.get("free_mib"),
                                                         host_available_mib(), card=self.card,
                                                         label=self.class_label())
                except _rows.AdRowError as error:
                    raise AdDoorError(str(error)) from error
                admission["card_words"] = filled
            doc = program.start(argv, env, self.plan, ",".join(self.cards), name, out)
        else:
            admission = {"card": self.card, "loaded_program": f"served by the program admitted for {program.first}"}
            doc = program.request(name, _request_words(argv, out), out)
        label = (program.stage_label(first, name) if hasattr(program, "stage_label")
                 else f"{'loaded program start' if first else 'loaded program request'} {name}")
        self.stage(f"window (window, {label})", t0, "GPU (runner, one loaded program)")
        return self._loaded_doc(doc, out, argv, line, admission)

    def window_loaded_fallback(self, doc: dict, extra: list[str], words: dict, line: dict) -> dict:
        """The first objective's window when the runner ended after it without serving requests (NotLoaded)."""
        argv, _ = self.window_line(extra, sub="window", extra_env=words["env"], replace_args=words["replace"])
        self.stage("window (window, one program: the runner serves no requests)", time.monotonic()
                   - float(doc.get("door_wall_s") or 0.0), "GPU (runner)")
        return self._loaded_doc(doc, self.out / "window", argv, line, {"card": self.card})

    def _loaded_doc(self, doc: dict, out: Path, argv: list[str], line: dict, admission: dict) -> dict:
        doc["door_validity"] = gradient_line_validity(window_validity(out, doc, argv, streams=()), line)
        doc["_admission"] = admission
        self.memory_retries = []
        said = keep_forward_words(doc, getattr(self, "keep_forward", None))
        if said:   # which happened for this gradient: the forward of the gradient before taken, or its own run
            print(f"woof hex {self.door}: {said}", flush=True)
        self.windows.append({"window": "window", "card_peak_mib": doc.get("door_card_peak_mib"),
                             "wall_s": round(doc.get("door_wall_s") or 0.0, 1), **doc["door_validity"],
                             "keep_forward": doc.get("keep_forward"), "keep_forward_said": said})
        return doc

    def split_block(self) -> dict:
        """What a split window was: ranks, cards, the exchange asked for and the one each rank's runner reports."""
        s = self.split
        seen = sorted({x for w in getattr(self, "window_docs", []) for x in ((w.get("ranks") or {}).get("exchange_seen")
                                                                             or [])})
        return {"ranks": s["ranks"], "cards": list(self.cards), "rank_on_card": {str(k): c for k, c in enumerate(self.cards)},
                "exchange": s["exchange"], "exchange_env": s["exchange_env"], "exchange_seen_by_ranks": seen,
                "rank_plans": f"{self.rank_prefix}-r<k>", "need_free_mib_rank": s["need_free_mib_rank"],
                "need_host_mib_rank": s["need_host_mib_rank"], "shm_mib": s["shm_mib"],
                "gather": "woof.hex.ad.rank_gather (the one reader; tc3_views.py gather reads through the same file): a "
                          "gradient is every rank's local copies, owned, halo and replicated, summed into the one-card "
                          "allocation, then read as on one card; an objective is cut to the owning rank, a direction "
                          "to every copy",
                "rank_maps_against_plan": getattr(self, "rank_plan_check", None),
                "ownerreduce": self.split.get("ownerreduce"),
                "measured": s.get("measured"), "what": s.get("what"), "experimental": s.get("experimental")}

    def rank_json(self, path: Path, owner_only: bool) -> str:
        """The runner's word for a door JSON of terms ([alloc, word, weight]): the file itself on one card; on a split
        window, one cut per rank (path-r<k>.json) and the word path-r{rank}.json the rank launcher fills in."""
        if not getattr(self, "split", None):
            return str(path)
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        for k, gmap in enumerate(self.rank_maps):
            try:
                terms = _ranks.cut_terms(doc["terms"], gmap, owner_only)
            except _ranks.RankSplitError as error:
                raise AdDoorError(f"{path}: rank {k}: {error}") from error
            _write_json(path.with_name(f"{path.stem}-r{k}.json"), {**doc, "terms": terms, "rank": k})
        return str(path.with_name(f"{path.stem}-r{{rank}}.json"))

    def rank_list(self, name: str, items: list[tuple[int, int, int]], form: str) -> str:
        """The runner's word for a dump list of global (alloc, byte offset, bytes) copies: the list itself on one
        card; on a split window, @FILE per rank. Form "alloc": every rank allocation holding a copy of any of the
        global allocations (what rank_gather.gather_allocs reads back); form "alloc:offset:bytes": the rank's copy of
        each global copy."""
        def word(entries):
            if form == "alloc":
                return ",".join(str(a) for a in sorted({a for a, _, _ in entries}))
            return ",".join(f"{a}:{o}:{b}" for a, o, b in entries)
        if not getattr(self, "split", None):
            return word(items)
        from .ad import rank_gather as _rg
        for k, gmap in enumerate(self.rank_maps):
            if form == "alloc":
                local = [(a, 0, 0) for a in _rg.local_allocs(gmap, [a for a, _, _ in items])]
                if not local:
                    raise AdDoorError(f"rank {k}: no allocation of the rank map holds a copy of "
                                      f"{sorted({a for a, _, _ in items})}")
            else:
                local = []
                for alloc, off, _ in items:
                    hit = _rg.find_copy(gmap, alloc, off)
                    if hit is None:
                        raise AdDoorError(f"rank {k}: allocation {alloc} at byte {off} is in no field of the rank map")
                    c = gmap["fields"][hit[0]]["local"][hit[1]]
                    local.append((int(c["alloc"]), int(c["offset"]), _rg.extent_bytes(c)))
            (self.out / f"{name}-r{k}.txt").write_text(word(sorted(set(local))) + "\n")
        return f"@{self.out / name}-r{{rank}}.txt"

    def gather_rank_grads(self, win: Path, inputs: list[dict]) -> dict:
        """The ranks' gradient dumps (win/r<k>/grad-<alloc>.f32) summed back into the one-card allocations
        (win/grad-<alloc>.f32), so the one-card reader (_sum_copies) reads them unchanged. Returns the gather report."""
        import numpy as np
        from .ad import rank_gather as _rg
        need = {}
        for f in inputs:
            for (alloc, off), shape in zip(f["copies"], f["copy_shapes"]):
                need[int(alloc)] = max(need.get(int(alloc), 0), int(off) // 4 + math.prod(shape),
                                       _rg.global_words(self.rank_maps, alloc))
        try:
            grads, rep = _rg.gather_allocs(self.rank_maps, sorted(need),
                                           lambda r, a: np.fromfile(win / f"r{r}" / f"grad-{a}.f32", "<f4"), need)
        except (_rg.RankGatherError, OSError) as error:
            raise AdDoorError(f"gathering the ranks' gradients in {win}: {error}") from error
        for alloc, g in grads.items():
            g.tofile(win / f"grad-{alloc}.f32")
        report = {"rule": "every rank's copies (owned, halo, replicated) summed into the one-card allocation",
                  "allocs": {str(k): v for k, v in rep.items()}}
        _write_json(win / "gather.json", report)
        return report

    def runner_assets(self) -> dict[str, str]:
        """Every file the row hands the runner (its argument and environment assets and the refuse list), named below
        the asset root when it lies there: what pins the kernel set a receipt's numbers came from."""
        keys = set(self.row.get("asset_args", {}).values()) | set(self.row.get("asset_env", {}).values()) | {"refuse"}
        keys |= set(_door_asset_env(getattr(self, "table", None), getattr(self, "door", None)).values())
        out = {}
        for key in sorted(keys):
            if key in self.assets:
                path = Path(self.assets[key])
                try:
                    out[key] = str(path.relative_to(self.root))
                except ValueError:
                    out[key] = str(path)
        return out

    def gradient_block(self) -> dict:
        """The receipt's word on the gradient's mesh: a block set by the coarse route (gradient_block_override), else
        the mesh itself, with what was asked and why (gradient_route)."""
        own = getattr(self, "gradient_block_override", None)
        if own is not None:
            return own
        route = getattr(self, "gradient_route", None) or {}
        mesh = (getattr(self, "row", None) or {}).get("mesh")
        asked = getattr(getattr(self, "args", None), "gradient_mesh", None)
        return {"asked": route.get("asked") or asked or GRADIENT_MESH_DEFAULT,
                "how": route.get("why") or "the mesh itself", "carried": False,
                "made_on": {"mesh": mesh, "row": getattr(self, "row_name", None)}, "carried_to": None,
                "method": "none: the gradient was taken on the mesh itself",
                "answer_check": "made on this operator (the strict check in this receipt)"}

    def gradient_nc_attrs(self) -> dict:
        """The output file's word on the gradient's mesh (global attributes)."""
        b = Context.gradient_block(self)
        made = b["made_on"] or {}
        to = b.get("carried_to") or {}
        return {"gradient_mesh_asked": b["asked"], "gradient_made_on_mesh": made.get("mesh"),
                "gradient_carried": ("yes: made on mesh " + str(made.get("mesh")) + ", carried to mesh "
                                     + str(to.get("mesh")) + " by " + str(b["method"])) if b["carried"]
                                    else "no: taken on the mesh itself",
                "gradient_answer_check": b["answer_check"]}

    def _refused_names(self) -> list[str]:
        path = self.assets.get("refuse")
        if path is None or not path.is_file():
            return []
        return [x for x in path.read_text(encoding="utf-8").strip().split(",") if x]

    def receipt(self, result: dict, passed: bool | None = None, strict: dict | None = None) -> dict:
        """The door's receipt. ``strict`` is the strict whole-state check of this operator (``strict_check``): with
        it, "valid" also needs that check to pass, its left-out memory printed and proved; a check not run
        (--no-self-dot) leaves "valid" null (not checked), never true."""
        from . import __version__
        doc = {"schema": RECEIPT_SCHEMA, "door": self.door, "version": __version__, "row": self.row_name,
               "row_what": Context.row_what(self), "mesh": self.row["mesh"], "suite": self.row["suite"],
               "steps": self.steps, "window_hours": self.steps * self.row["dt_s"] / 3600.0,
               "plan": str(self.plan), "plan_how": self.plan_how, "plan_sha256": self.fmap.doc["plan_sha256"],
               "plan_recording": plan_recording(self.plan),
               "physics_mode": {"linear_model": getattr(self, "physics_mode", None),
                                "plan_recorded": getattr(self, "plan_physics_mode", None),
                                "linearised": physics_mode_block(getattr(self, "table", None)
                                                                 or load_table())["linearised"]},
               "alloc_map": str(self.fmap.source), "alloc_map_copies_found": self.fmap.found,
               "window_end_boundary": self.fmap.end_boundary(),
               "runner": str(self.runner),
               "runner_sha256": _sha256_file(self.runner), "assets": str(self.root), "card": self.card,
               "cards": self.cards,
               "argv": sys.argv, "stages": self.stages, "wall_s": round(time.time() - self.started, 1),
               "before_clock_s": getattr(self, "before_clock_s", None),
               "plan_facts": _plan_facts(self.plan).how,
               "host_peak_mib": host_peak_mib(), "need_host_mib": Context.need_host_mib(self),
               "store_need": getattr(self, "store_need", None), "disk_free_mib": getattr(self, "disk_free", None),
               "host_available_mib": getattr(self, "host_available", None),
               "card_memory_retries": getattr(self, "memory_retries", []),
               "arch": self.arch, "row_generated": self.row.get("generated"),
               "arch_assets": {k: str(v) for k, v in self.arch_assets.items()},
               "kset_assets": {k: str(v) for k, v in self.kset_assets.items()},
               "need_mib": self.need_mib(), "runner_assets": self.runner_assets(),
               # launches refused by name (identity both ways): the row's list, as passed to the runner. The
               # runner's own per-rule counts exist only for windows that run the tangent forward.
               "refused_by_name": self._refused_names(),
               "windows": self.windows,
               "card_peak_mib": max([w["card_peak_mib"] or 0 for w in self.windows], default=0) or None,
               "valid": all(w.get("valid", False) for w in self.windows) if self.windows else None,
               "result": result}
        if strict is not None:
            doc["strict_check"] = strict
            if doc["valid"] is not False:
                doc["valid"] = None if strict.get("how") == "not run" else bool(strict.get("pass"))
            doc["valid_rule"] = ("every window valid (no identity beyond the refuse list, every step boundary equal "
                                 "to the recording) and the strict whole-state dot under its bar, any left-out memory "
                                 "proved by a NaN-poison window on this plan (own windows; a passing verdict of those "
                                 "windows, run by an earlier command on the same key of plan and its files, runner, "
                                 "kernel set, card family, steps, row words and test direction, and kept in the "
                                 "door's cache; or a bound adjoint-test receipt of the same plan, row, runner and "
                                 "kernel set)")
            keep = getattr(self, "check_key", None) is not None and strict.get("how") == "own"
            if keep:   # where this check's verdict is kept for the next commands on this operator
                doc["strict_check"] = {**strict, "kept_as": str(check_verdict_path(self.check_key))}
        doc["gradient_mesh"] = Context.gradient_block(self)
        if getattr(self, "route_block", None):   # which boundary-layer route's linear model every window ran
            doc["boundary_layer"] = self.route_block
        for key, value in (getattr(self, "receipt_extra", None) or {}).items():   # the sensitivity door's route blocks
            doc[key] = value
        split = getattr(self, "split", None)
        if split:   # only a split window's receipt carries this block; a one-card receipt is as before
            doc["split"] = self.split_block()
            if split.get("experimental"):
                doc["experimental"] = split["experimental"]["line"]
            if getattr(self, "gather_report", None):
                doc["split"]["gathered"] = self.gather_report
        plant = getattr(self, "plant", None)
        if plant:   # an error planted on purpose (adjoint-test --plant-error): never valid, never passed, never kept
            doc["planted_error"] = plant
            doc["valid"] = False
        if passed is not None:
            doc["passed"] = bool(passed) and doc["valid"] is not False
        for w in self.windows:
            if w.get("valid") is False:
                print(f"woof hex {self.door}: window {w['window']} is NOT a valid derivative: launch kinds run as "
                      f"identity beyond the row's refuse list {w['refused_beyond_row_list']}; step boundaries equal "
                      f"{w['boundaries_equal']} of {w['boundaries_checked']}", file=sys.stderr)
        _write_json(self.out / "receipt.json", doc)
        if strict is not None and getattr(self, "check_key", None) is not None and strict.get("how") == "own":
            if store_check(self, strict, self.out / "receipt.json") is None:   # said on stderr; the receipt too
                doc["strict_check"]["kept_as"] = None
                _write_json(self.out / "receipt.json", doc)
        return doc


def _perturbation(ctx: Context, state: PlanState) -> dict:
    a = ctx.args
    spec = ctx.table["perturbations"]["theta"]["default"]
    direction = theta_direction(ctx.table, ctx.fmap, state, ctx.geometry, resolve_at(a.at, ctx.geometry),
                                a.amplitude if a.amplitude is not None else spec["amplitude"],
                                a.radius_km if a.radius_km is not None else spec["radius_km"],
                                a.level, a.level_width if a.level_width is not None else spec["level_width"])
    path = ctx.out / "direction.json"
    _write_json(path, direction)
    return {"path": path, "doc": direction}


def run_tangent(args: argparse.Namespace) -> int:
    plant_request(args, load_table(), "tangent")   # a planting word in the environment: refused before any download
    route = gradient_route(args, "tangent")   # a coarse mesh is refused by name here
    refuse_speed_words(args, load_table())    # a route the row does not offer: refused before any download
    ctx = Context(args, "tangent")
    ctx.gradient_route = route
    import numpy as np
    t0 = time.monotonic()
    state = PlanState(ctx.plan)
    direction = _perturbation(ctx, state)
    names = args.fields.split(",") if args.fields else [
        n for n in ctx.table["tangent_outputs"]
        if not (ctx.row.get("holds_physics") and ctx.table["fields"][n].get("physics_diagnostic"))]
    # A plan recorded by the recorder maps the prognostic state only (no theta or rho copy). A field the table can
    # derive from mapped ones is derived on the host from their tangents; a default output that is neither mapped
    # nor derivable is left out and named in the receipt; one the user asked for by name (--fields) is refused.
    derived, absent = {}, []
    for n in list(names):
        spec = ctx.table["fields"].get(n) or {}
        if not spec or (spec.get("field") in ctx.fmap.fields and ctx.fmap.at_window_end(ctx.table, n)):
            continue
        sources = spec.get("derived_from") or []
        if sources and all(ctx.fmap.at_window_end(ctx.table, s) for s in sources):
            derived[n] = sources
        elif not args.fields:
            absent.append(n)
        if n in derived or n in absent:
            names.remove(n)
    for sources in derived.values():
        names += [s for s in sources if s not in names]
    fields = [ctx.fmap.resolve(ctx.table, n, ctx.row, window_end=True) for n in names]
    dumps = ",".join(f"{f['alloc']}:{f['offset']}:{4 * math.prod(f['read_shape'])}" for f in fields)
    ctx.stage("inputs (direction, field map)", t0, "1 thread (setup, small)")
    strict_from, no_self_dot = getattr(args, "strict_from", None), getattr(args, "no_self_dot", False)
    if strict_from is not None and not no_self_dot:   # refuse a receipt of another operator before the card
        bind_strict(ctx, strict_from)
    doc = ctx.window(["--taylor-dir", str(direction["path"]), "--dump-state", dumps])
    strict = strict_check(ctx, no_self_dot, strict_from)
    t0 = time.monotonic()
    win = ctx.out / "window"
    out_fields, summary = {}, []
    for f in fields:
        suffix = f"-{f['offset']}"
        tl = on_mesh(np.fromfile(win / f"tl-{f['alloc']}{suffix}.f32", "<f4").reshape(f["read_shape"]), f["shape"])
        base = on_mesh(np.fromfile(win / f"state-{f['alloc']}{suffix}.f32", "<f4").reshape(f["read_shape"]),
                       f["shape"])
        out_fields[f"d_{f['name']}"] = (tl, f["units"], f"tangent of {f['what']} at the window end", f["grid"])
        out_fields[f["name"]] = (base, f["units"], f"{f['what']} at the window end (base forecast)", f["grid"])
        summary.append({"field": f["name"], "units": f["units"], "max_abs_tangent": float(np.max(np.abs(tl))),
                        "rms_tangent": float(np.sqrt(np.mean(tl.astype(np.float64) ** 2))),
                        "nonfinite": int((~np.isfinite(tl)).sum())})
    for n, sources in derived.items():
        rule = ctx.table["fields"][n]["derivation"]
        if rule != "theta_m / (1 + (Rv/Rd) qv)":
            raise AdDoorError(f"field {n!r}: the table names a derivation this door does not have ({rule})")
        k = ctx.table["constants"]
        rvord = k["rv"] / k["rgas"]
        thm, qv = (out_fields[s][0].astype(np.float64) for s in sources)
        d_thm, d_qv = (out_fields[f"d_{s}"][0].astype(np.float64) for s in sources)
        a = 1.0 + rvord * qv
        tl = d_thm / a - thm * rvord * d_qv / (a * a)
        spec = ctx.table["fields"][n]
        out_fields[f"d_{n}"] = (tl, spec["units"], f"tangent of {spec['what']} at the window end (derived: {rule})",
                                spec["grid"])
        out_fields[n] = (thm / a, spec["units"], f"{spec['what']} at the window end (base forecast; derived)",
                         spec["grid"])
        summary.append({"field": n, "units": spec["units"], "max_abs_tangent": float(np.max(np.abs(tl))),
                        "rms_tangent": float(np.sqrt(np.mean(tl ** 2))), "nonfinite": int((~np.isfinite(tl)).sum()),
                        "derived_from": sources})
    nc = ctx.out / "tangent.nc"
    write_mesh_fields(nc, out_fields, {"title": "hex tangent-linear state at the window end",
                                       "perturbation": direction["doc"]["name"], "row": ctx.row_name,
                                       "steps": ctx.steps, "plan_sha256": ctx.fmap.doc["plan_sha256"]},
                      coords=mesh_coordinates(ctx.geometry))
    ctx.stage("tangent.nc (Rust NetCDF)", t0, "1 thread (host write)")
    ok = all(s["nonfinite"] == 0 for s in summary)
    result = {"perturbation": direction["doc"]["name"], "tangent_nc": str(nc), "fields": summary,
              "fields_not_in_this_plan": absent,
              "window_receipt": str(win / "window.json"), "window_wall_s": doc.get("wall_s"),
              "fct_frozen": doc.get("fct_frozen")}
    rec = ctx.receipt(result, ok and (strict is None or strict["pass"] or strict.get("how") == "not run"), strict)
    ok = rec["passed"]
    print(f"tangent: {ctx.steps} steps ({window_length_words(ctx.steps * ctx.row['dt_s'])}) of {ctx.row_name} from "
          f"{direction['doc']['name'].split(';')[0]}")
    for s in summary:
        print(f"  d{s['field']:<11} max |.| {s['max_abs_tangent']:.4g} {s['units']}   rms {s['rms_tangent']:.4g}")
    print_strict(strict)
    print(f"  valid {rec['valid']}")
    print(f"wrote {nc} and {ctx.out / 'receipt.json'}")
    print(used_line(rec))
    return 0 if ok else 1


def dot_bar_in_force(table: dict, args: Any = None) -> dict:
    """The whole-state dot's bar this command judges by (woof.hex.dot_bar): --dot-bar, else WOOF_HEX_DOT_BAR, else
    the table's default (roundings)."""
    try:
        return _dot_bar.bar_of(table, getattr(args, "dot_bar", None))
    except ValueError as error:
        raise AdDoorError(str(error)) from error


def self_dot_species_planes(plan, steps: int | None = None) -> bool:
    """Whether the whole-state self-dot gives each scalar species its own unit (AD_DOT_PLANES=scalars, the default).

    THE BREAKAGE THIS PREVENTS: with the 13 scalar species as ONE self-dot unit, aerosol number (about 1e9 per kg) sets
    the unit's weight and the water-vapour words weigh nothing: on the 6 km hrrrv5 plan (l6-s1h, 29 steps, TEMPO live)
    the qv plane's own transpose missed by 77 roundings while the one-unit dot read -40.3 roundings with no species
    named, and "TEMPO held passes" was read on that unit too, so the miss sat unseen for a day. Per species: +5.47.
    On where the plan's field map (alloc-map.json) has a plane-major float32 `scalars` field, which the runner needs to
    write the planes; WOOF_HEX_SELF_DOT_UNITS=one (or an AD_DOT_PLANES the caller set) keeps the old way.

    With ``steps``, the map must also hold the window-end boundary (steps + 1), which is where the runner reads the
    planes. THE BREAKAGE THIS PREVENTS: the README's 12 km example (adjoint-test --row hrrrv5-q12.96.2890.n40.60w91.30
    --steps 15, 19 Nov 05:49Z) printed "the species-unit window did not finish": its map holds boundaries
    1 to 15 and the runner stopped on boundary 16 after 0 s."""
    if os.environ.get("WOOF_HEX_SELF_DOT_UNITS", "").strip() == "one" or os.environ.get("AD_DOT_PLANES", "").strip():
        return False
    try:
        m = json.loads((Path(plan) / "alloc-map.json").read_text())
        f = (m.get("fields") or {}).get("scalars") or {}
        if steps is not None and str(int(steps) + 1) not in (m.get("boundaries") or {}):
            return False
        return len(f.get("shape") or []) == 3 and len(f.get("strides") or []) == 3 and (f.get("strides") or [0, 0, 0])[2] == 4
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def dot_terms(dot: dict) -> dict | None:
    """How well the self-form dot can see a miss: the reverse side's absolute terms (sum |x_i (M^T y)_i|) against the
    dot, and the miss in single-precision roundings of those terms. Printed beside every reading, never the verdict.

    THE BREAKAGE THIS PREVENTS: on the README's 3 km storm example the species-unit dot read +4.29 roundings, over the
    bar of 4, and nothing said why (19 Nov). The self form's bar takes the dot (y_Mx, a sum of squares that
    cannot cancel) as its rounding scale, but the reverse side's sum can cancel: there its terms were 128.6 against a
    dot of 22.0 (5.8 times; with the 13 species as one unit 34.1 against 12.0, 2.8 times), and the miss was 0.73 of a
    rounding of those terms (0.53 for the one-unit reading). One reverse per species (af-q3split) read every species
    measured: qv -0.33, qc -0.74, qr +0.78, qi +0.54, nwfa +0.03, nr -1.43 roundings (2^-24) of each one's own terms,
    with rain and ice bringing 11.5 and 16.5 times their vote of 1 in absolute terms; and cloud droplet number nc
    +5.50 roundings of its terms (24.9 times its vote), 72 per cent of the whole miss (af-q3nc). So on that example
    the whole dot read 0.73 of a rounding of its terms while one species missed by 5.5 of its own: the terms do not
    clear a reading. A judged reading on the terms would hide real misses (the 6 km qv plane's 77-rounding miss read
    +5.47 on the whole species dot), so the verdict keeps the dot's own scale, and this line only says how far the
    dot can see."""
    c = dot.get("conditioning") or {}
    y, x = dot.get("y_Mx"), dot.get("x_MTy")
    at, tl = c.get("abs_terms_ad"), c.get("abs_terms_tl")
    if not (isinstance(at, (int, float)) and isinstance(y, (int, float)) and isinstance(x, (int, float))) or not y:
        return None
    scale = max(abs(float(at)), abs(float(tl or 0.0)), abs(float(y)))
    return {"abs_terms_ad": at, "abs_terms_tl": tl, "ratio": scale / abs(float(y)),
            "roundings_of_terms": (float(x) - float(y)) / (_dot_bar.EPS32 * scale)}


def terms_words(terms: dict | None) -> str:
    """The conditioning clause printed beside a whole-state dot reading (dot_terms)."""
    if not terms:
        return ""
    return (f"; its reverse side's terms are {terms['ratio']:.1f} times the dot, and the miss is "
            f"{terms['roundings_of_terms']:+.2f} roundings of those terms")


def self_dot_check(doc: dict | None, bar: dict, left_out: dict | None = None,
                   proof: dict | None = None) -> dict:
    """The whole-state dot test that cannot cancel: the reverse is seeded with the tangent result itself, each unit
    scaled to unit weight, so <y, M x> is the number of units that carry tangent and <M^T y, x> must equal it. One
    wrong transpose in any field shows; the objective-direction dot can hide it. The units are the model's prognostic
    fields at the window end when the plan's field map (alloc-map.json) holds that boundary, and memory allocations
    otherwise; the runner says which (``units``) and why (``units_why``), and the door prints it.

    With ``left_out`` (the row's AD_DOT_DEAD list: scratch memory given zero weight) the check passes only with a
    passing ``proof`` on this plan (``dead_proof_check``): a masked dot under the bar with no proof is not a pass."""
    dots = (doc or {}).get("dots") or [{}]
    info = (doc or {}).get("self_dot") or {}
    # a runner without the whole-state dot writes no self_dot block: then there is no value, and no pass
    value = dots[0].get("rel_aligned") if info else None
    # the verdict in roundings of the dot (woof.hex.dot_bar), the same in every door: one direction within the bar;
    # several directions (--dot-directions) also free of one-signed drift
    one = _dot_bar.judge_one(dots[0].get("y_Mx"), dots[0].get("x_MTy"), bar, value) if info else \
        {"roundings": None, "rel": None, "pass": False}
    many = (_dot_bar.judge_many([_dot_bar.roundings(d.get("y_Mx"), d.get("x_MTy")) for d in dots], bar)
            if info and len(dots) > 1 else None)
    under = value is not None and (many["pass"] if many else one["pass"])
    proved = left_out is None or bool((proof or {}).get("ok"))
    units = info.get("units")
    return {"value": value, "bar": bar["rel"], "pass": under and proved, "terms": dot_terms(dots[0]) if info else None,
            "roundings": one["roundings"], "bar_roundings": bar["each_within_roundings"],
            "bar_setting": bar["setting"], "bar_words": bar["words"],
            "directions": None if many is None else {
                **many, "seeds": [d.get("seed") for d in dots], "rel": [d.get("rel_aligned") for d in dots]},
            "allocations": info.get("allocations"), "words": info.get("words"),
            "units": units, "units_why": info.get("units_why"),
            "y_Mx": dots[0].get("y_Mx"), "x_MTy": dots[0].get("x_MTy"),
            "left_out": left_out, "proof": proof,
            "what": "<y, M x> vs <M^T y, x> with y = the tangent result, every "
                    + ("model field at the window end" if units == "fields" else "state allocation")
                    + " at unit weight (float32 window; no cancellation)" + ("" if left_out is None else
                    f"; {left_out['bytes_total']} bytes of scratch in {left_out['ranges_n']} ranges given zero "
                    "weight (left_out), valid only with a passing NaN-poison proof window on this plan (proof)")}


def dead_ranges(path: Path) -> dict:
    """The row's left-out memory list (ranges in the runner's own allocation numbering, which is not plan.json's).
    Whether the list fits this plan is the proof window's question: the runner poisons exactly the listed bytes, and
    memory the model reads changes the gradient words."""
    if not path.is_file():
        raise AdDoorError(f"the row's left-out memory list {path} is missing from the asset root")
    ranges = [{"allocation": int(r["allocation"]), "offset": int(r["offset"]), "bytes": int(r["bytes"]),
               "name": r.get("name")} for r in json.loads(path.read_text()).get("ranges") or []]
    return {"list": str(path), "sha256": _sha256_file(path), "ranges": ranges, "ranges_n": len(ranges),
            "allocations": len({r["allocation"] for r in ranges}), "bytes_total": sum(r["bytes"] for r in ranges)}


def _gradient_words(doc: dict | None) -> str | None:
    rev = (((doc or {}).get("dots") or [{}])[0].get("reverse") or [{}])
    return (rev[0].get("result_sha256") or "")[:16] or None


def dead_proof_check(control: dict | None, poison: dict | None, poison_out: Path, left_out: dict | None = None) -> dict:
    """The proof that the left-out memory is never read: the same self-dot window with NaN written into the listed
    bytes before the reverse walk-back (AD_DOT_DEAD_POISON=rev) gives gradient words equal, bit for bit, to the window
    without the NaN, every boundary of both equal to the recording, and the runner says it wrote the NaN."""
    reasons = []
    events = []
    stderr = poison_out / "stderr"
    if stderr.is_file():
        for line in stderr.read_text(errors="replace").splitlines():
            if line.startswith("{") and '"window_dead_poison_rev"' in line:
                try:
                    events.append(json.loads(line))
                except ValueError:
                    pass
    walk = [{k: e.get(k) for k in ("step", "at", "ranges", "bytes")} for e in events]
    if not any(e.get("at") == "walk" for e in walk):
        reasons.append("the proof window wrote no NaN before a reverse walk-back (no window_dead_poison_rev at=walk)")
    elif left_out is not None and not any(e.get("at") == "walk" and e.get("bytes") == left_out["bytes_total"]
                                          for e in walk):
        reasons.append(f"the proof window poisoned {[e.get('bytes') for e in walk]} bytes, not the "
                       f"{left_out['bytes_total']} bytes the list leaves out")
    for what, doc in (("control", control), ("proof", poison)):
        v = (doc or {}).get("door_validity") or {}
        if not v.get("boundaries_checked") or v.get("boundaries_equal") != v.get("boundaries_checked"):
            reasons.append(f"{what} window: step boundaries equal {v.get('boundaries_equal')} of "
                           f"{v.get('boundaries_checked')}")
    words, control_words = _gradient_words(poison), _gradient_words(control)
    if words is None or words != control_words:
        reasons.append(f"gradient words {words} differ from the window without NaN ({control_words})")
    return {"ok": not reasons, "reasons": reasons, "run": str(poison_out), "words": words,
            "control_words": control_words, "poison_events": walk,
            "what": "NaN written into the left-out bytes before the reverse walk-back read them; gradient words equal "
                    "to the self-dot window without NaN, so the model never reads that memory"}


# What a bound adjoint-test receipt must share with this door's run to stand for its strict check: the same recorded
# forecast, the same operator row, the same runner build, window length, kernel set and refuse list. A path under the
# asset root is compared by its name below the root (two installs of one LATEST name the same files). runner_assets
# (every file the row hands the runner: kernel set, cubins, schemes, bundle, images, refuse list) pins the kernel set:
# one runner build serves several LATEST sets, so the runner's sha256 alone does not.
STRICT_BIND_KEYS = ("plan_sha256", "row", "mesh", "suite", "steps", "runner_sha256", "runner_assets", "kset_assets",
                    "arch_assets", "refused_by_name")


def _below_root(doc: dict, key: str):
    value, root = doc.get(key), doc.get("assets")
    if not isinstance(value, dict) or not root:
        return value
    out = {}
    for k, v in value.items():
        try:
            out[k] = str(Path(v).relative_to(root))
        except ValueError:
            out[k] = v
    return out


def bind_strict(ctx: "Context", path: Path) -> dict:
    """The strict check of a passing adjoint-test receipt on the same operator, bound instead of rerun (no card time).
    Refused by name when the receipt did not pass its own strict check with a proved left-out memory, or when any
    STRICT_BIND_KEYS differs from this run."""
    path = Path(path)
    try:
        bound = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise AdDoorError(f"--strict-from {path}: not a readable receipt ({error})") from error
    if bound.get("schema") != RECEIPT_SCHEMA or bound.get("door") != "adjoint-test":
        raise AdDoorError(f"--strict-from {path}: not an adjoint-test door receipt ({bound.get('schema')}, "
                          f"door {bound.get('door')})")
    if bound.get("planted_error"):   # a run broken on purpose (--plant-error) never stands for a real check
        raise AdDoorError(f"--strict-from {path}: that run had an error planted on purpose (--plant-error "
                          f"{bound['planted_error'].get('kernel')}:{bound['planted_error'].get('scale')}); it checks "
                          "nothing but the check itself")
    sd = ((bound.get("result") or {}).get("checks") or {}).get("self_dot") or {}
    # the checks of the operator (dot, FD, Taylor, zero and the whole-state dot): a receipt that names them apart
    # from its verdict over several test directions is bound on them (the strict check is the operator's, and a
    # direction the forecast cannot resolve says nothing about it)
    checks = (bound.get("result") or {}).get("checks_passed")
    if not ((bound.get("passed") if checks is None else checks) and bound.get("valid") is True and sd.get("pass")):
        raise AdDoorError(f"--strict-from {path}: the receipt did not pass its strict whole-state dot (passed "
                          f"{bound.get('passed')}, valid {bound.get('valid')}, self dot pass {sd.get('pass')})")
    if sd.get("left_out") and not (sd.get("proof") or {}).get("ok"):
        raise AdDoorError(f"--strict-from {path}: its left-out memory has no passing proof")
    if not bound.get("runner_assets"):
        raise AdDoorError(f"--strict-from {path}: the receipt does not name the files its row handed the runner "
                          "(runner_assets; receipts from before this door had none), so its kernel set cannot be "
                          "matched to this run: run adjoint-test again, or drop --strict-from to run the check here")
    mine = {"plan_sha256": ctx.fmap.doc["plan_sha256"], "row": ctx.row_name, "mesh": ctx.row["mesh"],
            "runner_assets": ctx.runner_assets(),
            "suite": ctx.row["suite"], "steps": ctx.steps, "runner_sha256": _sha256_file(ctx.runner),
            "assets": str(ctx.root), "kset_assets": {k: str(v) for k, v in ctx.kset_assets.items()},
            "arch_assets": {k: str(v) for k, v in ctx.arch_assets.items()}, "refused_by_name": ctx._refused_names()}
    differ = [k for k in STRICT_BIND_KEYS if _below_root(bound, k) != _below_root(mine, k)]
    if differ:
        raise AdDoorError(f"--strict-from {path}: the receipt is not of this operator: {', '.join(differ)} differ "
                          f"(receipt {[_below_root(bound, k) for k in differ]}, this run "
                          f"{[_below_root(mine, k) for k in differ]})")
    return {**sd, "how": "bound", "pass": True,
            "bound": {"receipt": str(path.resolve()), "receipt_sha256": _sha256_file(path),
                      "matched": {k: _below_root(mine, k) for k in STRICT_BIND_KEYS},
                      "bound_card": bound.get("card"), "bound_wall_s": bound.get("wall_s"),
                      "bound_card_peak_mib": bound.get("card_peak_mib")},
            "what": (sd.get("what") or "") + "; taken from a passing adjoint-test receipt of the same plan, row, "
                                             "runner, window length, kernel set and refuse list (bound, not rerun)"}


# ----------------------------------------------------------------------------- the strict check, once per plan
# THE BREAKAGE THIS PREVENTS: every sensitivity, tangent and param-grad command ran the strict check's two windows again
# on an operator nothing had changed in. At 120 km (fresh install, one 96 GB card, receipts f41) that was 127 of the
# 175.2 s of a sensitivity whose gradient window took 41.6 s. The check belongs to the operator, so its verdict is kept
# with a key of that operator and taken by the next door on the same key; nothing outside the key enters the check.
#: Where the door keeps its check verdicts (one file a key): default <hex cache>/ad/checks, beside the prepared plans.
CHECKS_ENV = "WOOF_HEX_AD_CHECKS"
CHECK_SCHEMA = "gpuwm-hex.ad-check-verdict/v1"
#: Words of the door's own environment the runner reads (it inherits the door's environment): a user's AD_* or HEXAD_*
#: changes the operator. HEXAD_LIBRARY_PATH is the door's own (the libraries are pinned by sha256 in the plan).
_KEY_ENV_PREFIXES = ("AD_", "HEXAD_")
#: Row keys that are not the operator's words: the verdicts shipped with its example (they would key themselves).
_ROW_NOT_WORDS = frozenset({"shipped_verdicts", "shipped_verdicts_what"})
#: The key parts a shipped verdict is matched on: plan, runner program, kernel set and runner files, card images and
#: family, steps, test direction, the check's words, cards and ranks; not the user's file stamps or the door program.
SHIPPED_PARTS = ("plan_sha256", "runner_sha256", "runner_assets", "kset_assets", "arch_assets", "card_family", "steps",
                 "seed", "check_sha256", "check_line", "left_out_list", "door_swap", "cards", "split_ranks", "row",
                 "row_sha256")


def shipped_key_sha256(key: dict | None) -> str | None:
    """The sha256 a shipped verdict names: the key's SHIPPED_PARTS (the receipt's strict_check.shipped_key_sha256)."""
    return None if key is None else _json_sum({k: key.get(k) for k in SHIPPED_PARTS})


def shipped_verdict(row: dict, key: dict | None, plan_how: str | None) -> tuple[dict | None, str | None]:
    """The publisher's verdict shipped with this row's packed example whose key matches this run's, or (None, why
    not; None when the row ships none or the plan is not the packed example). Never for a plan of the user's own."""
    entries = row.get("shipped_verdicts") or []
    if not entries or key is None or not str(plan_how or "").startswith("the row's example plan"):
        return None, None
    want = shipped_key_sha256(key)
    for entry in entries:
        if entry.get("key_sha256") == want:
            return entry, None
    return None, ("the verdict shipped with this example was made on another plan, runner program, kernel set or "
                  "card family")
_KEY_ENV_SKIP = frozenset({"HEXAD_LIBRARY_PATH"})
#: What a Context must hold for its operator to be named by a key (a context without them never reuses or stores).
_KEY_PARTS = ("fmap", "plan", "runner", "row_name", "row", "steps", "assets", "kset_assets", "arch_assets", "root",
              "runner_assets", "table")


def checks_dir() -> Path:
    declared = os.environ.get(CHECKS_ENV)
    if declared:
        return Path(declared).expanduser().absolute()
    from . import assets as _assets
    return _assets.cache_root() / "ad" / "checks"


def _json_sum(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _content_sum(path: Path | None) -> str | None:
    """sha256 of a file's bytes, or of a folder's files (each one's name below the folder and sha256, in name order);
    None for a path that is not there. The runner's files are read whole: about 215 MB at 120 km, under a second."""
    if path is None:
        return None
    from .ad_plan_facts import kept_sum
    return kept_sum(Path(path), _content_sum_read)


def _content_sum_read(path: Path) -> str | None:
    """``_content_sum`` from the files themselves (kept by ``_content_sum`` under their stamps)."""
    if path.is_file():
        return _sha256_file(path)
    if not path.is_dir():
        return None
    digest = hashlib.sha256()
    for sub in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(f"{sub.relative_to(path).as_posix()}\0{_sha256_file(sub)}\n".encode("utf-8"))
    return digest.hexdigest()


def _plan_files_sum(plan: Path) -> dict:
    """Every file of the plan package by name, size, inode and change times, and every link by its target. A plan is
    gigabytes (2.1 GB at 120 km), so its files are not read: a file written, touched, replaced or added since the
    verdict changes the sum (a write moves the change time, which no user call can set back)."""
    plan = Path(plan)
    digest, files = hashlib.sha256(), 0
    for dirpath, dirnames, filenames in os.walk(plan):   # links are listed, never followed (<site-packages>)
        dirnames.sort()
        base = Path(dirpath)
        for name in sorted(dirnames + filenames):
            p = base / name
            st = p.lstat()
            rel = p.relative_to(plan).as_posix()
            if os.path.islink(p):
                record = f"{rel}\0link\0{os.readlink(p)}"
            elif p.is_dir():
                continue
            else:
                record = f"{rel}\0{st.st_size}\0{st.st_ino}\0{st.st_mtime_ns}\0{st.st_ctime_ns}"
            digest.update(record.encode("utf-8", "surrogateescape") + b"\n")
            files += 1
    return {"files": files, "sha256": digest.hexdigest()}


def check_key(ctx: "Context") -> dict | None:
    """The key a strict-check verdict is kept under: what the two windows depend on and nothing else (not the
    objective, the metric, the region, the output folder or the card's index). None when the context cannot name its
    operator (then the check runs and nothing is stored)."""
    if any(not hasattr(ctx, part) for part in _KEY_PARTS) or not getattr(ctx, "arch", None):
        return None
    from . import __version__
    table, door = ctx.table, getattr(ctx, "door", None)
    spec = table["doors"]["adjoint-test"]
    seed = getattr(getattr(ctx, "args", None), "seed", None)
    dead = (ctx.row.get("self_dot_dead") or {}).get("list")
    split = getattr(ctx, "split", None)
    env = sorted((k, v) for k, v in os.environ.items() if k.startswith(_KEY_ENV_PREFIXES) and k not in _KEY_ENV_SKIP)
    # the route's words every window took (route_window_words), named in the key only where they exist, so a verdict
    # of one boundary-layer route never stands for another and every other key is as before
    try:
        route = ({"route_env": _check_line_key({"extra": [], "env": ctx.route_env})}
                 if getattr(ctx, "route_env", None) else {})
        ndir = int(getattr(getattr(ctx, "args", None), "dot_directions", None) or 1)
        if ndir > 1:   # a verdict over several directions is kept apart from the one-direction verdicts
            route["dot_directions"] = ndir
        bar = dot_bar_in_force(table, getattr(ctx, "args", None))
        return {**route, "schema": CHECK_SCHEMA,
                "plan_sha256": ctx.fmap.doc["plan_sha256"], "plan_files": _plan_files_sum(ctx.plan),
                "row": ctx.row_name, "row_sha256": _json_sum({k: v for k, v in ctx.row.items()
                                                              if k not in _ROW_NOT_WORDS}), "steps": int(ctx.steps),
                "seed": 1 if seed is None else int(seed),
                "runner_sha256": _sha256_file(ctx.runner),
                "door_code": {"version": __version__, "ad_doors_sha256": _sha256_file(Path(__file__))},
                "check_sha256": _json_sum({"self_dot": spec.get("self_dot"), "self_dot_args": spec.get("self_dot_args"),
                                           "bar": {k: bar[k] for k in ("setting", "rel", "drift")}}),
                "runner_assets": {k: _content_sum(ctx.assets.get(k)) for k in sorted(ctx.runner_assets())},
                "kset_assets": {k: _content_sum(v) for k, v in sorted(ctx.kset_assets.items())},
                "arch_assets": {k: _content_sum(v) for k, v in sorted(ctx.arch_assets.items())},
                "left_out_list": _content_sum(ctx.root / dead) if dead else None,
                "door_swap": _door_asset_env(table, door) or None,
                "card_family": ctx.arch,
                "cards": len(getattr(ctx, "cards", None) or [None]),
                "split_ranks": (split or {}).get("ranks"),
                "runner_env_sha256": _json_sum(env) if env else None,
                "check_line": _check_line_key(getattr(ctx, "check_words", None))}
    except OSError as error:   # a file of the operator that cannot be read: the check runs, nothing is kept
        print(f"woof hex {door}: answer check: the operator's key cannot be read ({error}); the check runs and its "
              "verdict is not kept", file=sys.stderr, flush=True)
        return None


def check_verdict_path(key: dict) -> Path:
    return checks_dir() / f"{key['plan_sha256'][:16]}-{_json_sum(key)[:32]}.json"


def _key_differs(old: dict, new: dict) -> list[str]:
    return [k for k in sorted(set(old) | set(new)) if old.get(k) != new.get(k)]


#: What a changed key part is called in the door's line (the parts not named here are called by their key word).
_KEY_WORDS = {"plan_sha256": "plan", "plan_files": "plan files", "row_sha256": "row words", "runner_sha256": "runner",
              "door_code": "door program", "check_sha256": "check words", "runner_assets": "kernel set or runner files",
              "kset_assets": "kernel set", "arch_assets": "card images", "left_out_list": "left-out list",
              "card_family": "card family", "seed": "test direction", "runner_env_sha256": "runner environment"}


def stored_check(ctx: "Context", key: dict) -> tuple[dict | None, str | None]:
    """A passing verdict kept under ``key`` as this door's strict check, or (None, why it is not taken). A verdict is
    taken only when its key equals this run's, its own check passed with any left-out memory proved, its windows
    were valid and its dot is under today's bar."""
    path = check_verdict_path(key)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        doc = None
    except (OSError, ValueError) as error:
        return None, f"the stored verdict {path} cannot be read ({error})"
    if doc is None:
        older = []
        for other in checks_dir().glob(f"{key['plan_sha256'][:16]}-*.json"):
            try:
                older.append((other.stat().st_mtime, json.loads(other.read_text(encoding="utf-8"))))
            except (OSError, ValueError):
                continue
        if not older:
            return None, None   # never checked on this plan: the first command checks it
        last = max(older, key=lambda x: x[0])[1]
        differ = _key_differs(last.get("key") or {}, key)
        return None, ("the verdict stored on this plan on " + str(last.get("checked_utc")) + " was made with another "
                      + ", ".join(_KEY_WORDS.get(k, k) for k in differ))
    check = doc.get("check") or {}
    bar = dot_bar_in_force(ctx.table, getattr(ctx, "args", None))["rel"]
    value = check.get("value")
    reasons = []
    if doc.get("schema") != CHECK_SCHEMA or doc.get("key") != key:
        reasons.append("its key is not this run's")
    if not (doc.get("pass") is True and check.get("pass") is True and check.get("how") == "own"):
        reasons.append("it did not pass")
    if check.get("left_out") and not (check.get("proof") or {}).get("ok"):
        reasons.append("its left-out memory has no passing proof")
    windows = doc.get("windows") or []
    if not windows or not all(w.get("valid") is True for w in windows):
        reasons.append("its windows were not all valid")
    if not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > bar:
        reasons.append(f"its dot {value} is not within the bar {bar:.3g} "
                       f"({bar / _dot_bar.EPS32:g} single-precision roundings of the dot)")
    if reasons:
        return None, f"the verdict stored on this plan on {doc.get('checked_utc')} is not taken: {'; '.join(reasons)}"
    stored = {"verdict": str(path), "verdict_sha256": _sha256_file(path), "checked_utc": doc.get("checked_utc"),
              "by_door": doc.get("door"), "receipt": doc.get("receipt"), "receipt_sha256": doc.get("receipt_sha256"),
              "receipt_present": bool(doc.get("receipt")) and Path(doc["receipt"]).is_file(),
              "key_sha256": doc.get("key_sha256"), "windows": windows}
    return {**check, "how": "stored", "pass": True, "windows": [], "stored": stored,
            "what": (check.get("what") or "") + f"; run once on this operator by the {doc.get('door')} command on "
                    f"{doc.get('checked_utc')} and stored under its key (plan and its files, runner, kernel set and "
                    "runner files, card family, steps, row words, test direction, door program): not run again"}, None


def store_check(ctx: "Context", strict: dict, receipt: Path) -> Path | None:
    """Keep the verdict of a check this door ran itself (passing or not; only a passing one is ever taken), under the
    key strict_check named. A cache that cannot be written is said, and the next command runs the check again."""
    key = getattr(ctx, "check_key", None)
    if key is None or strict.get("how") != "own":
        return None
    names = set(strict.get("windows") or [])
    windows = [w for w in getattr(ctx, "windows", []) if w.get("window") in names]
    path = check_verdict_path(key)
    doc = {"schema": CHECK_SCHEMA, "key": key, "key_sha256": _json_sum(key),
           "checked_utc": time.strftime("%Y-%m-%d %H:%MZ", time.gmtime()), "door": getattr(ctx, "door", None),
           "receipt": str(receipt), "receipt_sha256": _sha256_file(receipt) if receipt.is_file() else None,
           "pass": bool(strict.get("pass")) and len(windows) == len(names) and all(w.get("valid") is True for w in windows),
           "windows": windows, "check": strict}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    except OSError as error:
        print(f"woof hex {getattr(ctx, 'door', None)}: answer check: its verdict was not kept ({error}); the next "
              "command on this plan runs it again", file=sys.stderr, flush=True)
        return None
    return path


def _shipped_check(ctx: "Context", door: str | None, shipped: dict) -> dict:
    """A verdict shipped with the packed example, taken: one line says who made it, on which card family and date,
    its reading (a miss said as a miss) and that --strict reruns it on this card."""
    # judged by today's bar; a verdict the publisher made under a stricter bar stands, one made under a looser bar
    # is judged again on its reading
    today = dot_bar_in_force(ctx.table, getattr(ctx, "args", None))
    bar = min(float(shipped.get("bar") or today["rel"]), today["rel"])
    value = shipped.get("value")
    ok = bool(shipped.get("pass")) and value is not None and float(value) <= bar
    r = _dot_bar.roundings_of_rel(value)
    print(f"woof hex {door}: answer check: the verdict shipped with this example was made by the publisher on an "
          f"{shipped.get('card_family')} card ({shipped.get('card') or 'card family ' + str(shipped.get('card_family'))})"
          f" on {shipped.get('measured_utc')}: whole-state dot {float(value):.3e} relative, "
          f"{'n/a' if r is None else format(r, '.2f')} roundings, against {bar:.3g} "
          f"({bar / _dot_bar.EPS32:g} roundings), "
          f"{'within the bar' if ok else 'NOT within the bar'}; --strict reruns it on your card", flush=True)
    return {"how": "shipped", "value": value, "bar": bar, "pass": ok, "allocations": shipped.get("allocations"),
            "left_out": None, "proof": None, "windows": [], "seed": shipped.get("seed", 1),
            "shipped": {**shipped, "what": "made by the publisher on that card family and date, on this packed "
                                           "example's plan, runner program and kernel set; --strict reruns it here"}}


def strict_check(ctx: "Context", no_self_dot: bool = False, strict_from: Path | None = None) -> dict | None:
    """The strict check (strict_check_inner) with the direction it tested and how long its window was named, and the
    row's readings in other directions beside it: a whole-state PASS is never printed as if it covered a storm
    objective's direction."""
    check = strict_check_inner(ctx, no_self_dot, strict_from)
    if not check or check.get("how") == "not run":
        return check
    spec = ctx.table["doors"]["adjoint-test"].get("self_dot") or {}
    steps = int(getattr(ctx, "steps", 0) or 0)
    minutes = steps * float(ctx.row.get("dt_s") or 0.0) / 60.0
    return {**check, "direction": f"{spec.get('direction') or 'the whole-state direction'}; over {steps} steps "
                                  f"({minutes:g} minutes of forecast)",
            "other_directions": ctx.row.get("check_other_directions")}


def strict_check_inner(ctx: "Context", no_self_dot: bool = False, strict_from: Path | None = None) -> dict | None:
    """The strict whole-state check of this door's operator, with its left-out memory printed and proved.

    Every door replays the same operator (plan, row, runner, kernel set, refuse list), so every door can carry the
    same check: by default its own two windows (the self dot on the runner's seeded whole-state direction, then the
    NaN-poison proof of the row's left-out memory, as adjoint-test runs them); with ``strict_from`` the check of a
    passing adjoint-test receipt on the same operator is bound instead. THE BREAKAGE THIS PREVENTS: at 120 km the
    tangent and sensitivity receipts read valid with no strict check of their own (their windows read
    valid_derivative false; 2026-11-11); their validity rested on another receipt nobody bound.

    Once per plan (the table's ``self_dot.once_per_plan``): a door named in ``reuse`` takes a passing verdict kept under
    this operator's key (``check_key``) by an earlier command instead of the two windows; the first command on a key,
    a door named in ``always`` (adjoint-test), --self-dot, a failed or unreadable verdict and any change of key run
    them, and their verdict is kept by the receipt (``store_check``)."""
    spec = ctx.table["doors"]["adjoint-test"].get("self_dot")
    if not spec:
        return None
    if no_self_dot:
        return {"how": "not run", "pass": False,
                "what": "--no-self-dot: the whole-state dot was not run; this receipt's validity is not checked"}
    seed = getattr(getattr(ctx, "args", None), "seed", None)
    if strict_from is not None:
        if seed is not None:
            raise AdDoorError("--seed and --strict-from are alternatives: a bound receipt keeps its own direction")
        return bind_strict(ctx, strict_from)
    # Once per plan (the table's self_dot.once_per_plan): a passing verdict stored on this operator's key stands for
    # the two windows; any change of key, a failed or missing verdict, --self-dot or a door that always checks runs them.
    door = getattr(ctx, "door", None)
    once = spec.get("once_per_plan") or {}
    key = check_key(ctx) if once else None
    if key is not None:
        forced = bool(getattr(getattr(ctx, "args", None), "self_dot", False))
        if door in (once.get("reuse") or []) and door not in (once.get("always") or []) and not forced:
            stored, why = stored_check(ctx, key)
            if stored is None:
                shipped, why_not = shipped_verdict(ctx.row, key, getattr(ctx, "plan_how", None))
                if shipped is not None:
                    return _shipped_check(ctx, door, shipped)
                why = why or why_not
            if stored is not None:
                print(f"woof hex {door}: answer check passed on this plan on {stored['stored']['checked_utc']}: "
                      f"whole-state dot {stored['value']:.3e} against {stored['bar']:.3g} "
                      f"({_dot_bar.roundings_of_rel(stored['value']):.2f} of {stored['bar'] / _dot_bar.EPS32:g} "
                      f"single-precision roundings) in the whole-state "
                      f"direction over {int(ctx.steps)} steps; not run again; --strict runs it (verdict "
                      f"{stored['stored']['verdict']})", flush=True)
                return stored
            print(f"woof hex {door}: answer check: "
                  + (why if why else "not yet run on this plan") + "; running it now (its verdict is kept for the "
                  "next commands on this plan)", flush=True)
        elif forced:
            print(f"woof hex {door}: answer check: --self-dot: running it on this plan (a stored verdict is not used)",
                  flush=True)
        if door in (once.get("store_from") or []) and not getattr(ctx, "plant", None):
            # a planted run's verdict is never kept: no later command may stand on a derivative broken on purpose
            ctx.check_key = key   # the receipt stores this check's verdict under it (Context.receipt)
    args_key = "self_dot_args"
    ndir = int(getattr(getattr(ctx, "args", None), "dot_directions", None) or 1)
    if seed is not None or ndir > 1:
        ad_spec = ctx.table["doors"]["adjoint-test"]
        ad_spec["self_dot_args_seed"] = self_dot_args_seeded(ad_spec["self_dot_args"], seed, ndir)
        args_key = "self_dot_args_seed"
    bar = dot_bar_in_force(ctx.table, getattr(ctx, "args", None))
    env = {**dict(spec["window_env"]), **_dot_bar.runner_env(bar)}
    dead = ctx.row.get("self_dot_dead")
    left_out = proof = None
    if dead:   # the row's left-out scratch memory (fold 1's opt-in), on by default and proved on this plan
        left_out = dead_ranges(ctx.root / dead["list"])
        left_out["recipe_proof"] = dead.get("recipe_proof")
        env["AD_DOT_DEAD"] = left_out["list"]
    # The strict windows run the operator THIS door runs: a door that swaps a runner file (param-grad: the
    # parameter-carrier bundle for the row's scheme bundle) is checked on that file, not on the row's. THE BREAKAGE
    # THIS PREVENTS: a param-grad receipt reading valid on the dot of the row's plain bundle, a derivative it did not run.
    swap = _door_asset_env(ctx.table, getattr(ctx, "door", None))
    own = {"env_overrides": swap} if swap else {}
    cw = getattr(ctx, "check_words", None)   # the gradient line's words the check can take (check_line_words)
    if cw:
        env = {**cw["env"], **env}
    line_extra = list(cw["extra"]) if cw else []
    if cw and cw.get("replace"):
        own["replace_args"] = dict(cw["replace"])
    sn = getattr(ctx, "store_need", None) or {}
    if (sn.get("host_check_mib") and getattr(ctx, "check_expected", True) is False
            and not getattr(getattr(ctx, "args", None), "need_host_mib", None)):
        # admitted on the gradient alone (a verdict looked kept) and the check runs after all: judge its own need
        refuse_host_too_small(ctx.row_name, int(sn["host_check_mib"]), host_available_mib(),
                              what="the answer check, which this command runs after all",
                              smaller="--no-self-dot runs the gradient without its answer check (the receipt then "
                                      "reads not checked).")
    planes = self_dot_species_planes(getattr(ctx, "plan", None) or ".", getattr(ctx, "steps", None))
    # a proven (shipped) row keeps judging on the 13 species as one unit, the verdict its release printed, and prints
    # the species-unit reading beside it. THE BREAKAGE THIS PREVENTS: on the README's 3 km storm example (sensitivity
    # --row hrrrv5-q3.96.38278.n40.60w91.30 --steps 59, 19 Nov 03:18Z) the species-unit default read +4.29
    # roundings, FAIL, where the release printed -1.50, PASS; the one-unit window of the same run read -1.50 again
    beside = None
    if planes and not (getattr(ctx, "row", None) or {}).get("generated"):
        planes, beside = False, True
    if planes:
        env["AD_DOT_PLANES"] = "scalars"   # one unit per scalar species (self_dot_species_planes)
    whole = ctx.window(list(line_extra), sub="selfdot", door_args_key=args_key, extra_env=env,
                       spec_door="adjoint-test", **own)
    if beside and os.environ.get("WOOF_HEX_SELF_DOT_ONE_UNIT", "1").strip() != "0":
        try:
            cs = self_dot_check(ctx.window(list(line_extra), sub="selfdot-species", door_args_key=args_key,
                                           extra_env={**env, "AD_DOT_PLANES": "scalars"}, spec_door="adjoint-test",
                                           **own), bar)
            beside = {"value": cs["value"], "roundings": cs["roundings"], "pass": cs["pass"],
                      "terms": cs.get("terms")}
        except AdDoorError as error:
            beside = {"error": f"the species-unit window did not finish: {error}"}
    elif beside:
        beside = {"error": "not run (WOOF_HEX_SELF_DOT_ONE_UNIT=0)"}
    one_unit = None
    if planes and os.environ.get("WOOF_HEX_SELF_DOT_ONE_UNIT", "1").strip() != "0":
        # the old reading beside the new: the same window with the 13 species as one unit (printed, not judged);
        # WOOF_HEX_SELF_DOT_ONE_UNIT=0 skips this second window
        env1 = {k: v for k, v in env.items() if k not in ("AD_DOT_PLANES", "AD_DOT_SELF_SPLIT")}
        try:
            c1 = self_dot_check(ctx.window(list(line_extra), sub="selfdot-one-unit", door_args_key=args_key,
                                           extra_env=env1, spec_door="adjoint-test", **own), bar)
            one_unit = {"value": c1["value"], "roundings": c1["roundings"], "pass": c1["pass"],
                      "terms": c1.get("terms")}
        except AdDoorError as error:
            one_unit = {"error": f"the one-unit window did not finish: {error}"}
    if dead:
        try:
            poison = ctx.window(list(line_extra), sub="selfdot-proof", door_args_key=args_key,
                                extra_env={**env, "AD_DOT_DEAD_POISON": "rev"}, spec_door="adjoint-test", **own)
        except AdDoorError as error:
            poison = None
            proof = {"ok": False, "reasons": [f"the proof window did not finish: {error}"]}
        if poison is not None:
            proof = dead_proof_check(whole, poison, ctx.out / "selfdot-proof", left_out)
    return {**self_dot_check(whole, bar, left_out, proof), "how": "own",
            "species_units": planes, "one_unit": one_unit, "species_beside": beside,
            "seed": (1 if seed is None else seed) if ndir == 1 else f"{seed or 1}..{(seed or 1) + ndir - 1}",
            "windows": ["selfdot"] + (["selfdot-proof"] if dead else []) + (["selfdot-one-unit"] if one_unit else [])
                       + (["selfdot-species"] if beside and "value" in beside else []),
            "words": ({"route": "the gradient line's words the check can take", "extra": line_extra,
                       "not_taken": cw["not_taken"]} if cw else {"route": "the row's own words"}),
            "shipped_key_sha256": shipped_key_sha256(key)}


def print_strict(check: dict | None) -> None:
    """The strict check's lines, the same in every door: the dot, its allocations, the left-out memory and its proof."""
    if not check:
        return
    if check.get("how") == "not run":
        print("  whole-state dot  not run (--no-self-dot): this receipt's validity is not checked")
        return
    fmt = lambda v: "n/a" if v is None else f"{v:.3e}"
    if check.get("how") == "bound":
        how = f"bound from {check['bound']['receipt']}"
    elif check.get("how") == "stored":
        s = check["stored"]
        how = (f"run once on this plan on {s['checked_utc']} by {s['by_door']} (receipt {s['receipt']}), stored, "
               "not run again; --self-dot runs it")
    elif check.get("how") == "same call":
        how = f"the check of this call's first objective (receipt {check['same_call']['receipt']})"
    elif check.get("how") == "shipped":
        s = check["shipped"]
        how = (f"shipped with this example: made by the publisher on an {s.get('card_family')} card on "
               f"{s.get('measured_utc')}; --strict reruns it on this card")
    else:
        how = "own windows"
    seed = f"; test direction {check['seed']}" if check.get("seed") is not None else ""
    scored = "model fields" if check.get("units") == "fields" else "state allocations"
    r = check.get("roundings")
    if r is None:
        r = _dot_bar.roundings_of_rel(check.get("value"))
    k = check.get("bar_roundings") or check["bar"] / _dot_bar.EPS32
    print(f"  whole-state dot  {fmt(check.get('value'))} relative = "
          f"{'n/a' if r is None else format(r, '+.2f' if check.get('roundings') is not None else '.2f')} roundings   "
          f"bar {k:g} roundings ({check['bar']:.2e})   "
          f"{'PASS' if check.get('pass') else 'FAIL'}   ({check.get('allocations')} {scored}; {how}{seed})")
    if check.get("terms"):
        print(f"    conditioning: the bar's rounding is of the dot{terms_words(check['terms'])} (printed, not judged: "
              "where the terms are several times the dot, a miss of a few roundings of the dot can be the float32 "
              "sum; the whole dot cannot tell that from a small miss in one field, a reverse per field can)")
    if check.get("species_units"):
        o = check.get("one_unit") or {}
        old = (o.get("error") or ("n/a" if o.get("roundings") is None else f"{o['roundings']:+.2f} roundings "
               f"({fmt(o.get('value'))} relative{terms_words(o.get('terms'))})")) if o else \
            "not run (WOOF_HEX_SELF_DOT_ONE_UNIT=0)"
        print(f"    units: one per scalar species (the default); the 13 species as one unit (the old reading): {old}")
    elif check.get("species_beside"):
        o = check["species_beside"]
        new = (o.get("error") or ("n/a" if o.get("roundings") is None else f"{o['roundings']:+.2f} roundings "
               f"({fmt(o.get('value'))} relative{terms_words(o.get('terms'))})"))
        print(f"    units: the 13 species as one unit (this proven row's verdict, as its release printed it); one per "
              f"scalar species (the made rows' default, printed, not judged): {new}")
    dirs = check.get("directions")
    if dirs:
        print("    directions " + ", ".join(f"{s}: {'n/a' if v is None else format(v, '+.2f')}"
                                          for s, v in zip(dirs.get("seeds") or [], dirs["readings"]))
              + f" roundings; each within {k:g}: {'yes' if dirs['within_each'] else 'NO'}")
        print(f"    drift: {dirs['positive']} of {dirs['n']} positive; {dirs['no_drift_why']}")
    words = check.get("bar_words")
    if words:
        print(f"    the bar: {words}")
    if check.get("units") == "pool" and check.get("units_why"):
        print(f"    scored on memory allocations, not the model's fields: {check['units_why']}")
    if check.get("direction"):
        # the table's text names test direction 1, the default; a --seed or --dot-directions run names its own
        said = check["direction"]
        if check.get("seed") is not None and str(check["seed"]) != "1":
            many = ".." in str(check["seed"])   # --dot-directions N: a range of test directions
            said = said.replace("test direction 1", f"test direction{'s' if many else ''} {check['seed']}")
        print(f"    direction: {said}")
    if check.get("other_directions"):
        print(f"    other directions: {check['other_directions']}")
    lo, pr = check.get("left_out"), check.get("proof") or {}
    if lo:
        print(f"    left out of the dot: {lo['bytes_total']} bytes in {lo['ranges_n']} ranges of "
              f"{lo['allocations']} allocations ({lo['list']}); NaN-poison proof on this plan: "
              f"{'PASS' if pr.get('ok') else 'FAIL ' + '; '.join(pr.get('reasons') or [])}"
              f"{' (words ' + str(pr.get('words')) + ' = control ' + str(pr.get('control_words')) + ')' if pr.get('ok') else ''}")
    else:
        print("    left out of the dot: nothing (the row lists no left-out memory)")


def thresholds_of(ctx: "Context") -> dict:
    return ctx.table["thresholds"]


def used_line(doc: dict) -> str:
    """The door's last line: what it used, from its receipt (the fullest card's peak, the host peak, the wall time)."""
    card = doc.get("card_peak_mib")
    host = doc.get("host_peak_mib")
    before = doc.get("before_clock_s")
    return (f"used: card peak {'n/a' if card is None else format(card, ',')} MiB, host peak "
            f"{'n/a' if host is None else format(host, ',')} MiB, wall {doc.get('wall_s')} s"
            + ("" if before is None else f" (and {before} s before that clock started)"))


SETTLE_FACTOR = 10.0


def settled_rows(hs: list, values: list, factor: float = SETTLE_FACTOR) -> tuple:
    """The step sizes whose error a neighbouring step size confirms, and the ones set aside.

    The Taylor h grid steps by about 3 (1, 0.3, 0.1, ...). A real error curve never changes faster than that between
    neighbours: truncation error falls as h, the real model's noise grows as 1/h. Only a sign crossing of noise reads
    10 or 1000 times smaller than both neighbours; on the zoom globe such a row read 3.4e-5 between 6.4e-2 and 0.65,
    and the door printed it as the agreement. A row counts when a neighbour reads within `factor` of it.
    """
    kept, aside = [], []
    for i, v in enumerate(values):
        nb = [values[j] for j in (i - 1, i + 1) if 0 <= j < len(values)]
        ok = any(max(v, n) <= factor * max(min(v, n), 1e-300) for n in nb)
        (kept if ok else aside).append((hs[i], v))
    return kept, aside


def adjoint_verdict(doc: dict, thresholds: dict, probe: float | None = None) -> dict:
    """The three numbers (dot, FD, Taylor) and the zero pass from one window receipt.

    With ``probe`` (a storm measure; roughness_view) the FD and Taylor checks read the centred step sizes whose
    roughness bound is at most half their bar; with none such, each check is UNRESOLVED (``unresolved``: neither a
    pass nor a failure; the command cannot then pass)."""
    doc, rough = roughness_view(doc, probe)
    td = doc.get("tangent_dir") or {}
    taylor = doc.get("taylor") or {}
    # A row whose err is null (the tangent along the objective is zero, so the window could not divide) is no step
    # size a check can read. THE BREAKAGE THIS PREVENTS: such rows stopped the door with a TypeError in settled_rows
    # before its whole-state check ran (a window with every kind on the path to the objective held, 18 Nov).
    rows = sorted((r for r in taylor.get("rows", []) if r.get("err") is not None), key=lambda r: -r["h"])
    all_rows = rows
    if rough:   # only the step sizes the roughness lets the checks read
        rows = [r for r in rows if rough_counts(r, min(thresholds["fd_tangent_err"], thresholds["taylor_best_err"]))]
    w_mx = td.get("w_Mx")
    dot = td.get("rel")
    hs = [r["h"] for r in rows]
    fd = [abs(r["fd_quotient"] / w_mx - 1.0) for r in rows] if w_mx not in (None, 0) else []
    # A lucky single h is not a pass: each value is the best step size that a neighbouring step size confirms
    # (settled_rows); a row far below both neighbours is a sign crossing of the real model's noise and is set aside.
    fd_kept, fd_aside = settled_rows(hs, fd) if fd else ([], [])
    fd_best = min(v for _, v in fd_kept) if fd_kept else None
    kept, aside = settled_rows(hs, [r["err"] for r in rows])
    best = min(v for _, v in kept) if kept else None
    resolved = bool(kept)
    zero = (doc.get("zero_direction") or {}).get("nonzero_words")
    checks = {
        "dot": {"value": dot, "bar": thresholds["dot_rel"], "pass": dot is not None and dot <= thresholds["dot_rel"],
                "what": "<w, M x> (tangent) vs <M^T w, x> (adjoint), relative"},
        "fd": {"value": fd_best, "bar": thresholds["fd_tangent_err"],
               "pass": fd_best is not None and fd_best <= thresholds["fd_tangent_err"],
               "set_aside_h": [h for h, _ in fd_aside],
               "what": "min over the confirmed h of |(J(x0 + h x) - J(x0)) / h / <w, M x> - 1|: the tangent against "
                       "forward differences (an h counts when a neighbouring h reads within 10 times it)"},
        "taylor": {"value": best, "bar": thresholds["taylor_best_err"],
                   "pass": best is not None and best <= thresholds["taylor_best_err"] and resolved,
                   "resolved_interval": resolved, "set_aside_h": [h for h, _ in aside],
                   "what": "min over the confirmed h of |dJ / (h <x, M^T w>) - 1|: the adjoint gradient against "
                           "forward differences (an h counts when a neighbouring h reads within 10 times it; a row "
                           "far below both neighbours is a sign crossing of the real model's noise)"},
        "zero": {"value": zero, "bar": thresholds["zero_nonzero_words"],
                 "pass": zero is not None and zero <= thresholds["zero_nonzero_words"],
                 "what": "nonzero tangent words after the whole window from x = 0"},
    }
    if rough and not rows:   # the roughness reads over half the bar at every step size: not a reading either way
        for key in ("fd", "taylor"):
            checks[key].update(unresolved=True, why=(
                f"the forecast's own roughness (a nudge of {rough['probe_h']:g} moved the measure by "
                f"{rough['floor']:.2e}) moves the ratio by more than half the bar at every step size"))
    def field(r: dict) -> dict:
        # the metric field at the window end, word by word: the real model's difference quotient against the tangent
        f = (r.get("state_fields") or [{}])[0]
        return {"field_fd_minus_tl_rel_l2": f.get("fd_minus_tl_rel_l2"), "field_fd_tl_cos": f.get("fd_tl_cos"),
                "field_changed_words": f.get("changed_words")}

    return {"checks": checks, "passed": all(c["pass"] or c.get("unresolved") for c in checks.values()),
            **({"roughness": rough} if rough else {}),
            "taylor_rows": [{"h": r["h"], "fd_quotient": r["fd_quotient"], "ratio": r["ratio"], "err": r["err"],
                             **({"noise": r.get("noise"), "centred": True} if rough else {}),
                             **field(r)} for r in all_rows], "w_Mx": w_mx, "x_MTw": td.get("x_MTw"),
            "taylor_rows_what": "per step size h (the bump is h times the perturbation): err = |dJ / (h <x, M^T w>) "
                                "- 1| on the scalar metric; field_fd_minus_tl_rel_l2 = |FD - TL| / |TL| over the "
                                "whole metric field at the window end (FD = the real model's difference quotient). "
                                "A step size below the real model's own noise reads large on the field and "
                                "erratic on the metric; the largest step sizes are the resolved ones"}


# ------------------------------------------------------------ adjoint-test: several independent test directions
# THE BREAKAGE THIS PREVENTS: the gradient test ran its finite-difference and Taylor legs along ONE direction (a
# mid-level temperature bump), and its PASS was read as a verdict on the gradient. A test of the same
# engine along a near-ground moisture direction read 1.23 (the real change over the gradient's prediction) where the
# steepest-descent direction read 0.85, and a smooth random direction changed the measure by less than the forecast's
# own roughness, so it tested nothing; and a "second" direction drawn with the seed of another draw was that draw.
# The directions are table rows (doors.adjoint-test.directions), each with a seed of its own, and a direction counts
# only where the forecast's roughness lets it be read.
#: the stored fields whose gradient the descent direction is made from (theta_qv_gradient reads these six)
DESCENT_FIELDS = ("theta_m", "qv", "rtheta_p", "rho_theta", "exner", "pressure_p")


def direction_names(table: dict, row: dict | None, asked: str | None) -> list[str]:
    """The test directions an adjoint-test command runs, in the table's order: --directions (a comma list of the
    table's rows, or all), else the row's own list (rows.<row>.adjoint_test_directions), else the table's default.
    The table's first direction (the bump) always runs: its window gives the dot, FD, Taylor and zero checks and the
    gradient the descent direction is made from. An unknown name is refused before any card or download."""
    spec = table["doors"]["adjoint-test"].get("directions") or {}
    rows, order = spec.get("rows") or {}, list(spec.get("order") or [])
    if asked is None:
        names = list((row or {}).get("adjoint_test_directions") or spec.get("default") or order)
    elif asked.strip() == "all":
        names = list(order)
    else:
        names = [x.strip() for x in asked.split(",") if x.strip()]
    unknown = [n for n in names if n not in rows]
    if unknown or not names:
        raise AdDoorError(f"--directions {asked}: "
                          + (f"{', '.join(unknown)} is not a test direction of the table; " if unknown else "")
                          + f"the directions are {', '.join(order)} (or all). Nothing was downloaded")
    first = spec.get("first")
    return [n for n in order if n in names or n == first]


def smooth_draw(rng: Any, geometry: dict, nlev: int, waves: int, wavelengths_km: list) -> Any:
    """A smooth pattern on the mesh's cells and levels, in [-1, 1] before any envelope: a sum of ``waves`` plane
    waves through the sphere (a random unit vector, a wavelength between the two given, a phase and a weight each)
    times a half-wave count 0, 1 or 2 over the levels. Every random number is one uniform draw of ``rng`` (numpy's
    PCG64 stream of the direction's seed), taken in a fixed order, so the same seed gives the same pattern."""
    import numpy as np
    la, lo = np.radians(np.asarray(geometry["latCell"], float)), np.radians(np.asarray(geometry["lonCell"], float))
    xyz = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=1)
    z = (np.arange(nlev) + 0.5) / nlev
    lo_km, hi_km = (float(x) for x in wavelengths_km)
    out = np.zeros((nlev, xyz.shape[0]))
    for _ in range(int(waves)):
        v = 2.0 * rng.random(3) - 1.0
        v /= max(float(np.sqrt((v * v).sum())), 1e-12)
        kappa = 2.0 * math.pi * EARTH_RADIUS_KM / (lo_km + (hi_km - lo_km) * rng.random())
        phase, weight = 2.0 * math.pi * rng.random(), 0.5 + rng.random()
        half_waves, shift = int(3 * rng.random()), 2.0 * math.pi * rng.random()
        out += weight * np.cos(kappa * (xyz @ v) + phase)[None, :] * np.cos(math.pi * half_waves * z + shift)[:, None]
    peak = float(np.max(np.abs(out)))
    return out / peak if peak > 0 else out


def tq_direction(table: dict, fmap: FieldMap, state: PlanState, dth: Any, dqv: Any, name: str,
                 max_words: int | None = None, sizes: tuple | None = None) -> dict:
    """A change of potential temperature (``dth``, K) and water vapour (``dqv``, kg/kg) at fixed density and wind,
    carried through the linearised hex write-back to theta_m, rtheta_p, rho_theta, exner, pressure_p and qv on every
    copy: the transpose pair of ``theta_qv_gradient``, so <direction, stored gradient> = <dth, dJ/dtheta> + <dqv,
    dJ/dqv>. ``max_words``: at most that many (level, cell) words are moved, the largest by ``sizes`` (the change of
    each word over its own scale); a direction cut to its largest words is still a direction."""
    import numpy as np
    k = table["constants"]
    rvord, rgas, rcv, p0 = k["rv"] / k["rgas"], k["rgas"], k["rgas"] / (k["cp"] - k["rgas"]), k["p0"]
    get = {n: fmap.resolve(table, n) for n in ("theta_m", "qv", "rho_zz", "rho_p", "rtheta_p", "rho_theta", "exner",
                                               "pressure_p")}
    thm, qv, rz, rtp, rth, ex = (state.field(get[n]) for n in ("theta_m", "qv", "rho_zz", "rtheta_p", "rho_theta",
                                                               "exner"))
    dth, dqv = np.asarray(dth, np.float64), np.asarray(dqv, np.float64)
    rtbase = rth - rtp
    zz = ex ** (1.0 / rcv) * p0 / (rgas * rth)
    a = 1.0 + rvord * qv
    t_thm = dth * a + thm / a * rvord * dqv
    t_rtp = t_thm * rz
    t_ex = rcv * ex / rth * t_rtp
    t_pp = zz * rgas * (t_ex * rtp + ex * t_rtp + rtbase * t_ex)
    size = np.maximum(np.abs(dth), np.abs(dqv)) if sizes is None else np.maximum(np.abs(sizes[0]), np.abs(sizes[1]))
    peak = float(size.max()) if size.size else 0.0
    keep = size > 1e-4 * peak
    cut = False
    if max_words and int(keep.sum()) > int(max_words):
        keep &= size >= np.partition(size[keep], -int(max_words))[-int(max_words)]
        cut = True
    idx = np.argwhere(keep)
    terms: list = []
    for field, values in (("theta_m", t_thm), ("rtheta_p", t_rtp), ("rho_theta", t_rtp), ("exner", t_ex),
                          ("pressure_p", t_pp), ("qv", dqv)):
        vals = values[keep]
        moved = vals != 0.0
        if not moved.any():
            continue
        for (alloc, off), shape in zip(get[field]["copies"], get[field]["copy_shapes"]):
            words = off // 4 + idx[:, 0] * shape[-1] + idx[:, 1]
            terms.extend([alloc, w, v] for w, v in zip(words[moved].tolist(), vals[moved].tolist()))
    return {"name": name, "terms": terms, "words_per_field": int(keep.sum()),
            "cut_to_largest_words": int(max_words) if cut else None,
            "dth": np.where(keep, dth, 0.0), "dqv": np.where(keep, dqv, 0.0)}


def _bump_fields(table: dict, fmap: FieldMap, state: PlanState, doc: dict) -> tuple:
    """(dth, dqv) of a direction file's terms, read from its theta_m and qv words on their first copies (the bump:
    its file is the runner's, and its fields are read back from it rather than made a second time)."""
    import numpy as np
    k = table["constants"]
    rvord = k["rv"] / k["rgas"]
    out = {}
    for field in ("theta_m", "qv"):
        f = fmap.resolve(table, field)
        (alloc, off), shape = f["copies"][0], f["copy_shapes"][0]
        dense = np.zeros(f["shape"], np.float64)
        mine = [t for t in doc.get("terms") or [] if t[0] == alloc]   # this copy's words only: a 3 km bump is millions
        rows = np.asarray(mine, np.float64).reshape(-1, 3)
        words = rows[:, 1].astype(np.int64) - off // 4
        lev, col = words // shape[-1], words % shape[-1]
        ok = (lev >= 0) & (lev < f["shape"][0]) & (col < f["shape"][-1])
        dense[lev[ok], col[ok]] = rows[ok, 2]
        out[field] = dense
    qv = state.field(fmap.resolve(table, "qv"))
    thm = state.field(fmap.resolve(table, "theta_m"))
    a = 1.0 + rvord * qv
    return (out["theta_m"] - thm / a * rvord * out["qv"]) / a, out["qv"]


def descent_part(g_th: Any, g_qv: Any, spec: dict) -> tuple:
    """The part of the gradient a descent direction follows, and its words: the row's ``fields`` (theta, qv; default
    both) on the row's ``levels`` (1-based, default every level). Elsewhere the direction moves nothing.

    THE BREAKAGE THIS PREVENTS: on the 120 km example (3 steps, 36 minutes) the whole gradient's descent read 0.968
    to 1.056 of the real change with no three neighbouring step sizes agreeing, so the one direction made from the
    adjoint's own answer never counted and a correct install printed NOT PASSED (two 96 GB card runs, 15 Nov). A row
    can now name the part of the state its window can resolve."""
    import numpy as np
    fields = [str(f) for f in (spec.get("fields") or ("theta", "qv"))]
    unknown = [f for f in fields if f not in ("theta", "qv")]
    if unknown:
        raise AdDoorError(f"a descent direction follows theta or qv; the table names {', '.join(unknown)}")
    nlev = g_th.shape[0]
    mask = np.zeros(nlev)
    mask[parse_levels(spec.get("levels"), nlev)] = 1.0
    part_th = g_th * mask[:, None] if "theta" in fields else np.zeros_like(g_th)
    part_qv = g_qv * mask[:, None] if "qv" in fields else np.zeros_like(g_qv)
    whole = not spec.get("fields") and not spec.get("levels")
    where = "theta and qv" if whole else f"{' and '.join(fields)} on model levels {spec.get('levels') or f'1-{nlev}'}"
    return part_th, part_qv, where


def make_direction(ctx: "Context", state: PlanState, name: str, spec: dict, grads_win: Path | None) -> dict:
    """One test direction of the table as the runner's direction file's content (and its temperature and moisture
    fields, for the cosines). Kinds: ``draw`` (a smooth seeded pattern inside the bump's own envelope, on the named
    fields and levels) and ``descent`` (minus the gradient the first window dumped, each field over its own scale)."""
    import numpy as np
    a = ctx.args
    table, fmap, geometry = ctx.table, ctx.fmap, ctx.geometry
    bump = table["perturbations"]["theta"]["default"]
    at = resolve_at(a.at, geometry)
    radius = a.radius_km if a.radius_km is not None else bump["radius_km"]
    lat0, lon0 = (float(x) for x in at.split(","))
    f_qv = fmap.resolve(table, "qv")
    qv = state.field(f_qv)
    nlev, ncol = qv.shape
    sizes = spec.get("size") or {}
    s_th, rel_qv = float(sizes.get("theta_K", 1.0)), float(sizes.get("qv_relative", 0.1))
    s_qv = rel_qv * np.maximum(qv, 0.0)          # the moisture scale: a share of the local vapour, never below zero
    cap = int(spec["max_words"]) if spec.get("max_words") else None
    if spec["kind"] == "draw":
        levels = parse_levels(spec.get("levels"), nlev)
        mask = np.zeros(nlev)
        mask[levels] = 1.0
        env = np.exp(-(great_circle_km(lat0, lon0, geometry["latCell"], geometry["lonCell"]) / radius) ** 2)
        waves = [float(x) * radius for x in spec["wavelengths_in_radii"]]
        dth, dqv = np.zeros((nlev, ncol)), np.zeros((nlev, ncol))
        fields = list(spec["fields"])
        rng = np.random.Generator(np.random.PCG64(int(spec["seed"])))   # one stream a direction: its seed is its own
        for field in fields:   # each field its own pattern, drawn one after the other from that stream
            pattern = smooth_draw(rng, geometry, nlev, int(spec["waves"]), waves)
            pattern = pattern * mask[:, None] * env[None, :]
            peak = float(np.max(np.abs(pattern)))
            pattern = pattern / peak if peak > 0 else pattern
            if field == "theta":
                dth = s_th * pattern
            elif field == "qv":
                dqv = s_qv * pattern
            else:
                raise AdDoorError(f"test direction {name}: the table names field {field!r}; a draw moves theta or qv")
        where = (f"a smooth seeded draw (seed {spec['seed']}, {spec['waves']} waves) on "
                 f"{' and '.join(fields)}, model levels {spec.get('levels') or f'1-{nlev}'}, inside the bump's "
                 f"envelope (Gaussian R {radius:g} km at {lat0}N {lon0}E)")
        doc = tq_direction(table, fmap, state, dth, dqv, f"{name}: {where}; peak {s_th:g} K and {rel_qv:g} of the "
                           "local vapour; density and wind fixed; through the write-back to every copy",
                           cap, (dth / s_th, np.divide(dqv, s_qv, out=np.zeros_like(dqv), where=s_qv > 0)))
    elif spec["kind"] == "descent":
        if grads_win is None:
            raise AdDoorError(f"test direction {name}: the first window dumped no gradient to make it from")
        inputs = [fmap.resolve(table, n) for n in DESCENT_FIELDS]
        if getattr(ctx, "split", None):   # the ranks' copies summed back into the one-card allocations
            ctx.gather_rank_grads(grads_win, inputs)
        try:
            grads = {f["name"]: _sum_copies(grads_win, f) for f in inputs}
        except (OSError, ValueError) as error:   # the runner skips an allocation it holds no gradient for
            raise AdDoorError(f"test direction {name}: the first window dumped no gradient for one of "
                              f"{', '.join(DESCENT_FIELDS)} ({error})") from error
        g_th, g_qv = theta_qv_gradient(table, fmap, state, grads)
        g_th, g_qv, where = descent_part(g_th, g_qv, spec)   # the row's fields and levels (default: all of it)
        step = np.maximum(s_th * np.abs(g_th), s_qv * np.abs(g_qv))    # each word's move over its own scale
        peak = float(step.max()) if np.isfinite(step).all() else float("nan")
        if not peak > 0.0:
            raise AdDoorError(f"test direction {name}: the gradient the first window dumped is zero (or not finite) "
                              f"on {where}, so it has no descent direction")
        dth, dqv = -(s_th ** 2) * g_th / peak, -(s_qv ** 2) * g_qv / peak
        predicted = float((dth * g_th).sum() + (dqv * g_qv).sum())
        doc = tq_direction(table, fmap, state, dth, dqv,
                           f"{name}: minus the gradient of this objective to {where} (the first window's own "
                           f"answer), each over its scale ({s_th:g} K, {rel_qv:g} of the local vapour), the largest "
                           "word moved by one scale; density and wind fixed; through the write-back to every copy",
                           cap, (dth / s_th, np.divide(dqv, s_qv, out=np.zeros_like(dqv), where=s_qv > 0)))
        doc["predicted_change_all_words"] = predicted
    else:
        raise AdDoorError(f"test direction {name}: the table names kind {spec['kind']!r}; this door makes bump, "
                          "draw and descent")
    doc["scales"] = {"theta_K": s_th, "qv_relative": rel_qv}
    return refuse_zero_direction(doc)


def direction_cosine(a: tuple, b: tuple, s_th: float, s_qv: Any) -> float | None:
    """The cosine of two directions' temperature and moisture fields, each word over its own scale."""
    import numpy as np
    def scaled(d):
        return np.concatenate([(d[0] / s_th).ravel(), np.divide(d[1], s_qv, out=np.zeros_like(d[1]),
                                                                 where=s_qv > 0).ravel()])
    x, y = scaled(a), scaled(b)
    norm = float(np.sqrt((x * x).sum()) * np.sqrt((y * y).sum()))
    return float((x * y).sum() / norm) if norm > 0 else None


def steady_steps(rows: list[dict], within: float, run: int) -> tuple | None:
    """Marks the step sizes at which the real model's change of the measure can be read: ``counts`` and ``spread``
    on each row (rows sorted from the largest step), and returns the steadiest run, (its spread, its rows): the
    ``run`` neighbouring step sizes whose quotients agree best (of two that agree equally, the smaller steps), or
    None when no run is steady. Its rows are marked ``steadiest``. The forecast is not smooth: a branch taken in one run and not
    in the other moves the measure by an amount that does not shrink with the step, so below some step size the
    difference quotient dJ/h jumps from one step size to the next (the forecast's own roughness). A step size counts
    when it lies in a run of ``run`` neighbouring step sizes whose quotients agree within ``within`` of the largest;
    its spread is the smallest such disagreement, the uncertainty of its quotient. A run, not a pair: two noisy
    quotients agree by luck far more often than three."""
    best = None
    for r in rows:
        r["counts"], r["spread"], r["steadiest"] = False, None, False
    for i in range(len(rows) - run + 1):
        q = [r["dJ"] / r["h"] for r in rows[i:i + run]]
        top = max(abs(x) for x in q)
        if top <= 0.0:
            continue
        spread = (max(q) - min(q)) / top
        if spread <= within:
            for r in rows[i:i + run]:
                r["counts"] = True
                r["spread"] = spread if r["spread"] is None else min(r["spread"], spread)
            if best is None or spread <= best[0]:
                best = (spread, rows[i:i + run])
    for r in (best[1] if best else []):
        r["steadiest"] = True
    return best


def storm_measure_probe(table: dict, args: argparse.Namespace) -> float | None:
    """The roughness probe step of a storm measure (table storm_measures.probe_h) for adjoint-test's --metric, else
    None (every other measure is read as before, one-sided)."""
    block = table.get("storm_measures") or {}
    if getattr(args, "metric", None) in (block.get("metrics") or []) and block.get("probe_h"):
        return float(block["probe_h"])
    return None


def storm_taylor_h(ladder: list, probe: float, own: list | None = None) -> list[float]:
    """A storm measure's step sizes: the row's storm ladder (table storm_measures.taylor_h.<row>, else the row's
    adjoint-test ladder) at both signs, and the probe at both signs."""
    pos = [abs(float(h)) for h in (own or ladder) if float(h) != 0.0]
    return pos + [probe, -probe] + [-h for h in reversed(pos)]


def storm_measure_lines(table: dict, args: argparse.Namespace, door: str, metrics: list) -> list[str]:
    """The table's measured line for a storm measure (``storm_measures``: rain water, cloud water, vertical velocity)
    on the row this command names, printed before anything runs or is downloaded.

    THE BREAKAGE THIS PREVENTS: the release's PASSes are for temperature and moisture measures, and a measure of the
    storm itself ran with nothing on the screen saying so: on the 12 km example's storm band adjoint-test read rain
    FAIL (0.118 against 0.01) and updraft UNRESOLVED, on a 3 km storm cell updraft FAIL (1.008 to 1.023), while
    the sensitivity maps of the same measures read valid True (the transpose check), 16 Nov 2026."""
    block = table.get("storm_measures") or {}
    asked = sorted({m for m in metrics if m in (block.get("metrics") or [])})
    if not asked:
        return []
    mesh = getattr(args, "mesh", None)
    if mesh:   # a mesh named with --mesh has no reading of its own: say so rather than nothing
        return [f"woof hex {door}: storm measure ({', '.join(asked)}): no PASS of this release covers it, and its accuracy "
                f"has not been read on a mesh named with --mesh ({mesh}) (docs/differentiable/LIMITS.md, 'Storm measures')"]
    name = getattr(args, "row", None) or table.get("default_row")
    said = (block.get("rows") or {}).get(name)
    words = ", ".join(asked)
    if said:
        return [f"woof hex {door}: storm measure ({words}): no PASS of this release covers it; {said} "
                "(docs/differentiable/LIMITS.md, 'Storm measures')"]
    return [f"woof hex {door}: storm measure ({words}): no PASS of this release covers it, and its accuracy has "
            f"not been read on row {name} (docs/differentiable/LIMITS.md, 'Storm measures')"]


def say_storm_measures(table: dict, args: argparse.Namespace, door: str, metrics: list) -> None:
    for line in storm_measure_lines(table, args, door, metrics):
        print(line, flush=True)


def roughness_view(doc: dict, probe: float | None) -> tuple[dict, dict | None]:
    """A window receipt read against the forecast's own roughness: (the receipt with its Taylor rows centred, the
    roughness), or (the receipt unchanged, None) when ``probe`` is None or the window ran no probe step.

    The probe rows are the step sizes of magnitude at most ``probe`` (both signs). The gradient predicts a change of
    the measure of ``probe`` times its directional derivative there, far below one rounding of any field, so what
    they read is the forecast's own reaction to a nudge: its roughness, ``floor`` = the largest |dJ| of a probe row.
    Every other step size is read centred, (dJ(+h) - dJ(-h)) / 2 (which cancels the curvature and the part of the
    roughness both signs share), and carries ``noise`` = floor / (h |prediction per unit step|): how far the
    roughness alone can move its ratio. A step size run at one sign only is left out.

    THE BREAKAGE THIS PREVENTS: on the 12 km example's storm band a rain-water measure printed FAIL (the first
    window's finite-difference check 0.118 against 0.01) and the ratios 0.40 to 2.93 below a step of 0.03, read as a
    wrong gradient. A rerun at step 0 is byte-equal to the base, but a nudge of 1e-06 of the 1 K bump moved the mean
    rain by -7.6e-08 and +3.0e-08 kg/kg, 13,000 times the gradient's prediction for it, and 1e-04 by -5.5e-08: the
    rain field reacts to a one-rounding change of the start, and that floor equals the linear signal near a step of
    0.01. Centred, the steps 0.1 to 0.003 read 0.99 to 1.04 of the gradient; no step size can read it to 1 per cent
    (one RTX 5090, 0.6.0's wheel, 16 Nov 2026). The floor is a lower bound (a larger step flips more switches:
    over 5 steps the 12 km updraft bump moved 8 per cent at step 0.1 where it read 0.3 per cent), so the steady
    rule of direction_reading still applies to the rows it lets count."""
    if not probe:
        return doc, None
    td = doc.get("tangent_dir") or {}
    pred = td.get("x_MTw")
    pred = float(pred) if isinstance(pred, (int, float)) and math.isfinite(pred) and pred != 0.0 else None
    num = lambda v: float(v) if isinstance(v, (int, float)) and math.isfinite(v) else None
    rows = [r for r in (doc.get("taylor") or {}).get("rows", []) if num(r.get("h")) and num(r.get("dJ")) is not None]
    probes = [r for r in rows if abs(float(r["h"])) <= probe * (1 + 1e-9)]
    if not probes:
        return doc, None
    floor = max(abs(float(r["dJ"])) for r in probes)
    by_h = {float(r["h"]): r for r in rows if abs(float(r["h"])) > probe * (1 + 1e-9)}
    out = []
    for h in sorted({abs(x) for x in by_h}, reverse=True):
        plus, minus = by_h.get(h), by_h.get(-h)
        if plus is None or minus is None:
            continue
        dj = (float(plus["dJ"]) - float(minus["dJ"])) / 2.0
        ratio = None if pred is None else dj / (h * pred)
        row = {"h": h, "dJ": dj, "fd_quotient": dj / h, "ratio": ratio,
               "err": None if ratio is None else abs(ratio - 1.0), "centred": True,
               "noise": None if pred is None else floor / (h * abs(pred)),
               "one_sided_ratio": [num(plus.get("ratio")), num(minus.get("ratio"))]}
        if plus.get("state_fields"):
            row["state_fields"] = plus["state_fields"]
        out.append(row)
    view = dict(doc)
    view["taylor"] = {**(doc.get("taylor") or {}), "rows": out}
    return view, {"probe_h": probe, "floor": floor, "probe_dJ": [[float(r["h"]), float(r["dJ"])] for r in probes],
                  "predicted_per_unit_h": pred,
                  "floor_over_prediction_at_largest_h": (None if pred is None or not out
                                                         else floor / (out[0]["h"] * abs(pred)))}


def rough_counts(row: dict, bar: float) -> bool:
    """A centred step size counts only where the roughness alone moves its ratio by at most half the bar."""
    return row.get("noise") is None or row["noise"] <= bar / 2.0


def direction_reading(doc: dict, bar: float, within: float, run: int, probe: float | None = None) -> dict:
    """What one direction's window says of the gradient: per step size the real model's change of the measure over
    the gradient's prediction (ratio), and the direction's verdict:

    * a step size counts where the real change is steady (``steady_steps``);
    * ``inside``: at a step size that counts, |ratio - 1| plus that step's spread is within the bar;
    * ``outside``: at every step size of the steadiest run, |ratio - 1| less that run's spread is beyond the bar;
    * ``unresolved``: no step size counts (the measure changes less than the forecast's own roughness along this
      direction), or the steadiest ones read within their own spread of the bar on either side. Never a pass, and
      never a fail.

    THE BREAKAGE THE STEADIEST RUN PREVENTS: asked of every step size that counts, "outside" was undone by one
    step size at the measure's own noise floor. On a 12 km cut of the storm box the real runner read, along the
    near-ground moisture direction, 0.850, 0.881 and 0.882 at h = 0.3, 0.1 and 0.03 (agreeing within 3.6 per cent)
    and 0.966 at h = 0.01, where the measure's change was 2.6e-05 K, the last digits of a float32 mean near 300 K;
    that step size still agreed with its neighbours within the steady bound (8.8 per cent), so a gradient 12 per
    cent off was printed UNRESOLVED, which fails no verdict.

    With ``probe`` (a storm measure) the rows are read centred against the forecast's roughness (roughness_view): a
    step size whose roughness bound is over half the bar does not count, and its spread is at least that bound."""
    doc, rough = roughness_view(doc, probe)
    td = doc.get("tangent_dir") or {}
    num = lambda v: float(v) if isinstance(v, (int, float)) and math.isfinite(v) else None
    predicted = num(td.get("x_MTw"))
    rows = []
    for r in sorted((r for r in (doc.get("taylor") or {}).get("rows", []) if num(r.get("h"))), key=lambda r: -r["h"]):
        h, q = float(r["h"]), num(r.get("fd_quotient"))
        dj = num(r.get("dJ")) if num(r.get("dJ")) is not None else (None if q is None else q * h)
        if dj is None:
            continue
        ratio = num(r.get("ratio"))
        if ratio is None and predicted:
            ratio = dj / (h * predicted)
        rows.append({"h": h, "dJ": dj, "ratio": ratio, "err": None if ratio is None else abs(ratio - 1.0),
                     **({"noise": r.get("noise"), "one_sided_ratio": r.get("one_sided_ratio")} if rough else {})})
    best = steady_steps(rows, within, run)
    if rough:   # the roughness bound: a step size it can move by more than half the bar does not count
        for r in rows:
            if r["counts"] and not rough_counts(r, bar):
                r["counts"] = False
            if r["counts"] and r.get("noise") is not None:
                r["spread"] = max(r["spread"] or 0.0, r["noise"])
        if best is not None and not any(r["counts"] for r in best[1]):
            best = None
    steadiest = [] if best is None else best[1]
    counted = [r for r in rows if r["counts"]]
    judged = [r for r in counted if r["err"] is not None]
    inside = [r for r in judged if r["err"] + r["spread"] <= bar]
    value, why = None, None
    if counted and not judged:
        status = "outside" if predicted == 0.0 else "unresolved"
        why = ("the gradient predicts no change of the measure along this direction and the forecast changes it "
               "steadily" if predicted == 0.0 else "the window gave no <x, M^T w> for this direction")
    elif inside:
        status, value = "inside", min(r["err"] for r in inside)
    elif steadiest and all(r["err"] is not None and r["err"] - best[0] > bar for r in steadiest):
        status, value = "outside", min(r["err"] for r in steadiest)
    else:
        status = "unresolved"
        value = min((r["err"] for r in (steadiest or judged) if r["err"] is not None), default=None)
        why = (f"fewer than {run} step sizes: whether the real change is steady cannot be read" if len(rows) < run
               and not rough else
               (f"the forecast's own roughness (a nudge of {rough['probe_h']:g} moves the measure by "
                f"{rough['floor']:.2e}) moves the ratio by more than half the bar at every step size "
                f"({rows[0]['noise']:.2g} at the largest, centred on both signs)")
               if rough and rows and not counted and rows[0].get("noise") is not None else
               f"the real change is not steady over any {run} neighbouring step sizes (within {within:g}): "
               "the measure changes less than the forecast's own roughness along this direction"
               if not counted else
               "the steadiest step sizes read within their own spread of the bar on either side")
    return {"status": status, "value": value, "bar": bar, "why": why, "rows": rows,
            "predicted_per_unit_h": predicted, "w_Mx": num(td.get("w_Mx")), "dot_rel": num(td.get("rel")),
            "steady_within": within, "steady_steps": run, **({"roughness": rough} if rough else {})}


def directions_verdict(directions: list[dict], checks_passed: bool, rule: dict) -> dict:
    """The command's verdict over its directions: ``fail`` when a check of the first window (or the whole-state dot)
    fails, a direction could not run, or a resolved direction is outside the bar; else ``unresolved`` with fewer
    than the rule's count of resolved directions (a direction that repeats an earlier one, cosine at or above the
    rule's, does not count); else ``pass``."""
    counted = [d for d in directions if d.get("independent", True)]
    resolved = [d for d in counted if d["status"] in ("inside", "outside")]
    outside = [d["name"] for d in directions if d["status"] == "outside"]
    not_run = [d["name"] for d in directions if d["status"] == "not run"]
    need = int(rule.get("min_resolved", 2))
    if not checks_passed or outside or not_run:
        verdict = "fail"
    elif len(resolved) < need:
        verdict = "unresolved"
    else:
        verdict = "pass"
    return {"verdict": verdict, "resolved": [d["name"] for d in resolved], "outside": outside, "not_run": not_run,
            "unresolved": [d["name"] for d in directions if d["status"] == "unresolved"],
            "not_independent": [d["name"] for d in directions if not d.get("independent", True)],
            "min_resolved": need,
            "rule": "PASS: the first window's checks and the whole-state dot pass, every direction ran, every direction "
                    "the forecast's roughness lets its window resolve is inside the bar, and at least "
                    f"{need} independent directions are resolved; a direction that is not resolved never counts"}


def known_limits(row: dict, asked: str | None) -> dict:
    """The measured limits the door table names for the linear model this command tests: the boundary-layer route
    entry's ``known_limits`` (``{direction name or "window": {"text": ..., "source": ...}}``); on a row with no
    boundary-layer routes, the row's own ``known_limits``; else {}.

    THE BREAKAGE THIS PREVENTS: a new user's first command printed a bare UNRESOLVED (or, on a finer step ladder, a
    bare FAIL) along a direction whose miss was already measured and explained (near-ground moisture on the 12 km
    cut: the plume's own speed and condensation, which the default route holds), and nothing on the screen told a
    documented limit of the linear model from a broken install. A known limit never changes the verdict: it is not a
    pass, it never counts as a resolved direction, and a direction read outside the bar still fails the command."""
    if not row_boundary_layer(row):
        # a row without boundary-layer routes (the 120 km example) names its limits on the row itself
        return {k: (v if isinstance(v, dict) else {"text": str(v)}) for k, v in (row.get("known_limits") or {}).items()}
    _, route, _ = boundary_layer_route(row, asked)
    return {k: (v if isinstance(v, dict) else {"text": str(v)}) for k, v in (route.get("known_limits") or {}).items()}


def print_known_limits(directions: list[dict], limits: dict) -> None:
    """One line per known limit, each with what this run read along that direction (if it ran one)."""
    if not limits:
        return
    status = {d["name"]: d["status"] for d in directions}
    print("  known limits of this linear model (measured before this release, not by this run; the release's "
          "docs/differentiable/LIMITS.md):")
    for key, item in limits.items():
        here = status.get(key)
        tag = "" if here is None else f"[this run: {here.upper()}] "
        print(f"    {key:<21} {tag}{item['text']}")


def failed_checks(checks: dict) -> list[str]:
    """The checks of an adjoint-test verdict that failed, each named with its own reading and bar, the whole-state
    dot last. THE BREAKAGE THIS PREVENTS: a seeded run whose only failure was the whole-state dot ended "FAIL: a check
    of the first window or the whole-state dot failed; a window was not a valid derivative", which did not say which
    check failed, and blamed the windows, which were valid (a 0.6.0 build, seed 3, 2026-11-15)."""
    names = {"dot": "the first window's dot", "fd": "the first window's finite-difference check",
             "taylor": "the first window's Taylor check", "zero": "the zero-direction check"}
    out = []
    for key, name in names.items():
        c = checks.get(key)
        if c and not c.get("pass") and not c.get("unresolved"):
            v, bar = c.get("value"), c.get("bar")
            if key == "zero":
                read = f"{v} nonzero words"
            else:
                read = ("n/a" if v is None else f"{v:.3e}") + ("" if bar is None else f" against {bar:.0e}")
            out.append(f"{name} ({read})")
    s = checks.get("self_dot")
    if s and not s.get("pass"):
        r = s.get("roundings")
        r = _dot_bar.roundings_of_rel(s.get("value")) if r is None else r
        k = s.get("bar_roundings") or s["bar"] / _dot_bar.EPS32
        v = s.get("value")
        out.append(f"the whole-state dot ({'n/a' if v is None else format(v, '.3e')}"
                   f"{'' if r is None else f', {r:+.2f} roundings'}, against the bar of {k:g} roundings, "
                   f"{s['bar']:.2e}; test direction {s.get('seed', 1)})")
    return out


def limits_word(over: dict, limits: dict, passed: bool, checks_passed: bool = True,
                dot_failed: bool | None = None) -> str:
    """The clause the verdict line adds for directions with a known limit: on a PASS, the unresolved ones it does
    not cover; on NOT PASSED (UNRESOLVED), the unresolved ones; on a FAIL, only what failed: the outside directions
    with a known limit, and the window's own limit (the whole-state consistency on seeded directions) when the
    whole-state dot failed (``dot_failed``; None, as callers before it: when any check failed). Empty when none applies.
    THE BREAKAGE THIS PREVENTS: a seeded run that failed on its whole-state dot ended "FAIL: ... near-ground-moisture:
    a known limit", naming a direction that read UNRESOLVED and did not cause the FAIL (a 0.6.0 build, seed 3,
    2026-11-15); and a failed finite-difference check is not the window limit."""
    if passed:
        left = [n for n in over.get("unresolved", []) if n in limits]
        return f"; not covered by this PASS: {', '.join(left)} (a known limit, named above)" if left else ""
    if over.get("verdict") == "unresolved" and checks_passed:
        known = [n for n in over.get("unresolved", []) if n in limits]
    else:
        known = [n for n in over.get("outside", []) if n in limits]
        window_failed = (not checks_passed) if dot_failed is None else dot_failed
        if window_failed and "window" in limits:
            known.append("window")
    return f"; {', '.join(known)}: a known limit of this linear model, named above" if known else ""


def print_directions(directions: list[dict], rule: dict) -> None:
    bar = directions[0]["bar"] if directions else rule.get("bar")
    print(f"  directions: the real model's change over the gradient's prediction, per step size h (bar: within "
          f"{bar:g} of 1; a step size counts where the real change per unit step is steady over "
          f"{int(rule['steady_steps'])} neighbouring step sizes, within {float(rule['steady_within']):g}; "
          "* = does not count):")
    for d in directions:
        seed = "no draw" if d.get("seed") is None else f"seed {d['seed']}"
        if d["status"] == "not run":
            print(f"    {d['name']:<21} {seed:<10} NOT RUN: {d.get('why')}")
            continue
        cells = "  ".join(f"h {r['h']:g}: " + ("n/a" if r["ratio"] is None else f"{r['ratio']:.4f}")
                          + ("" if r["counts"] else "*") for r in d["rows"])
        word = {"inside": "INSIDE", "outside": "OUTSIDE", "unresolved": "UNRESOLVED"}[d["status"]]
        tail = "" if d["value"] is None else f" ({d['value']:.2e} from 1)"
        if d["status"] == "unresolved":
            tail += f": {d['why']}; it does not count"
        if not d.get("independent", True):
            tail += f"; NOT INDEPENDENT of {d['repeats']} (cosine {d['cosine'][d['repeats']]:.3f}): it does not count"
        print(f"    {d['name']:<21} {seed:<10} {cells}   {word}{tail}")


# ------------------------------------------------------------------- adjoint-test --plant-error: break it on purpose
#: Exit status of an adjoint-test run with an error planted on purpose (--plant-error): 4 when the transpose check
#: caught it (FAIL, as it must), 5 when it did not. Never 0 or 1, so a planted run is never read as a real verdict.
PLANTED_CAUGHT_EXIT = 4
PLANTED_MISSED_EXIT = 5
#: The runner's test-only planting words (window.rs AD_PLANT, AD_PLANT_RECOMPUTE, AD_PLANT_LEFT_OUT; step_hold.rs
#: AD_PLANT_HOLD). A door sets AD_PLANT itself, and only from adjoint-test --plant-error.
PLANT_ENV_PREFIX = "AD_PLANT"


def refuse_planted_env(door: str, before_download: bool = True) -> None:
    """A runner planting word in the environment is refused. THE BREAKAGE THIS PREVENTS: the door hands its own
    environment to every runner it starts and judges by its own checks, not the runner's planted_error, so an
    AD_PLANT* word left in a shell made a sensitivity map, a tangent or an adjoint-test verdict of a derivative broken
    on purpose, with nothing on screen saying so."""
    words = sorted(k for k in os.environ if k.startswith(PLANT_ENV_PREFIX))
    if words:
        raise AdDoorError(f"{', '.join(words)} is set in the environment: a test-only switch of the runner that breaks "
                          f"the derivative on purpose, which this {door} run would take. Unset it. To watch the check "
                          "catch a planted error, run adjoint-test --plant-error KERNEL:SCALE"
                          + (". Nothing was downloaded" if before_download else ""))


def _plant_mesh_kind(row: dict) -> str:
    kind = row.get("mesh_kind") or (row.get("generated") or {}).get("mesh_kind")
    return "limited-area" if kind == "limited-area" else "global"


def plant_request(args: argparse.Namespace, table: dict, door: str, row: dict | None = None) -> dict | None:
    """--plant-error KERNEL:SCALE against the table's ``doors.adjoint-test.plant_error``: None when off (the default),
    else what is planted and the runner's word for it (AD_PLANT). Every door refuses the runner's planting words in its
    environment; a door other than adjoint-test refuses the option. ``row``: the row the run uses when it is known (a
    table row, before any download; the row made for --mesh, after Context). Refusals come before any card or download.
    THE BREAKAGE THIS PREVENTS: a check that only ever prints PASS shows nobody that it can fail; the runner's planting
    switch was reachable only through an environment word no command named or reported."""
    asked = getattr(args, "plant_error", None)
    if asked is not None and door != "adjoint-test":
        raise AdDoorError(f"--plant-error is refused by {door}: an error planted on purpose belongs only in "
                          "adjoint-test, whose verdict then says PLANTED; a gradient, a tangent or a map is never made "
                          "from a derivative broken on purpose. Nothing was downloaded")
    refuse_planted_env(door)
    if asked is None:
        return None
    spec = table["doors"]["adjoint-test"]["plant_error"]
    kernels = spec["kernels"]
    lo, hi = spec["scale_range"]
    name, sep, word = str(asked).rpartition(":")
    try:
        scale = float(word)
    except ValueError:
        scale = float("nan")
    if not sep or name not in kernels or not math.isfinite(scale) or not lo <= scale <= hi or scale == 1.0:
        raise AdDoorError(f"--plant-error {asked}: give KERNEL:SCALE, KERNEL one of {', '.join(kernels)} and SCALE a "
                          f"number from {lo:g} to {hi:g} other than 1 (1.01 plants a 1 per cent error). Nothing was "
                          "downloaded")
    if getattr(args, "no_self_dot", False):
        raise AdDoorError("--plant-error with --no-self-dot: the whole-state dot is the transpose check a planted error "
                          "is there to fail; give one. Nothing was downloaded")
    if row is None:
        row_name = None if getattr(args, "mesh", None) else (getattr(args, "row", None) or table["default_row"])
        row = table["rows"].get(row_name) if row_name else None
    plant = {"kernel": name, "scale": scale, "error_per_cent": round(abs(scale - 1.0) * 100.0, 6),
             "what": kernels[name]["what"], "switch": "AD_PLANT", "side": spec["side"], "target": None,
             "runner_word": None, "mesh_kind": None}
    if row is not None:
        kind = _plant_mesh_kind(row)
        target = (kernels[name].get("plant") or {}).get(kind)
        if not target:
            offered = [k for k, v in kernels.items() if (v.get("plant") or {}).get(kind)]
            raise AdDoorError(f"--plant-error {asked}: {name} is not offered on a {kind} mesh; there: "
                              f"{', '.join(offered) or 'none'}. Nothing was downloaded")
        plant.update(target=target, runner_word=f"{target}:{scale!r}", mesh_kind=kind)
    return plant


def plant_line(plant: dict) -> str:
    """The one plain line a planted run prints before anything runs."""
    return (f"woof hex adjoint-test: AN ERROR IS PLANTED ON PURPOSE (--plant-error {plant['kernel']}:{plant['scale']:g}): "
            f"in {plant['what']}, the backward (adjoint) side only, each change is scaled by {plant['scale']:g}, a "
            f"{plant['error_per_cent']:g} per cent error. The forward model and the tangent are untouched, so the "
            "transpose check is expected to FAIL. This run says nothing about the release's derivative, and its "
            "receipt is never valid")


def planted_reading(out: Path) -> dict:
    """What the runner says it planted, window by window (window.json planted_error: launches and words changed)."""
    per = {}
    for path in sorted(out.glob("*/window.json")):
        try:
            pe = json.loads(path.read_text(encoding="utf-8")).get("planted_error")
        except (OSError, ValueError):
            pe = None
        per[path.parent.name] = None if not isinstance(pe, dict) else {
            "kernel": pe.get("kernel"), "argument": pe.get("argument"), "factor": pe.get("factor"),
            "launches": int(pe.get("launches") or 0), "words_changed": int(pe.get("words_changed") or 0)}
    return per


def planted_verdict(plant: dict, checks: dict, reading: dict) -> tuple[str, int, bool]:
    """The verdict line of a planted run, its exit status and whether the transpose check caught the error.
    Caught: the whole-state dot (or the bump's dot) outside its bar while the planted code ran in that window."""
    sd, dot = checks.get("self_dot") or {}, checks.get("dot") or {}
    hits = sum((r or {}).get("launches", 0) for r in reading.values())
    words = sum((r or {}).get("words_changed", 0) for r in reading.values())
    head = (f"PLANTED: an error was planted on purpose ({plant['kernel']}, {plant['target']}, x{plant['scale']:g}, "
            f"{plant['error_per_cent']:g} per cent; {hits} backward launches, {words:,} words changed)")
    if not (reading.get("selfdot") or {}).get("launches") or (dot and not (reading.get("window") or {}).get("launches")):
        return (head + f": NOTHING WAS PLANTED in a checked window: the runner did not run {plant['target']}'s backward "
                "code there, so this run tests nothing. Exit status 5"), PLANTED_MISSED_EXIT, False
    caught = []
    if sd and not sd.get("pass"):
        r = sd.get("roundings")
        r = _dot_bar.roundings_of_rel(sd.get("value")) if r is None else r
        k = sd.get("bar_roundings") or sd["bar"] / _dot_bar.EPS32
        caught.append(f"whole-state dot {sd.get('value'):.3e} = {abs(r):,.1f} roundings against the bar of {k:g}")
    if dot and not dot.get("pass"):
        caught.append(f"the bump's dot {dot.get('value'):.3e} against the bar of {dot.get('bar'):.0e}")
    if caught:
        return (head + ": the transpose check CAUGHT it: " + "; ".join(caught) + ", FAIL as it must. Exit status 4"), \
            PLANTED_CAUGHT_EXIT, True
    r = sd.get("roundings")
    r = _dot_bar.roundings_of_rel(sd.get("value")) if r is None else r
    return (head + f": NOT CAUGHT: the whole-state dot read {sd.get('value'):.3e} = "
            f"{'n/a' if r is None else format(abs(r), '.2f')} roundings, inside the bar: an error this small in this "
            "code, on this window, hides under single-precision rounding. Exit status 5"), PLANTED_MISSED_EXIT, False


def run_adjoint_test(args: argparse.Namespace) -> int:
    plant = plant_request(args, load_table(), "adjoint-test")   # refused before any card or download
    if plant:
        print(plant_line(plant), flush=True)
    route = gradient_route(args, "adjoint-test")   # a coarse mesh is refused by name here
    asked = getattr(args, "directions", None)
    direction_names(load_table(), None, asked)       # an unknown direction: refused before any card or download
    refuse_speed_words(args, load_table())           # a route the row does not offer: refused before any download
    say_storm_measures(load_table(), args, "adjoint-test", [getattr(args, "metric", None)])
    ctx = Context(args, "adjoint-test")
    if plant:   # every window of this run (bump, directions, whole-state dot) runs the planted backward code
        ctx.plant = plant = plant_request(args, ctx.table, "adjoint-test", ctx.row)
        ctx.plant_env = {"AD_PLANT": plant["runner_word"]}
    ctx.gradient_route = route
    import numpy as np
    t0 = time.monotonic()
    spec_at = ctx.table["doors"]["adjoint-test"]
    dspec = spec_at.get("directions") or {}
    rule = {"steady_within": 0.1, "steady_steps": 3, **dict(dspec.get("rule") or {})}
    bar = float(rule.get("bar") or ctx.table["thresholds"]["taylor_best_err"])
    storm = storm_measure_probe(ctx.table, args)   # a storm measure: read against the forecast's roughness
    reading = lambda window_doc: direction_reading(window_doc, bar, float(rule["steady_within"]),
                                                   int(rule["steady_steps"]), probe=storm)
    names = direction_names(ctx.table, ctx.row, asked)
    state = PlanState(ctx.plan)
    direction = _perturbation(ctx, state)
    objective = metric_objective(ctx.table, ctx.fmap, ctx.geometry, args.metric, args.levels,
                                 *adjoint_test_region(args, resolve_at(args.at, ctx.geometry)), ctx.row)
    metric_field = ctx.fmap.resolve(ctx.table, ctx.table["metrics"][args.metric]["field"], ctx.row, window_end=True)
    obj_path = ctx.out / "objective.json"
    _write_json(obj_path, objective)
    own = ((ctx.table.get("storm_measures") or {}).get("taylor_h") or {}).get(ctx.row_name)
    hs = args.taylor or ",".join(f"{h:g}" for h in (storm_taylor_h(adjoint_test_taylor_h(spec_at, ctx.row), storm, own)
                                                     if storm else adjoint_test_taylor_h(spec_at, ctx.row)))
    ctx.stage("inputs (direction, objective)", t0, "1 thread (setup, small)")
    # the window end's tangent of the metric field: the runner forms <w, M x> beside <x, M^T w> only with a dump
    dump = ctx.rank_list("dump-state", [(metric_field["alloc"], metric_field["offset"],
                                         4 * math.prod(metric_field["read_shape"]))], "copy")
    # on a split window each rank gets its own cut: the direction on every copy, the objective on the owner only
    first_words = ["--taylor-dir", ctx.rank_json(Path(direction["path"]), owner_only=False),
                   "--objective", ctx.rank_json(obj_path, owner_only=True), "--taylor", hs, "--dump-state", dump]
    others = [n for n in names if (dspec.get("rows") or {}).get(n, {}).get("kind") != "bump"]
    wants_gradient = any(dspec["rows"][n]["kind"] == "descent" for n in others)
    if wants_gradient:   # the gradient the descent direction is made from: this window's own, dumped (reads only)
        gfields = [ctx.fmap.resolve(ctx.table, n) for n in DESCENT_FIELDS]
        first_words += ["--dump-grad",
                        ctx.rank_list("dump-grad", [(a, o, 0) for f in gfields for a, o in f["copies"]], "alloc")
                        if getattr(ctx, "split", None)
                        else ",".join(str(a) for a in sorted({a for f in gfields for a, _ in f["copies"]}))]
    doc = ctx.window(first_words)
    verdict = adjoint_verdict(doc, ctx.table["thresholds"], probe=storm)
    # one more window for each further direction of the table: the same objective and step sizes, its own direction
    first_name = dspec.get("first") or "bump"
    scales = (dspec.get("rows") or {}).get(first_name, {}).get("size") or {}
    s_th = float(scales.get("theta_K", 1.0))
    s_qv = float(scales.get("qv_relative", 0.1)) * np.maximum(state.field(ctx.fmap.resolve(ctx.table, "qv")), 0.0)
    fields = {first_name: _bump_fields(ctx.table, ctx.fmap, state, direction["doc"])}
    directions = [{"name": first_name, "kind": "bump", "seed": None,
                   "what": (dspec.get("rows") or {}).get(first_name, {}).get("what"),
                   "direction": direction["doc"]["name"], "file": str(direction["path"]),
                   "sha256": _sha256_file(Path(direction["path"])), "terms": len(direction["doc"]["terms"]),
                   "window": "window", **reading(doc)}]
    for name in others:
        spec, t0 = dspec["rows"][name], time.monotonic()
        entry = {"name": name, "kind": spec["kind"], "seed": spec.get("seed"), "what": spec.get("what"),
                 "window": f"window-{name}"}
        try:
            made = make_direction(ctx, state, name, spec, ctx.out / "window")
            fields[name] = (made.pop("dth"), made.pop("dqv"))
            path = ctx.out / f"direction-{name}.json"
            _write_json_compact(path, made)
            entry.update(direction=made["name"], file=str(path), sha256=_sha256_file(path), terms=len(made["terms"]),
                         cut_to_largest_words=made.get("cut_to_largest_words"))
            ctx.stage(f"inputs (direction {name})", t0, "1 thread (host)")
            words = ["--taylor-dir", ctx.rank_json(path, owner_only=False),
                     "--objective", ctx.rank_json(obj_path, owner_only=True), "--taylor", hs, "--dump-state", dump]
            del made
            entry.update(reading(ctx.window(words, sub=f"window-{name}", door_args_key="direction_window_args")))
        except CardMemoryError:
            raise
        except AdDoorError as error:   # said, kept in the receipt, and the verdict cannot pass
            fields.pop(name, None)
            entry.update(status="not run", why=str(error), value=None, bar=bar, rows=[])
        directions.append(entry)
    for f in (ctx.out / "window").glob("grad-*.f32"):   # the dumped gradient was the descent direction's only
        f.unlink()
    for f in (ctx.out / "window").glob("r*/grad-*.f32"):
        f.unlink()
    limit = float(rule.get("independent_cosine", 0.99))
    for i, d in enumerate(directions):   # a direction that repeats an earlier one is one direction, not two
        d["cosine"] = {}
        for e in directions[:i]:
            if d["name"] in fields and e["name"] in fields:
                d["cosine"][e["name"]] = direction_cosine(fields[d["name"]], fields[e["name"]], s_th, s_qv)
        repeats = [n for n, c in d["cosine"].items() if c is not None and abs(c) >= limit]
        d["independent"], d["repeats"] = not repeats, (repeats[0] if repeats else None)
    del fields
    # a second window on the runner's own seeded direction (adjoint-test always runs its own: it is the check others bind)
    strict = strict_check(ctx, no_self_dot=args.no_self_dot)
    if strict is not None and strict.get("how") != "not run":
        verdict["checks"]["self_dot"] = strict
        verdict["passed"] = verdict["passed"] and strict["pass"]
    elif strict is not None:
        verdict["self_dot_skipped"] = "--no-self-dot: the whole-state dot was not run; this verdict does not cover it"
    checks_passed = bool(verdict["passed"])
    over = directions_verdict(directions, checks_passed, rule)
    verdict["passed"] = over["verdict"] == "pass"
    limits = known_limits(ctx.row, getattr(args, "boundary_layer", None))
    result = {"perturbation": direction["doc"]["name"], "objective": objective["name"], **verdict,
              "checks_passed": checks_passed, "directions": directions, "directions_verdict": over,
              "directions_rule": {**rule, "bar": bar, "what": dspec.get("rule_what")},
              "window_receipt": str(ctx.out / "window" / "window.json"), "fct_frozen": doc.get("fct_frozen"),
              "refused_launch_kinds": len(doc.get("refused") or {}),
              "known_limits": limits}
    planted = None
    if plant:   # what the runner says it planted, and the planted run's verdict (the receipt carries both)
        reading = planted_reading(ctx.out)
        line, code, caught = planted_verdict(plant, verdict["checks"], reading)
        ctx.plant = planted = {**plant, "reading": reading, "caught": caught, "exit_status": code, "verdict": line}
    rec = ctx.receipt(result, verdict["passed"], strict)
    verdict["passed"] = rec["passed"]
    c = verdict["checks"]
    fmt = lambda v: "n/a" if v is None else f"{v:.3e}"
    print(f"adjoint-test: {ctx.row_name}, {ctx.steps} steps ({window_length_words(ctx.steps * ctx.row['dt_s'])})")
    bl = rec.get("boundary_layer") or {}
    if bl:   # which linear model this verdict is of
        print(f"  linear model: the boundary layer's {bl['route']} route"
              + (" (this row's default: the one a plain sensitivity runs)" if bl.get("is_default") else
                 f" (--boundary-layer {bl['route']}; a plain sensitivity runs {bl.get('default')})"))
    print(f"  dot     {fmt(c['dot']['value'])}   bar {c['dot']['bar']:.0e}   {'PASS' if c['dot']['pass'] else 'FAIL'}"
          "   (this objective's direction; the bar is the float32 window dot's: the program's bar, in "
          "single-precision roundings of the dot, is the whole-state dot's below)")
    word3 = lambda k: "PASS" if c[k]["pass"] else ("UNRESOLVED" if c[k].get("unresolved") else "FAIL")
    print(f"  fd      {fmt(c['fd']['value'])}   bar {c['fd']['bar']:.0e}   {word3('fd')}")
    print(f"  taylor  {fmt(c['taylor']['value'])}   bar {c['taylor']['bar']:.0e}   "
          f"{word3('taylor')}{'' if c['taylor']['resolved_interval'] else ' (no step size confirmed by a neighbour)'}"
          f"{'   set aside (no neighbour within 10x): h = ' + ', '.join(f'{h:g}' for h in c['taylor']['set_aside_h']) if c['taylor']['set_aside_h'] else ''}")
    print(f"  zero    {c['zero']['value']} nonzero words   {'PASS' if c['zero']['pass'] else 'FAIL'}")
    if rec.get("split"):
        s = rec["split"]
        print(f"  split   {s['ranks']} ranks, one on each of cards {','.join(s['cards'])}; exchange {s['exchange']} "
              f"(the ranks' runners report: {', '.join(s['exchange_seen_by_ranks']) or 'nothing'})")
    if verdict.get("self_dot_skipped"):
        print("  whole-state dot  not run (--no-self-dot): the verdict below does not cover it")
    rough = verdict.get("roughness")
    if rough:   # a storm measure: the forecast's own roughness, read by the probe steps, and the centred rows
        print(f"  roughness: a nudge of +-{rough['probe_h']:g} x the perturbation moved the measure by up to "
              f"{rough['floor']:.3e} (the gradient predicts {abs(rough['predicted_per_unit_h'] or 0) * rough['probe_h']:.1e}); "
              "each step size below is read centred (both signs), and counts only where that roughness moves its "
              "ratio by at most half the bar (noise column)")
    print("  real model against the tangent, per bump size (h x the perturbation)"
          + (", centred on both signs:" if rough else ":"))
    print("       h     metric err    field |FD - TL| / |TL|" + ("    noise" if rough else ""))
    for r in verdict["taylor_rows"]:
        rel = r.get("field_fd_minus_tl_rel_l2")
        err = "n/a" if r.get("err") is None else format(r["err"], ">9.3e")
        print(f"    {r['h']:>5g}   {err:>9}    {'n/a' if rel is None else format(rel, '.3g')}"
              + (f"    {r['noise']:.3g}" if rough and r.get("noise") is not None else ""))
    print_directions(directions, {**rule, "bar": bar})
    print_known_limits(directions, limits)
    if "self_dot" in c:
        print_strict(c["self_dot"])
    if verdict["passed"]:
        word = f"PASS: {len(over['resolved'])} directions resolved, each inside the bar"
    elif over["verdict"] == "unresolved" and checks_passed:
        word = (f"NOT PASSED (UNRESOLVED): {len(over['resolved'])} of {len(directions)} directions resolved "
                f"({', '.join(over['resolved']) or 'none'}), and a PASS needs {over['min_resolved']}: one direction "
                "is not a gradient test")
    else:
        failed = failed_checks(c)
        why = ([f"outside the bar along {', '.join(over['outside'])}"] if over["outside"] else []) \
            + ([f"not run: {', '.join(over['not_run'])}"] if over["not_run"] else []) \
            + ([] if checks_passed else
               [f"failed: {'; '.join(failed)}" if failed else "a check of the first window or the whole-state dot "
                                                                 "failed"]) \
            + (["a window was not a valid derivative"] if any(w.get("valid") is False for w in ctx.windows) else [])
        word = "FAIL: " + "; ".join(why or ["see the lines above"])
    word += limits_word(over, limits, bool(verdict["passed"]), checks_passed,
                        dot_failed=None if checks_passed or not failed else failed[-1].startswith("the whole-state"))
    if planted:   # the verdict line of a planted run says PLANTED, never PASS or FAIL, and its exit status is its own
        print(f"{planted['verdict']}  (receipt {ctx.out / 'receipt.json'})")
        print(used_line(rec))
        return planted["exit_status"]
    print(f"{word}  (receipt {ctx.out / 'receipt.json'})")
    print(used_line(rec))
    return 0 if verdict["passed"] else 1


def _sum_copies(win: Path, f: dict):
    """dJ/d(field at boundary 0): the gradients of every byte-equal copy the write-back fills, summed in order."""
    import numpy as np
    total = np.zeros(f["shape"], np.float64)
    for (alloc, off), shape in zip(f["copies"], f["copy_shapes"]):
        g = np.fromfile(win / f"grad-{alloc}.f32", "<f4", count=math.prod(shape), offset=off).reshape(shape)
        total += on_mesh(g, f["shape"])
    return total


def theta_qv_gradient(table: dict, fmap: FieldMap, state: PlanState, grads: dict):
    """dJ/dtheta and dJ/dqv at fixed density and wind: the thermodynamic transpose of the hex write-back (T0^T).

    theta and qv enter the stored state only through theta_m (and qv's own scalar slot), and theta_m only through
    rtheta_p, rho_theta, exner and pressure_p, so this part of T0^T is exact without the wind and terrain terms."""
    k = table["constants"]
    rvord, rgas, rcv, p0 = k["rv"] / k["rgas"], k["rgas"], k["rgas"] / (k["cp"] - k["rgas"]), k["p0"]
    f = {n: fmap.resolve(table, n) for n in ("theta_m", "qv", "rho_zz", "rho_p", "rtheta_p", "rho_theta", "exner")}
    thm, qv, rz, rtp, rth, ex = (state.field(f[n]) for n in ("theta_m", "qv", "rho_zz", "rtheta_p", "rho_theta", "exner"))
    rtbase = rth - rtp
    zz = ex ** (1.0 / rcv) * p0 / (rgas * rth)
    a = 1.0 + rvord * qv
    th = thm / a
    b_ex = grads["exner"] + zz * rgas * (rtp + rtbase) * grads["pressure_p"]
    b_rtp = grads["rtheta_p"] + grads["rho_theta"] + zz * rgas * ex * grads["pressure_p"] + rcv * ex / rth * b_ex
    b_thm = grads["theta_m"] + rz * b_rtp
    return a * b_thm, grads["qv"] + th * rvord * b_thm


# ------------------------------------------------------------------- sensitivity: what a user waits for a gradient
OBJECTIVES_ENV = "WOOF_HEX_AD_OBJECTIVES"
OBJECTIVE_SCHEMA = "gpuwm-hex.ad-objective-kept/v1"
CALL_SCHEMA = "gpuwm-hex.ad-sensitivity-call/v1"
OBJECTIVE_SPEC_KEYS = ("metric", "levels", "box", "circle", "name")


def first_gradient_route(args: argparse.Namespace, table: dict | None = None) -> dict:
    """--first-gradient ROUTE against the table's ``doors.sensitivity.first_gradient``: the route's block for the
    receipt, or a refusal by name before any card or download (a route this build does not carry says why, what it
    covers, its measured seconds and its break-even against the recorded route)."""
    table = table or load_table()
    spec = table["doors"]["sensitivity"].get("first_gradient") or {}
    routes = spec.get("routes") or {}
    default = spec.get("default") or "recorded"
    asked = getattr(args, "first_gradient", None) or default
    route = routes.get(asked)
    if route is None:
        raise AdDoorError(f"--first-gradient {asked}: the table names the routes {', '.join(sorted(routes)) or 'none'}")
    if not route.get("in_this_build"):
        measured = "; ".join(f"{k}: {v}" for k, v in (route.get("measured") or {}).items())
        even = "; ".join(f"{k}: {v}" for k, v in (route.get("break_even") or {}).items())
        raise AdDoorError(f"--first-gradient {asked} is not in this build: {route.get('why_not_in_this_build')}. It "
                          f"covers {route.get('covers')}. Measured: {measured}. Break-even: {even}. "
                          f"--first-gradient {default} (the default) runs the recorded route. Nothing was downloaded")
    return {"route": asked, "default": default, "covers": route.get("covers"), "what": route.get("what"),
            "other_routes": {k: {"in_this_build": bool(v.get("in_this_build")), "covers": v.get("covers"),
                                 "break_even": v.get("break_even")}
                             for k, v in routes.items() if k != asked}}


def objective_specs(args: argparse.Namespace, table: dict | None = None) -> list[dict]:
    """The objectives of one sensitivity call: the command's own (--metric, --levels, --box, --circle), or every entry
    of --objectives FILE (a JSON list, or {"objectives": [...]}, of objects with metric, levels, box, circle and an
    optional name for its folder; a word an entry leaves out is the command's, and an entry that names a region
    replaces the command's region). Refused before any card or download: an unknown word, a metric the table lacks."""
    base = {"metric": args.metric, "levels": args.levels, "box": args.box, "circle": args.circle, "name": None}
    path = getattr(args, "objectives", None)
    if path is None:
        return [base]
    table = table or load_table()
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise AdDoorError(f"--objectives {path}: {error}. Nothing was downloaded") from error
    items = doc.get("objectives") if isinstance(doc, dict) else doc
    if not isinstance(items, list) or not items:
        raise AdDoorError(f"--objectives {path}: a JSON list of objectives (or {{\"objectives\": [...]}}) with at "
                          "least one entry. Nothing was downloaded")
    specs = []
    for k, item in enumerate(items, 1):
        if not isinstance(item, dict):
            raise AdDoorError(f"--objectives {path}: entry {k} is not an object of "
                              f"{', '.join(OBJECTIVE_SPEC_KEYS)}. Nothing was downloaded")
        unknown = sorted(set(item) - set(OBJECTIVE_SPEC_KEYS))
        if unknown:
            raise AdDoorError(f"--objectives {path}: entry {k} names {', '.join(unknown)}; an entry takes "
                              f"{', '.join(OBJECTIVE_SPEC_KEYS)}. Nothing was downloaded")
        spec = dict(base)
        if "box" in item or "circle" in item:   # an entry's own region replaces the command's
            spec["box"], spec["circle"] = None, None
        spec.update({key: (None if v is None else str(v)) for key, v in item.items()})
        if spec["box"] and spec["circle"]:
            raise AdDoorError(f"--objectives {path}: entry {k}: box and circle are alternatives")
        if spec["metric"] not in table["metrics"]:
            raise AdDoorError(f"--objectives {path}: entry {k}: metric {spec['metric']!r} is not in the table "
                              f"(metrics: {', '.join(sorted(table['metrics']))})"
                              f"{metric_not_offered(table, spec['metric'])}. Nothing was downloaded")
        specs.append(spec)
    return specs


def cloud_route_for_window(args: argparse.Namespace, table: dict) -> None:
    """A cloud-derivative route that needs the transport limiter's live derivative (table routes.<name>.needs_fct_live)
    on a window longer than the row runs the limiter live (fct_live_max_s, past which the row holds its coefficients):
    asked for by name it is refused before any download; as the row's default the row's past_live_route runs instead,
    and the command says so in one line (args.cloud_derivative then names it, so the receipt does too).
    THE BREAKAGE THIS PREVENTS: with the limiter's coefficients held, rain number's tangent on the number route grows
    until it is not finite: the README's 3 km adjoint-test with the limiter held stopped at step 28 of 59 on a non-finite
    cloud tangent in rain mass, rain number and reflectivity (one 96 GB card; the cloud tangent's norm 1.5e+12 at
    step 11, 1.0e+18 at step 27, where the gated route holds 3e+09 to 9e+09 over all 59 steps)."""
    if getattr(args, "mesh", None) or getattr(args, "checked_forward", False) or getattr(args, "steps", None) is None:
        return
    name = getattr(args, "row", None) or table["default_row"]
    row = table["rows"].get(name) or {}
    cd = (row.get("gradient_line") or {}).get("cloud_derivative") or {}
    live_max, dt = row.get("fct_live_max_s"), row.get("dt_s")
    if not cd or live_max is None or not dt or int(args.steps) * float(dt) <= float(live_max):
        return
    asked = getattr(args, "cloud_derivative", None)
    route = (cd.get("routes") or {}).get(asked or cd.get("default")) or {}
    if not route.get("needs_fct_live"):
        return
    back = cd.get("past_live_route") or "gated"
    window, live = window_length_words(int(args.steps) * float(dt)), window_length_words(float(live_max))
    if asked:
        raise AdDoorError(f"--cloud-derivative {asked}: {route['needs_fct_live']} This window is {window}, and row {name} "
                          f"holds the limiter past {live}: --cloud-derivative {back} runs it. Nothing was downloaded")
    args.cloud_derivative = back
    print(f"woof hex: cloud derivative {back} for this window ({window}; row {name} holds the transport limiter past "
          f"{live}, and its default cloud route, {cd.get('default')}, needs it live): {route['needs_fct_live']}", flush=True)


def refuse_speed_words(args: argparse.Namespace, table: dict) -> None:
    """--boundary-layer and --checked-forward against the row the command names, before any card or download."""
    if getattr(args, "keep_forward", None) is not None:
        name = None if getattr(args, "mesh", None) else (getattr(args, "row", None) or table["default_row"])
        row = (table["rows"].get(name) if name else None) or {}
        bl_name, bl_route, _ = boundary_layer_route(row, getattr(args, "boundary_layer", None))
        if not bl_route.get("keep_forward") or getattr(args, "checked_forward", False):
            why = ("--checked-forward turns the gradient line off" if getattr(args, "checked_forward", False)
                   and bl_route.get("keep_forward") else
                   bl_route.get("keep_forward_off") or "this row and route name no kept forward")
            raise AdDoorError(f"--keep-forward {args.keep_forward}: row {name or 'made for --mesh'}, boundary layer "
                              f"{bl_name}: {why}. Nothing was downloaded")
    cloud = getattr(args, "cloud_derivative", None)
    if cloud is not None:
        name = None if getattr(args, "mesh", None) else (getattr(args, "row", None) or table["default_row"])
        offered = ((((table["rows"].get(name) if name else None) or {}).get("gradient_line") or {})
                   .get("cloud_derivative") or {}).get("routes") or {}
        if cloud not in offered:
            raise AdDoorError(f"--cloud-derivative {cloud}: row {name or 'made for --mesh'} offers "
                              f"{', '.join(sorted(offered)) or 'no cloud-derivative option (it runs as the row says)'}"
                              ". Nothing was downloaded")
        if getattr(args, "checked_forward", False):
            raise AdDoorError(f"--cloud-derivative {cloud} runs on the gradient line, and --checked-forward turns "
                              "the gradient line off: give one")
    cloud_route_for_window(args, table)
    asked = getattr(args, "boundary_layer", None)
    if asked is None:
        return
    if getattr(args, "checked_forward", False) and asked != "taped":
        raise AdDoorError(f"--boundary-layer {asked} runs on the gradient line, and --checked-forward turns the "
                          "gradient line off: give one")
    name = None if getattr(args, "mesh", None) else (getattr(args, "row", None) or table["default_row"])
    row = table["rows"].get(name) if name else None
    if getattr(args, "mesh", None):
        # a row made for --mesh offers its template's routes (rows.<template>.boundary_layer). THE BREAKAGE THIS
        # PREVENTS: the held MYNN-EDMF closure leaves a 30 km made row's tangent 12.5 per cent off the real model at
        # 11 steps, and the live route that brings it to 0.77 per cent was refused on every made row
        template = ((table.get("suites") or {}).get(getattr(args, "suite", None) or "hrrrv5") or {}).get("template_row")
        row = table["rows"].get(template) if template else None
    routes = row_boundary_layer(row or {}).get("routes") or {}
    if not routes:
        if asked != "taped":
            raise AdDoorError(f"--boundary-layer {asked}: row {name or 'made for --mesh'} has no gradient line and "
                              "names no boundary-layer routes (the table's rows.<row>.gradient_line.boundary_layer "
                              "or rows.<row>.boundary_layer), so its boundary layer runs as the row says. Nothing "
                              "was downloaded")
        return
    if asked not in routes:
        raise AdDoorError(f"--boundary-layer {asked}: row {name} offers {', '.join(sorted(routes)) or 'none'}. "
                          "Nothing was downloaded")


#: The doors whose windows run the row's boundary-layer route (its default, or the one --boundary-layer names):
#: sensitivity's gradient, the adjoint-test that checks that gradient and the tangent of the same linear model.
#: THE BREAKAGE THIS PREVENTS (a 0.6.0 build, 2026-11-15): the README's first command, adjoint-test on the 12 km example,
#: checked the row's own (held, exact-route) linear model and ended FAIL (descent 4.42e-02, near-ground moisture
#: 1.18e-01 off), while the default route a sensitivity runs read PASS on the same cut through a harness only: the
#: test checked a linear model no default command computes. param-grad keeps the row's own files (its parameter
#: carrier bundle stands in for the row's, and no route of it has been run).
ROUTE_DOORS = ("tangent", "adjoint-test", "sensitivity")


def row_boundary_layer(row: dict) -> dict:
    """The row's boundary-layer routes: its gradient line's (the 3 km storm row), else the row's own
    (``rows.<row>.boundary_layer``: a row whose doors run plain windows, the 12 km cut), else {}."""
    own = ((row.get("gradient_line") or {}).get("boundary_layer")) or row.get("boundary_layer")
    if own:
        return own
    # a row made for --mesh offers its template's routes (rows.<template>.boundary_layer), also when it was made, and
    # a plan recorded with it, before the template named them. THE BREAKAGE THIS PREVENTS: adjoint-test
    # --boundary-layer guarded on a plan recorded with a made row ran the held closure (12.5 per cent off at 11 steps
    # at 30 km) under the word for the live one
    made = row.get("generated") or {}
    template = made.get("from_row")
    if template:
        table = load_table()
        bl = (table["rows"].get(template) or {}).get("boundary_layer") or {}
        recipe = (made.get("window_class") or {}).get("from_recipe")
        for cls in ((table.get("suites") or {}).get(made.get("suite")) or {}).get("window_classes") or ():
            if bl and recipe and cls.get("from_recipe") == recipe and cls.get("boundary_layer_default"):
                return {**bl, "default": cls["boundary_layer_default"]}
        return bl
    return {}


def boundary_layer_for_window(args: argparse.Namespace, row: dict, steps: int, dt_s: float | None,
                              door: str | None) -> None:
    """A made row whose class runs a live boundary layer by default (row_boundary_layer's default, the limited-area
    classes: guarded) runs it on windows up to its template's boundary_layer.default_live_max_s and the held closure
    (taped) on longer ones, when the command names no route; the command says so in one line, and args.boundary_layer
    then names taped, so the receipt does too. A route named by the command runs on any window.
    THE BREAKAGE THIS PREVENTS: on the 30 km made row the guarded route is the closest linear model to the real model
    over 11 steps (0.77 per cent at h 0.1, against 15.7 held) and the furthest at 1 hour (4.3 per cent, against 0.94
    held; its tangent below the real model's change at every step size): the table's default_live_max_what."""
    if getattr(args, "boundary_layer", None) is not None or door not in ROUTE_DOORS:
        return
    made = row.get("generated") or {}
    if not made.get("from_row") or not dt_s:
        return
    default = row_boundary_layer(row).get("default") or "taped"
    template = ((load_table()["rows"].get(made["from_row"]) or {}).get("boundary_layer")) or {}
    live_max = template.get("default_live_max_s")
    if default == "taped" or live_max is None or int(steps) * float(dt_s) <= float(live_max):
        return
    args.boundary_layer = "taped"
    print(f"woof hex {door}: boundary layer taped for this window ({window_length_words(int(steps) * float(dt_s))}; "
          f"this made row runs its default route, {default}, on windows up to {window_length_words(float(live_max))} "
          f"and the held closure past that, measured on the 30 km made row; --boundary-layer {default} runs it on "
          "any window)", flush=True)


def routes_on(row: dict, args: argparse.Namespace, door: str | None, split: Any = None) -> bool:
    """Whether this command's windows run the row's named routes (boundary layer, cloud derivative): a route door
    (ROUTE_DOORS) on a row that names them, not --checked-forward (every step boundary compared, the row's own files)
    and not a window split over cards (the runner takes the row's own words on rank plans)."""
    return bool((row_boundary_layer(row).get("routes") or (row.get("gradient_line") or {}).get("cloud_derivative"))
                and door in ROUTE_DOORS and not getattr(args, "checked_forward", False) and not split)


def boundary_layer_route(row: dict, asked: str | None) -> tuple[str, dict, bool]:
    """(route name, its table entry, whether it is the row's default) of the row's boundary-layer routes
    (row_boundary_layer); ({} for a row without any)."""
    bl = row_boundary_layer(row)
    name = asked or bl.get("default") or "taped"
    return name, (bl.get("routes") or {}).get(name) or {}, not asked or asked == bl.get("default")


def refuse_route_card_family(row_name: str, row: dict, asked: str | None, arch: str | None, card: str) -> None:
    """A boundary-layer route whose images are built for other card families than this card's is refused by name,
    before any download. THE BREAKAGE THIS PREVENTS: the tape-free route (the 3 km storm row's default) runs two
    compiled images; on a card family they were not built for the runner would stop at its first boundary-layer
    launch, after the downloads and the forward, or the door would have to run another route than the one it named."""
    name, route, is_default = boundary_layer_route(row, asked)
    images = route.get("images") or {}
    other = sorted({item["arch"] for item in images.values() if item.get("arch") and arch and item["arch"] != arch})
    if other:
        files = ", ".join(Path(item["path"]).name for _, item in sorted(images.items()))
        tag = " (this row's default)" if is_default else ""
        exact = route.get("exact_word") or "--boundary-layer taped"
        raise AdDoorError(f"row {row_name}: the boundary layer's {name} route{tag} runs images built for "
                          f"{', '.join(other)} cards only ({files}) and card {card} is {arch}: {exact} runs the "
                          "exact route, which uses none of them. Nothing was downloaded")


def route_image_paths(ctx: "Context", name: str, route: dict) -> dict[str, str]:
    """The route's images in the asset root, each checked against the build the table pins; a missing or other file
    is refused by name (never left out: the route cannot run without it, and another route is another answer)."""
    out = {}
    for var, item in sorted((route.get("images") or {}).items()):
        path = Path(ctx.root) / item["path"]
        arch = getattr(ctx, "arch", None)
        if item.get("arch") and arch and item["arch"] != arch:
            refuse_route_card_family(ctx.row_name, ctx.row, name, arch, getattr(ctx, "card", "?"))
        exact = route.get("exact_word", "--boundary-layer taped")
        if not path.is_file():
            raise AdDoorError(f"row {ctx.row_name}: the boundary layer's {name} route needs {item['path']} "
                              f"({item.get('what')}), which the asset root {ctx.root} does not hold (this build pins "
                              f"it in {route.get('pinned_in')}): name an asset root that holds it, or run {exact} "
                              "(the exact route)")
        if item.get("sha256") and _sha256_file(path) != item["sha256"]:
            raise AdDoorError(f"row {ctx.row_name}: {path} is not the build the table pins for the boundary layer's "
                              f"{name} route (sha256 {item['sha256']}): {exact} runs the exact route without it")
        out[var] = str(path)
    return out


def cloud_derivative_route(row: dict, asked: str | None) -> tuple[str, dict, bool]:
    """(name, table entry, whether it is the row's default) of the cloud-derivative option of the row's gradient
    line (table gradient_line.cloud_derivative); ("", {}, True) for a row without one."""
    cd = ((row.get("gradient_line") or {}).get("cloud_derivative")) or {}
    if not cd:
        return "", {}, True
    name = asked or cd.get("default") or "shipped"
    return name, (cd.get("routes") or {}).get(name) or {}, not asked or asked == cd.get("default")


def line_routes(row: dict, args: argparse.Namespace) -> list[tuple[str, str, dict]]:
    """(the option's word, the route's name, its table entry) of each named route of the row's gradient line this
    command runs: the boundary layer's and the cloud derivative's."""
    out = []
    name, route, _ = boundary_layer_route(row, getattr(args, "boundary_layer", None))
    if route:
        out.append(("--boundary-layer", name, route))
    name, route, _ = cloud_derivative_route(row, getattr(args, "cloud_derivative", None))
    if route:
        out.append(("--cloud-derivative", name, route))
    return out


def route_row(row: dict, args: argparse.Namespace) -> tuple[dict, list[dict]]:
    """The row a gradient-line command runs, and what its routes replaced: each asset a route names (table
    routes.<name>.assets: key -> {path, sha256, arch}) stands in the row's place, so the kernel set and images the
    runner gets, the answer check's key, the session's key and the kept re-run decisions all name the same files.
    THE BREAKAGE THIS PREVENTS: a route's kernel set handed to the runner beside the row's words would leave those
    keys naming the row's own set, and a verdict or a loaded program of one route could stand for another."""
    assets, taken = dict(row["assets"]), []
    for word, name, route in line_routes(row, args):
        for key, item in sorted((route.get("assets") or {}).items()):
            taken.append({"option": word, "route": name, "asset": key, "path": item["path"],
                          "instead_of": row["assets"].get(key), "sha256": item.get("sha256"),
                          "pinned_in": item.get("pinned_in") or route.get("assets_pinned_in")})
            assets[key] = item["path"]
    return (dict(row, assets=assets), taken) if taken else (row, [])


def _route_back(row: dict, word: str, route: dict) -> str:
    """The word a route's refusal offers in its place: the option's other word, else the exact route; and where the
    exact route itself runs a kernel set or scheme bundle of its own (on 0.6.0 builds it runs the default route's), the
    word that runs the row's own files. THE BREAKAGE THIS PREVENTS: a refusal that said "--boundary-layer taped runs
    without it" about a file the taped route runs too."""
    if route.get("other_word"):
        return route["other_word"]
    taped = (row_boundary_layer(row).get("routes") or {}).get("taped") or {}
    if word == "--boundary-layer" and (taped.get("assets") or taped.get("scheme_bundle")):
        return "--checked-forward (the row's own kernel set and scheme bundle, slower)"
    return route.get("exact_word") or "--boundary-layer taped"


def refuse_routes_card_family(row_name: str, row: dict, args: argparse.Namespace, arch: str | None, card: str) -> None:
    """A route whose own kernel set or images are built for other card families than this card's is refused by name,
    before any download (as refuse_route_card_family does for the boundary-layer route's two images)."""
    for word, name, route in line_routes(row, args):
        items = list((route.get("assets") or {}).values()) + list(((route.get("scheme_bundle") or {}).get("images")
                                                                    or {}).values())
        other = sorted({i["arch"] for i in items if i.get("arch") and arch and i["arch"] != arch})
        if other:
            back = _route_back(row, word, route)
            raise AdDoorError(f"row {row_name}: {word} {name} runs files built for {', '.join(other)} cards only and "
                              f"card {card} is {arch}: {back} runs without them. Nothing was downloaded")


def route_scheme_bundle(ctx: "Context", word: str, name: str, route: dict) -> Path | None:
    """The scheme bundle of a route that lays pinned images over the row's own bundle (table
    routes.<name>.scheme_bundle): the row's bundle copied once into the door's cache with those images in place and
    its SHA256SUMS lines moved, or None for a route without one. Each image is checked against the build the table
    pins; a missing or other file is refused by name."""
    spec = route.get("scheme_bundle")
    if not spec:
        return None
    from .ad import rows as _rows
    over = spec.get("over", "bundle")
    base = Path(ctx.assets[over])
    back = _route_back(getattr(ctx, "row", None) or {}, word, route)
    sums = {}
    for rel, item in sorted(spec["images"].items()):
        path = Path(ctx.root) / item["path"]
        if not path.is_file():
            raise AdDoorError(f"row {ctx.row_name}: {word} {name} needs {item['path']} ({spec.get('what')}), which "
                              f"the asset root {ctx.root} does not hold (this build pins it in "
                              f"{spec.get('pinned_in')}): name an asset root that holds it, or run {back}")
        sums[rel] = _sha256_file(path)
        if item.get("sha256") and sums[rel] != item["sha256"]:
            raise AdDoorError(f"row {ctx.row_name}: {path} is not the build the table pins for {word} {name} "
                              f"(sha256 {item['sha256']}): {back} runs without it")
    listing = base / "SHA256SUMS"
    if not listing.is_file():
        raise AdDoorError(f"row asset {over} ({base}) has no SHA256SUMS: {word} {name} lays its images over that "
                          f"bundle and cannot name it; {back} runs without them")
    key = _json_sum({"base": _sha256_file(listing), "images": sums, "name": spec.get("name")})
    out = _rows.rows_cache().parent / "bundles" / f"{spec.get('name') or 'bundle'}-{key[:16]}"
    done = out.with_name(out.name + ".json")
    if done.is_file() and out.is_dir():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(base, tmp)
    lines = []
    for line in (tmp / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        _, _, rel = line.partition("  ")
        if rel in sums:
            line = f"{sums[rel]}  {rel}"
        lines.append(line)
    for rel, item in sorted(spec["images"].items()):
        if not (tmp / rel).is_file():
            shutil.rmtree(tmp, ignore_errors=True)
            raise AdDoorError(f"row asset {over} ({base}) has no {rel}: {word} {name} replaces that image; {back} "
                              "runs without it")
        shutil.copyfile(Path(ctx.root) / item["path"], tmp / rel)
    (tmp / "SHA256SUMS").write_text(chr(10).join(lines) + chr(10), encoding="utf-8")
    shutil.rmtree(out, ignore_errors=True)
    os.replace(tmp, out)
    _write_json(done, {"base": str(base), "base_sha256sums_sha256": _sha256_file(listing), "images": sums,
                       "name": spec.get("name"), "sha256sums_sha256": _sha256_file(out / "SHA256SUMS")})
    return out


def check_route_assets(ctx: "Context") -> None:
    """After the asset root is known: every file a route put in the row's place is there and is the pinned build
    (refused by name otherwise), and a route's scheme bundle is made and takes the row's bundle's place."""
    for item in ctx.route_assets:
        path = Path(ctx.root) / item["path"]
        back = (_route_back(ctx.row, item["option"], {}) if item["option"] == "--boundary-layer"
                else f"{item['option']} shipped")
        if not path.exists():
            raise AdDoorError(f"row {ctx.row_name}: {item['option']} {item['route']} runs {item['path']}, which the "
                              f"asset root {ctx.root} does not hold (this build pins it in {item.get('pinned_in')}): "
                              f"name an asset root that holds it, or run {back}")
        if item.get("sha256") and path.is_file() and _sha256_file(path) != item["sha256"]:
            raise AdDoorError(f"row {ctx.row_name}: {path} is not the build the table pins for {item['option']} "
                              f"{item['route']} (sha256 {item['sha256']}): {back} runs without it")
    for word, name, route in line_routes(ctx.row, ctx.args):
        bundle = route_scheme_bundle(ctx, word, name, route)
        if bundle is not None:
            over = (route.get("scheme_bundle") or {}).get("over", "bundle")
            ctx.route_assets.append({"option": word, "route": name, "asset": over, "path": str(bundle),
                                     "instead_of": ctx.row["assets"].get(over),
                                     "pinned_in": route["scheme_bundle"].get("pinned_in"),
                                     "sha256sums_sha256": _sha256_file(bundle / "SHA256SUMS")})
            ctx.assets[over] = bundle
            ctx.row = dict(ctx.row, assets={**ctx.row["assets"], over: str(bundle)})


def route_window_words(ctx: "Context", door: str) -> tuple[dict, dict | None]:
    """(the environment every window of the command takes, the receipt's block) for a route door that runs no
    gradient line: adjoint-test and tangent on any row with routes, and every route door of a row whose routes are
    its own (rows.<row>.boundary_layer: the 12 km cut). The boundary-layer route's window_env and its images
    (route_image_paths, each checked against its pin); its kernel set and scheme bundle already stand in the row's
    place (route_row, check_route_assets). Not taken: the gradient line's own words (the moved start, the checkpoint
    sizes, the route's replace_args and gradient_env, a kept forward), which belong to sensitivity's gradient windows.
    THE BREAKAGE THIS PREVENTS (a 0.6.0 build): adjoint-test ran the row's held linear model and ended FAIL on the README's
    first command while the default route sensitivity runs passed the same test in a harness; the test must check the
    linear model the user's gradient comes from, and say which one it checked."""
    name, route, is_default = boundary_layer_route(ctx.row, getattr(ctx.args, "boundary_layer", None))
    if not route:
        return {}, None
    env = {k: str(v) for k, v in (route.get("window_env") or {}).items()}
    images = route_image_paths(ctx, name, route)
    env.update(images)
    error_words, _ = route_error_words(ctx, route)
    tag = " (this row's default)" if is_default else f" (--boundary-layer {name})"
    if route.get("exact_word"):   # a route that is not the exact one says so every time, and how to ask for exact
        print(f"woof hex {door}: boundary layer {name}{tag}: {error_words}; {route['exact_word']} runs the exact "
              "route", flush=True)
    else:
        print(f"woof hex {door}: boundary layer {name}{tag}: {error_words}", flush=True)
    block = {"route": name, "default": row_boundary_layer(ctx.row).get("default"), "is_default": is_default,
             "error": error_words, "window_env": env, "images": images, "exact_word": route.get("exact_word"),
             "what": route.get("what"), "route_assets": list(getattr(ctx, "route_assets", None) or []),
             "every_window": "every window of this command (the test windows, the answer check and its proof) ran "
                             "this route's linear model",
             "not_taken": "the gradient line's own words (moved start, checkpoint sizes, the route's replace_args "
                          "and gradient_env, a kept forward): words of sensitivity's gradient windows only"}
    return env, block


def route_error_words(ctx: "Context", route: dict) -> tuple[str, bool]:
    """(what the command prints of a route's difference from the exact answer, whether measured figures are in it).
    The figures are printed only when the command runs the plan and window length they were measured on; on any
    other plan or length it says that none was measured. THE BREAKAGE THIS PREVENTS: the command printed "the
    sensitivity within 1.2e-6 and the raw gradient within 7.8e-6" on every run, figures of another recording; on the
    packed example its own answer files differed by 3.0e-06 (temperature) to 9.6e-05 (stored pressure). And a
    figure typed into a sentence stayed when the measurement moved: the command printed moisture 1.1e-06 and stored
    pressure 9.6e-05 where the next card run read 1.315e-06 and 9.778e-05. The figures are table data now
    (error_measured.figures, every field of the two answer files as the card run's compare wrote them), and the
    sentence is made from them, each rounded up (error_figures)."""
    words = str(route.get("error"))
    measured = route.get("error_measured")
    if not measured:
        return words, False
    if (str(getattr(ctx, "plan_how", "")).startswith("the row's example plan")
            and int(getattr(ctx, "steps", 0) or 0) == int(measured.get("steps") or -1)):
        return f"{words}: {error_figures_text(measured)}", True
    return f"{words}; {measured.get('elsewhere') or 'no figure was measured on this plan and window length'}", False


def round_up(value: float, digits: int = 2) -> float:
    """``value`` (positive) rounded UP to ``digits`` significant digits, in decimal: 1.315e-06 -> 1.4e-06, and
    3.0e-06 stays 3.0e-06. A printed figure made this way is never under its measurement."""
    from decimal import ROUND_CEILING, Decimal
    number = Decimal(repr(float(value)))
    if number <= 0:
        return 0.0
    return float(number.quantize(Decimal(1).scaleb(number.adjusted() - (digits - 1)), rounding=ROUND_CEILING))


def error_figures(measured: dict) -> list[dict]:
    """The fast-against-exact figures a route prints, from the table's measured ones (error_measured.figures:
    variable -> the largest difference over the field's largest value, one figure a target; error_measured.printed:
    the fields named in the sentence, each with the answer file's variables it stands for): per printed field the
    largest measured figure of its variables over the targets, and that figure rounded up; and the largest of every
    field of the answer files."""
    figures = measured.get("figures") or {}
    out = []
    for item in measured.get("printed") or []:
        worst = max(float(x) for v in item["variables"] for x in figures[v])
        out.append({"name": item["name"], "variables": list(item["variables"]), "measured": worst,
                    "printed": round_up(worst)})
    every = max((float(x) for values in figures.values() for x in values), default=None)
    if every is not None:
        out.append({"name": f"every one of the {len(figures) + len(measured.get('both_zero') or [])} fields",
                    "variables": sorted(figures), "measured": every, "printed": round_up(every), "all_fields": True})
    return out


def error_figures_text(measured: dict) -> str:
    """The sentence the command prints: where and when the figures were measured, each figure rounded up."""
    rows = error_figures(measured)
    named = ", ".join(f"{r['name']} {r['printed']:.1e}" for r in rows if not r.get("all_fields"))
    every = next((r for r in rows if r.get("all_fields")), None)
    tail = f"; {every['name']} within {every['printed']:.1e}" if every else ""
    return f"{measured['measured_what']}: {named}{tail}; {measured['pattern']}"


def predict_book(ctx: "Context", name: str, route: dict) -> str | None:
    """The file a route's gradient windows keep their taped re-run decisions in (table predict_book), or None when
    the door cannot name the window (no plan sum or runner) or cannot make its folder: the runner then keeps the
    decisions in memory only, for the later objectives of the same loaded program."""
    plan_sha = (getattr(getattr(ctx, "fmap", None), "doc", None) or {}).get("plan_sha256")
    runner = getattr(ctx, "runner", None)
    if not route.get("predict_book") or not plan_sha or runner is None:
        return None
    from . import assets as _assets
    kernels = Path(ctx.root) / str(ctx.row["assets"].get("kernels"))
    key = _json_sum({"plan_sha256": plan_sha, "kernels": _content_sum(kernels), "runner": _sha256_file(Path(runner)),
                     "route": name, "steps": ctx.steps, "window_env": route.get("window_env"),
                     "gradient_env": route.get("gradient_env")})
    folder = _assets.cache_root() / "ad" / "books"
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return str(folder / f"{key[:32]}.book")


def objectives_dir() -> Path:
    declared = os.environ.get(OBJECTIVES_ENV)
    if declared:
        return Path(declared).expanduser().absolute()
    from . import assets as _assets
    return _assets.cache_root() / "ad" / "objectives"


def _file_stat(path: Path | None) -> dict | None:
    try:
        st = Path(path).stat()
    except (OSError, TypeError):
        return None
    return {"size": st.st_size, "inode": st.st_ino, "mtime_ns": st.st_mtime_ns, "ctime_ns": st.st_ctime_ns}


def objective_key(ctx: "Context", spec: dict) -> dict:
    """What a sensitivity objective's terms depend on, and nothing else: the plan (its sha256), the field the metric
    reads at the window end (allocation, offset, shape), the metric's table entry, the level and region words, the
    mesh (its grid file by name, size, inode and change times, and the row's cell and edge counts) and the door
    program. Not the output folder, the card or the step count (the metric is read at the window end of any window)."""
    from . import __version__
    metric = ctx.table["metrics"][spec["metric"]]
    f = ctx.fmap.resolve(ctx.table, metric["field"], ctx.row, window_end=True)
    return {"schema": OBJECTIVE_SCHEMA, "plan_sha256": ctx.fmap.doc["plan_sha256"],
            "field": {k: f.get(k) for k in ("name", "alloc", "offset", "shape", "read_shape", "grid", "units")},
            "metric": spec["metric"], "metric_spec": metric,
            "levels": spec.get("levels"), "box": spec.get("box"), "circle": spec.get("circle"),
            "mesh": {"grid": str(ctx.grid), "grid_file": _file_stat(ctx.grid), "cells": ctx.row.get("cells"),
                     "edges": ctx.row.get("edges")},
            "door_code": {"version": __version__, "ad_doors_sha256": _sha256_file(Path(__file__))}}


def _write_json_compact(path: Path, value: Any) -> None:
    """A JSON file in one line (an objective holds a term per cell and level: 2,258,402 at the 3 km box)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(value, separators=(",", ":"), default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def _place(src: Path, dest: Path) -> None:
    """``dest`` holds ``src``'s bytes: a hard link where the two folders share a file system, else a copy."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    try:
        os.link(src, dest)
    except OSError:
        shutil.copyfile(src, dest)


def objective_file(ctx: "Context", spec: dict, dest: Path) -> tuple[dict, Path, dict]:
    """The objective of ``spec`` written at ``dest``: (the objective without its terms, ``dest``, its block for the
    receipt). The terms are made once per key (objective_key) and kept in the door's cache (``objectives_dir``,
    override WOOF_HEX_AD_OBJECTIVES); the same objective asked again on the same plan takes the kept file, checked
    against its sha256, instead of being made again. THE COST THIS REMOVES: making the terms of a whole-box objective
    at every command (the measured fast route translates its objective once and keeps it: 45.7 s off every later
    59-step 3 km gradient). A kept file that was changed, or a key that differs, makes it again and says why."""
    key = objective_key(ctx, spec)
    ksum = _json_sum(key)
    cache = objectives_dir()
    kept, meta_path = cache / f"{ksum[:32]}.json", cache / f"{ksum[:32]}.meta.json"
    why, meta = None, None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        meta = None
    except (OSError, ValueError) as error:
        why = f"its kept file cannot be read ({error})"
    if meta is not None:
        if meta.get("schema") != OBJECTIVE_SCHEMA or meta.get("key") != key:
            why, meta = "the kept file's key is not this objective's", None
        elif not kept.is_file() or _sha256_file(kept) != meta.get("sha256"):
            why, meta = "the kept file was changed or removed since it was made", None
    if meta is not None:
        _place(kept, dest)
        return dict(meta["objective"]), dest, {"how": "kept", "file": str(kept), "sha256": meta["sha256"],
                                               "made_utc": meta.get("made_utc"), "key_sha256": ksum}
    objective = metric_objective(ctx.table, ctx.fmap, ctx.geometry, spec["metric"], spec.get("levels"),
                                 spec.get("box"), spec.get("circle"), ctx.row)
    _write_json_compact(dest, objective)
    small = {k: v for k, v in objective.items() if k != "terms"}
    small["terms_n"] = len(objective["terms"])
    block = {"how": "made", "file": None, "key_sha256": ksum, "why_made": why or "not yet made on this plan"}
    try:
        cache.mkdir(parents=True, exist_ok=True)
        tmp = kept.with_name(kept.name + f".tmp{os.getpid()}")
        shutil.copyfile(dest, tmp)
        os.replace(tmp, kept)
        sha = _sha256_file(kept)
        _write_json(meta_path, {"schema": OBJECTIVE_SCHEMA, "key": key, "sha256": sha, "objective": small,
                                "made_utc": time.strftime("%Y-%m-%d %H:%MZ", time.gmtime())})
        block.update(file=str(kept), sha256=sha)
    except OSError as error:
        print(f"woof hex sensitivity: the objective was not kept ({error}); the next command makes it again",
              file=sys.stderr, flush=True)
    return small, dest, block


def gradient_line(ctx: "Context", args: argparse.Namespace) -> dict | None:
    """The row's gradient line (``rows.<row>.gradient_line`` under the door's ``doors.sensitivity.gradient_line``),
    or None, with the reason kept in ``ctx.gradient_line_off``: the row names none, --checked-forward, or a window
    split over cards (the runner refuses a moved start on rank plans)."""
    own = ctx.row.get("gradient_line")
    door = ctx.table["doors"].get(ctx.door, {}).get("gradient_line")
    why = None
    if not own or not door:
        why = "the row names no gradient line"
    elif getattr(args, "checked_forward", False):
        why = "--checked-forward: every step boundary of the forward compared with the recording"
    elif getattr(ctx, "split", None):
        why = "a window split over cards: the runner does not take a moved start on rank plans"
    ctx.gradient_line_off = why
    return None if why else {**door, "row": own}


def gradient_line_words(ctx: "Context", line: dict, args: argparse.Namespace, folder: Path) -> dict:
    """The gradient line's runner words (extra words with the empty move, flags set in place of the row's, the
    environment) and its block for the receipt. A word whose file the asset root lacks is left out and named."""
    own = line["row"]
    move = folder / "empty-move.json"
    _write_json(move, line["empty_move"])
    extra = list(own.get("window_args") or []) + ["--x0-dir", str(move)]
    env = {k: str(v) for k, v in (own.get("window_env") or {}).items()}
    used, left = {}, {}
    for var, item in (own.get("optional_assets") or {}).items():
        path = Path(ctx.root) / item["path"]
        arch = getattr(ctx, "arch", None)
        if item.get("arch") and arch and item["arch"] != arch:
            left[var] = f"{item['path']} is an {item['arch']} image and card {ctx.card} is {arch}"
        elif not path.is_file():
            left[var] = f"{item['path']} is not in the asset root"
        elif item.get("sha256") and _sha256_file(path) != item["sha256"]:
            left[var] = f"{path} is not the build the table pins (sha256 {item['sha256']})"
        else:
            env[var] = str(path)
            used[var] = str(path)
    bl = own.get("boundary_layer") or {}
    asked = getattr(args, "boundary_layer", None) or bl.get("default") or "taped"
    route = (bl.get("routes") or {}).get(asked)
    if route is None:
        raise AdDoorError(f"--boundary-layer {asked}: row {ctx.row_name} offers "
                          f"{', '.join(sorted(bl.get('routes') or {})) or 'none'}")
    env.update({k: str(v) for k, v in (route.get("window_env") or {}).items()})
    images = route_image_paths(ctx, asked, route)
    env.update(images)
    # words of the gradient windows only (the answer check's windows leave them out: check_line_words)
    gradient_only = {k: str(v) for k, v in (route.get("gradient_env") or {}).items()}
    book = predict_book(ctx, asked, route) if gradient_only else None
    if book:
        gradient_only[route["predict_book"]["env"]] = book
    kf_plan = getattr(ctx, "keep_forward", None) or {}
    if kf_plan.get("on"):   # the forward kept between gradients of one forecast: a word of the gradient windows only
        gradient_only.update(kf_plan["env"])
    env.update(gradient_only)
    replace = {**dict(own.get("replace_args") or {}), **dict(route.get("replace_args") or {})}
    ram_gib = bstore_ram_gib(own.get("bstore_ram"), host_available_mib())
    if ram_gib:
        extra += ["--bstore-ram-gib", str(ram_gib)]
    error_words, error_measured = route_error_words(ctx, route)
    if route.get("exact_word"):   # a route that is not the exact one says so every time, and how to ask for exact
        tag = " (this row's default)" if asked == bl.get("default") else ""
        print(f"woof hex sensitivity: boundary layer {asked}{tag}: {error_words}; {route['exact_word']} "
              "runs the exact route", flush=True)
    elif asked != (bl.get("default") or "taped"):
        print(f"woof hex sensitivity: boundary layer {asked} (--boundary-layer {asked}): {error_words}",
              flush=True)
    if route.get("runs"):
        print(f"woof hex sensitivity: boundary layer {asked}: {route['runs']}", flush=True)
    warning = route_window_warning(route, ctx.steps, ctx.row.get("dt_s"))
    if warning:   # said, not refused: the window still runs
        print(f"woof hex sensitivity: WARNING: --boundary-layer {asked}: {warning}", flush=True)
    kf_route = route.get("keep_forward") or {}
    if kf_plan.get("on"):
        print(f"woof hex sensitivity: forward kept between gradients of one forecast: {kf_route.get('says')}",
              flush=True)
    elif kf_plan.get("offered") or route.get("keep_forward_off"):
        say_forward_not_kept(ctx, kf_plan.get("why"))
    cd_name, cd_route, cd_default = cloud_derivative_route(ctx.row, getattr(args, "cloud_derivative", None))
    if cd_route:
        cd_tag = " (this row's default)" if cd_default else ""
        print(f"woof hex sensitivity: cloud derivative {cd_name}{cd_tag}: {cd_route.get('says')}", flush=True)
    block = {"on": True, "what": line.get("what"), "row_what": own.get("what"),
             "window_args": list(own.get("window_args") or []),
             "x0_dir": str(move), "replace_args": dict(own.get("replace_args") or {}), "window_env": env,
             "bstore_ram_gib": ram_gib, "optional_assets_used": used, "optional_assets_left_out": left,
             "boundary_layer": {"route": asked, "default": bl.get("default"), "error": error_words,
                                "error_figures_of_this_plan": error_measured, "runs": route.get("runs"),
                                "engine": route.get("engine"), "images": images,
                                "exact_word": route.get("exact_word"), "gradient_only_env": gradient_only},
             "cloud_derivative": ({"route": cd_name, "default": (own.get("cloud_derivative") or {}).get("default"),
                                   "says": cd_route.get("says")} if cd_route else None),
             "keep_forward": {"on": bool(kf_plan.get("on")), "why_off": kf_plan.get("why"),
                              "env": kf_plan.get("env"), "what": kf_route.get("what"),
                              "host_mib": (getattr(ctx, "store_need", None) or {}).get("keep_forward_mib")},
             "window_limit_warning": warning,
             "route_assets": list(getattr(ctx, "route_assets", None) or []),
             "not_yet_in_assets": own.get("not_yet_in_assets"), "measured": own.get("measured"),
             "boundaries": "the forward's step boundaries are not compared with the recording in this window; the "
                           "answer check's windows compare them on this plan and operator",
             "off_word": line.get("off_word")}
    for var, why in left.items():
        print(f"woof hex sensitivity: gradient line: {var} left out: {why}", flush=True)
    block["replace_args"] = dict(replace)
    return {"extra": extra, "env": env, "replace": replace, "block": block, "gradient_only": sorted(gradient_only)}


def check_line_words(words: dict | None) -> dict | None:
    """The answer check's windows on a row whose gradient window runs the gradient line: the line's words that do not
    change what the check proves (the checkpointed sweep with its checkpoints in host memory, the environment words,
    the optional images, the tape and scratch sizes, as the stack's own derivative check ran them), less the moved
    start (--x0-dir), because the check compares every forward step boundary with the recording (the table's
    gradient_line.check_windows). THE COST THIS REMOVES: the check's windows kept
    every step boundary (445 GB written at 59 steps on the 3 km storm box, 2026-11-14) where the gradient they vouch
    for keeps ten; a check made on another route did not vouch for the fast one anyway."""
    if not words:
        return None
    extra, it, i = [], list(words["extra"]), 0
    while i < len(it):
        if it[i] == "--x0-dir":
            i += 2
            continue
        extra.append(it[i])
        i += 1
    only = set(words.get("gradient_only") or ())
    not_taken = {"--x0-dir": "the check compares every forward step boundary with the recording"}
    not_taken.update({k: "a word of the gradient windows only: the check was measured without it" for k in sorted(only)})
    return {"extra": extra, "env": {k: v for k, v in words["env"].items() if k not in only},
            "replace": dict(words.get("replace") or {}), "not_taken": not_taken}


def _check_line_key(words: dict | None) -> str | None:
    """The check-window words as a key part: the words, the --bstore-ram-gib value left out (it moves with the host
    and the same bytes are restored either way), an image named by its sha256 rather than its path."""
    if not words:
        return None
    args, it = [], list(words["extra"])
    for k, w in enumerate(it):
        if k > 0 and it[k - 1] == "--bstore-ram-gib":
            continue
        args.append(w)
    env = {k: (_sha256_file(Path(v)) if os.path.isabs(str(v)) and Path(v).is_file() else v)
           for k, v in sorted(words["env"].items())}
    return _json_sum({"args": args, "env": env, "replace": dict(words.get("replace") or {})})


def gradient_line_validity(validity: dict, line: dict | None) -> dict:
    """A gradient-line window's validity. Its forward ran from the recording's start as an empty move, so the runner
    compared none of its step boundaries with the recording (an engine without the honest log still writes "equal"
    for them; one with it writes "not compared"): no boundary is counted as checked here, and the window stands on
    the answer check, whose windows compare every boundary of the same plan and operator. THE BREAKAGE THIS PREVENTS:
    a receipt reading "every step boundary equal to the recording" for a window that compared none."""
    out = dict(validity)
    out.update({"boundaries_runner_logged_checked": validity.get("boundaries_checked"),
                "boundaries_runner_logged_equal": validity.get("boundaries_equal"),
                "boundaries_checked": 0, "boundaries_equal": 0, "boundaries_compared": False,
                "valid": not validity.get("refused_beyond_row_list"),
                "what": "valid: no launch kind ran as identity beyond the row's refuse list. The forward ran as the "
                        "gradient line (the recording's start as an empty move), so its step boundaries were not "
                        "compared with the recording in this window; the answer check's windows compare them on "
                        "this plan and operator"})
    return out


class NotLoaded(AdDoorError):
    """The runner ended after its first objective without answering in its loaded-program folder (a runner that does
    not serve requests): ``doc`` is that objective's window receipt."""
    def __init__(self, message: str, doc: dict):
        super().__init__(message)
        self.doc = doc


class LoadedProgram:
    """One runner kept loaded for the objectives of one sensitivity call (the table's
    ``doors.sensitivity.gradient_line.loaded_program``). Started with --resident DIR it answers DIR/NAME.done for its
    first objective, then serves DIR/NAME.req requests in name order in the same process (the runner's resident
    route): the plan, the card context, the kernel set, the scheme images, the arenas and the page-locked pools stay
    loaded. A request changes only REQUEST_WORDS. THE COST THIS REMOVES from every objective after the first: the
    program's start, its first step's warm-up and its exit (59-step 3 km gradient: 298.0 s from a loaded program
    against 322.5 s for a fresh program's first, the same answer files). The program is a child of the command and
    ends with it: the door asks it to stop, and ends it by its own pid if it has not ended stop_wait_s later."""
    REQUEST_WORDS = ("out", "objective", "dump-grad", "bstore")

    def __init__(self, rdir: Path, idle_s: float = 120.0, stop_wait_s: float = 120.0):
        self.rdir, self.idle_s, self.stop_wait_s = Path(rdir), float(idle_s), float(stop_wait_s)
        self.proc = None
        self.peak: dict[str, int] = {}
        self.served: list[dict] = []
        self.first = None
        self._streams = ()

    def _watch(self, peak: dict) -> None:
        for bus, mib in process_card_mib(self.proc.pid).items():
            peak[bus] = max(peak.get(bus, 0), mib)
            self.peak[bus] = max(self.peak.get(bus, 0), mib)

    def start(self, argv: list[str], env: dict, plan: Path, card: str, name: str, out: Path) -> dict:
        refuse_no_local_gpu()
        from .ad import rows as _rows
        _rows.forget_plan_docs()   # as run_window: the parsed plan is let go before the runner starts
        shutil.rmtree(self.rdir, ignore_errors=True)
        self.rdir.mkdir(parents=True)
        out.mkdir(parents=True, exist_ok=True)
        (out / "bstore").mkdir(exist_ok=True)
        self.argv = list(argv) + ["--resident", str(self.rdir), "--resident-name", name,
                                  "--resident-idle-s", f"{self.idle_s:g}"]
        refuse_hidden_card()
        refuse_planted_env("session program", before_download=False)
        child = {**os.environ, **env, "CUDA_VISIBLE_DEVICES": str(card), "HEXAD_LIBRARY_PATH": library_path()}
        line = " ".join(f"{k}={v}" for k, v in sorted(env.items())) + " " + " ".join(self.argv[1:]) + "\n"
        (self.rdir / "cmd").write_text(line)
        (out / "cmd").write_text(line)
        so, se = open(self.rdir / "stdout", "wb"), open(self.rdir / "stderr", "wb")
        self._streams = (so, se)
        self.t0 = time.monotonic()
        self.first = name
        self.proc = subprocess.Popen(self.argv, cwd=plan, env=child, stdout=so, stderr=se)
        return self._answer(name, out)

    def request(self, name: str, words: dict, out: Path) -> dict:
        bad = sorted(set(words) - set(self.REQUEST_WORDS))
        if bad:
            raise AdDoorError(f"a loaded program's request changes only {', '.join(self.REQUEST_WORDS)} "
                              f"(asked: {', '.join(bad)})")
        if self.proc is None or self.proc.poll() is not None:
            raise AdDoorError(f"the loaded program in {self.rdir} is not running")
        out.mkdir(parents=True, exist_ok=True)
        (out / "bstore").mkdir(exist_ok=True)
        tmp = self.rdir / f"{name}.req.part"
        tmp.write_text(json.dumps({k: str(v) for k, v in words.items()}))
        os.replace(tmp, self.rdir / f"{name}.req")
        return self._answer(name, out)

    def _answer(self, name: str, out: Path) -> dict:
        done = self.rdir / f"{name}.done"
        t0, last, peak, answer = time.monotonic(), 0.0, {}, None
        while answer is None:
            if done.is_file():
                try:
                    answer = json.loads(done.read_text())
                    break
                except ValueError:
                    pass
            rc = self.proc.poll()
            if rc is not None and not done.is_file():
                self._ended(name, out, rc, t0)
            now = time.monotonic()
            if now - last >= 2.0:
                self._watch(peak)
                last = now
            time.sleep(0.05)
        wall = time.monotonic() - t0
        if answer.get("rc") != 0:
            error = str(answer.get("error"))
            kind = CardMemoryError if "CUDA status 2" in error else AdDoorError
            raise kind(f"the loaded program stopped on objective {name}: {error[:400]} (log {self.rdir / 'stderr'})")
        receipt = out / "window.json"
        if not receipt.is_file():
            raise AdDoorError(f"the loaded program answered {name} but wrote no {receipt}")
        doc = json.loads(receipt.read_text())
        doc["door_wall_s"] = wall
        doc["door_card_peak_mib"] = max(peak.values(), default=0) or None
        doc["door_card_peaks_mib"] = peak
        doc["loaded_program"] = {"name": name, "index": answer.get("index"), "pid": self.proc.pid,
                                 "runner_wall_s": answer.get("wall_s"), "dir": str(self.rdir),
                                 "first": name == self.first}
        self.served.append({"name": name, "index": answer.get("index"), "wall_s": round(wall, 1),
                            "runner_wall_s": answer.get("wall_s"), "out": str(out)})
        return doc

    def _ended(self, name: str, out: Path, rc: int, t0: float) -> None:
        lines = (self.rdir / "stderr").read_text(errors="replace").strip().splitlines() if (
            self.rdir / "stderr").is_file() else []
        if rc == 0 and name == self.first and not self.served and (out / "window.json").is_file():
            doc = json.loads((out / "window.json").read_text())
            doc["door_wall_s"] = time.monotonic() - t0
            doc["door_card_peak_mib"] = max(self.peak.values(), default=0) or None
            doc["door_card_peaks_mib"] = dict(self.peak)
            raise NotLoaded(f"the runner ended after objective {name} without serving requests in {self.rdir} (a "
                            "runner without the loaded-program route)", doc)
        tail = [x[:300] for x in lines if not x.startswith("{")][-2:] or [x[:300] for x in lines[-1:]]
        kind = CardMemoryError if any("CUDA status 2" in x for x in lines if not x.startswith("{")) else AdDoorError
        raise kind(f"the loaded program exited {rc} before answering objective {name}: {' | '.join(tail)} (full log "
                   f"{self.rdir / 'stderr'})")

    def stop(self) -> dict:
        """Ask the program to end (DIR/stop) and wait for it; end it by its own pid if it has not ended in time."""
        if self.proc is None:
            return {"on": False}
        killed = False
        if self.proc.poll() is None:
            (self.rdir / "stop").write_text("")
            try:
                self.proc.wait(timeout=self.stop_wait_s)
            except subprocess.TimeoutExpired:
                print(f"woof hex sensitivity: the loaded program (pid {self.proc.pid}) did not end "
                      f"{self.stop_wait_s:g} s after it was asked to stop: ended", file=sys.stderr, flush=True)
                self.proc.kill()
                self.proc.wait()
                killed = True
        for stream in self._streams:
            stream.close()
        for item in self.served:
            shutil.rmtree(Path(item["out"]) / "bstore", ignore_errors=True)
        return {"on": True, "dir": str(self.rdir), "pid": self.proc.pid, "rc": self.proc.returncode,
                "ended_by_door": killed, "served": self.served, "card_peak_mib": max(self.peak.values(), default=0)
                or None, "wall_s": round(time.monotonic() - self.t0, 1), "stderr": str(self.rdir / "stderr"),
                "what": "one runner served every objective of this call; it was asked to stop after the last"}


def _request_words(argv: list[str], out: Path) -> dict:
    """A later objective's words for the loaded program, read from the runner line its fresh window would get."""
    def word(flag: str) -> str:
        return argv[argv.index(flag) + 1]
    return {"out": str(out), "objective": word("--objective"), "dump-grad": word("--dump-grad"),
            "bstore": str(out / "bstore")}


def _sensitivity_inputs(ctx: "Context") -> tuple[list[dict], list[int]]:
    inputs = [ctx.fmap.resolve(ctx.table, n) for n in ctx.table["sensitivity_inputs"]
              if ctx.table["fields"][n]["field"] in ctx.fmap.fields]
    allocs = {a for f in inputs for a, _ in f["copies"]}
    sw = _start_winds(ctx) if any(f["name"] == "u" for f in inputs) else None
    if sw is not None:   # the physics' start cell winds: their gradient is folded into dJ/du (_sensitivity_answer)
        allocs |= set(_startwinds.dump_allocations(sw))
    return inputs, sorted(allocs)


def _start_winds(ctx: "Context") -> dict | None:
    """Where a one-card plan keeps the physics' start cell winds (woof.hex.ad.startwinds), or None on a split window
    (every rank plan rebuilds them from u at the start of step 1, so the ranks' dJ/du already carries the term). Read
    from plan.json once per plan and kept beside the kept objectives, by the plan's sha256."""
    if getattr(ctx, "split", None):
        return None
    if getattr(ctx, "_start_winds_info", None) is not None:
        return ctx._start_winds_info or None
    sha = ctx.fmap.doc["plan_sha256"]
    kept = objectives_dir().parent / "start-cell-winds" / f"{sha}.json"
    info = None
    try:
        doc = json.loads(kept.read_text(encoding="utf-8"))
        if doc.get("schema") == _startwinds.SCHEMA and doc.get("plan_sha256") == sha:
            info = doc
    except (OSError, ValueError):
        pass
    if info is None:
        from .ad import rows as _rows
        info = {**_startwinds.plan_info(ctx.plan, plan_doc=_rows.plan_doc(ctx.plan)), "plan_sha256": sha}
        try:
            kept.parent.mkdir(parents=True, exist_ok=True)
            _write_json(kept, info)
        except OSError:
            pass   # read again next time
    ctx._start_winds_info = info if info.get("needed") else {}
    return ctx._start_winds_info or None


def _objective_stage(kept: dict) -> str:
    return ("inputs (objective kept from an earlier command, field map)" if kept["how"] == "kept"
            else "inputs (objective made and kept, field map)")


def run_sensitivity(args: argparse.Namespace) -> int:
    table = load_table()
    plant_request(args, table, "sensitivity")   # --plant-error and the runner's planting words: refused, nothing downloaded
    first = first_gradient_route(args, table)   # every refusal of the speed words comes before any card or download
    specs = objective_specs(args, table)
    refuse_speed_words(args, table)
    say_storm_measures(table, args, "sensitivity", [spec.get("metric") for spec in specs])
    route = gradient_route(args, "sensitivity")
    if route["carried"]:   # made on a companion mesh of the same region, carried to this one
        if getattr(args, "objectives", None) is not None:
            raise AdDoorError("--objectives with a gradient on a companion mesh: that route takes one objective a "
                              "command (--gradient-mesh native runs several on the mesh itself). Nothing was "
                              "downloaded")
        return _sensitivity_carried(args, route)
    ctx = Context(args, "sensitivity")
    ctx.gradient_route = route
    ctx.first_gradient = first
    for note in ctx.row.get("accuracy_notes") or []:   # measured limits of the row's gradient, said on every run
        print(f"woof hex sensitivity: CAUTION: {note}", flush=True)
    try:
        if getattr(args, "objectives", None) is not None:
            return _sensitivity_many(ctx, args, specs)
        return _sensitivity(ctx, args)[0]
    finally:   # the session's last line (left loaded, or ended and why), and its lock given back
        if getattr(ctx, "session", None) is not None:
            ctx.session.finish()


def _speed_blocks(ctx: "Context", words: dict | None, kept: dict) -> dict:
    return {"first_gradient": getattr(ctx, "first_gradient", None) or first_gradient_route(ctx.args, ctx.table),
            "gradient_line": words["block"] if words else {"on": False, "why": getattr(ctx, "gradient_line_off", None)},
            "objective_kept": kept}


def _sensitivity(ctx: "Context", args: argparse.Namespace) -> tuple[int, dict]:
    """The sensitivity door on the row of ``ctx``: (exit code, the fields and receipt for a caller that carries them)."""
    t0 = time.monotonic()
    spec = {"metric": args.metric, "levels": args.levels, "box": args.box, "circle": args.circle, "name": None}
    objective, obj_path, kept = objective_file(ctx, spec, ctx.out / "objective.json")
    inputs, allocs = _sensitivity_inputs(ctx)
    ctx.stage(_objective_stage(kept), t0, "1 thread (setup, small)")
    strict_from, no_self_dot = getattr(args, "strict_from", None), getattr(args, "no_self_dot", False)
    if strict_from is not None and not no_self_dot:   # refuse a receipt of another operator before the card
        bind_strict(ctx, strict_from)
    split = getattr(ctx, "split", None)
    dump = (ctx.rank_list("dump-grad", [(a, o, 0) for f in inputs for a, o in f["copies"]], "alloc") if split
            else ",".join(str(a) for a in allocs))
    line = gradient_line(ctx, args)
    words = gradient_line_words(ctx, line, args, ctx.out) if line else None
    ctx.check_words = check_line_words(words)
    extra = ["--objective", ctx.rank_json(obj_path, owner_only=True) if split else str(obj_path),
             "--dump-grad", dump] + (words["extra"] if words else [])
    session = getattr(ctx, "session", None)
    program = session.program(line) if session is not None and words else None
    if program is not None:   # the session's program: attached to, or started here and left loaded (ad_session)
        try:
            doc = ctx.window_loaded(program, extra, "g", words, line)
        except NotLoaded as error:
            print(f"woof hex sensitivity: {error}: no session is kept", file=sys.stderr, flush=True)
            doc = ctx.window_loaded_fallback(error.doc, extra, words, line)
    else:
        doc = ctx.window(extra, extra_env=words["env"] if words else None,
                         replace_args=words["replace"] if words else None, unchecked_forward=line)
    strict = strict_check(ctx, no_self_dot, strict_from)
    ctx.receipt_extra = _speed_blocks(ctx, words, kept)
    if session is not None:
        ctx.receipt_extra["session"] = session.block()
    return _sensitivity_answer(ctx, args, objective, inputs, doc, strict)


def _objective_folder(k: int, spec: dict) -> str:
    name = "".join(c if c.isalnum() or c in "-_" else "-" for c in str(spec.get("name") or ""))[:40].strip("-")
    return f"o{k:02d}" + (f"-{name}" if name else "")


def _sensitivity_many(ctx: "Context", args: argparse.Namespace, specs: list[dict]) -> int:
    """sensitivity --objectives FILE: every objective of the file in one command, each with its own folder
    (o01, o02, ...: objective.json, sensitivity.nc, its map and receipt.json, its seconds in stages), on ONE loaded
    program where the row has a gradient line (LoadedProgram), else one program each, one after another. The answer
    check runs once (in o01's folder, or its kept verdict is taken) and every objective's receipt names it. The call's
    own receipt.json lists the objectives, the loaded program and the call's stages."""
    started, t_call = time.time(), time.monotonic()
    top = ctx.out
    line = gradient_line(ctx, args)
    words = gradient_line_words(ctx, line, args, top) if line else None
    ctx.check_words = check_line_words(words)
    loaded = bool(line and line.get("loaded_program"))
    strict_from, no_self_dot = getattr(args, "strict_from", None), getattr(args, "no_self_dot", False)
    if strict_from is not None and not no_self_dot:
        bind_strict(ctx, strict_from)
    inputs, allocs = _sensitivity_inputs(ctx)
    split = getattr(ctx, "split", None)
    call_stages, jobs = [], []
    for k, spec in enumerate(specs, 1):
        t0 = time.monotonic()
        ctx.out, ctx.stages = top / _objective_folder(k, spec), []
        ctx.out.mkdir(parents=True, exist_ok=True)
        objective, path, kept = objective_file(ctx, spec, ctx.out / "objective.json")
        dump = (ctx.rank_list("dump-grad", [(a, o, 0) for f in inputs for a, o in f["copies"]], "alloc") if split
                else ",".join(str(a) for a in allocs))
        extra = ["--objective", ctx.rank_json(path, owner_only=True) if split else str(path), "--dump-grad", dump]
        ctx.stage(_objective_stage(kept), t0, "1 thread (setup, small)")
        jobs.append({"k": k, "spec": spec, "out": ctx.out, "objective": objective, "kept": kept, "extra": extra,
                     "stages": ctx.stages, "windows": []})
    call_stages.append({"stage": f"objectives ({len(jobs)}: "
                                 f"{sum(1 for j in jobs if j['kept']['how'] == 'kept')} kept, "
                                 f"{sum(1 for j in jobs if j['kept']['how'] == 'made')} made)",
                        "wall_s": round(time.monotonic() - t_call, 3), "parallelism": "1 thread (setup)"})
    program, program_block, fallback = None, None, None
    if loaded:
        lp = line["loaded_program"]
        session = getattr(ctx, "session", None)   # the session's program where this command keeps one, else its own
        program = ((session.program(line) if session is not None else None)
                   or LoadedProgram(top / "program", lp.get("idle_s", 120), lp.get("stop_wait_s", 120)))
    t_win = time.monotonic()
    try:
        for job in jobs:
            ctx.out, ctx.stages, ctx.windows = job["out"], job["stages"], []
            extra = job["extra"] + (words["extra"] if words else [])
            if program is not None and fallback is None:
                try:
                    job["doc"] = ctx.window_loaded(program, extra, f"o{job['k']:02d}", words, line)
                except NotLoaded as error:
                    fallback = str(error)
                    print(f"woof hex sensitivity: {error}: the other objectives run one program each",
                          file=sys.stderr, flush=True)
                    job["doc"] = ctx.window_loaded_fallback(error.doc, extra, words, line)
            else:
                job["doc"] = ctx.window(extra, extra_env=words["env"] if words else None,
                                        replace_args=words["replace"] if words else None, unchecked_forward=line)
            job["windows"] = ctx.windows
    finally:
        if program is not None:
            program_block = program.stop()
    call_stages.append({"stage": f"gradients ({len(jobs)} objectives, "
                                 + ("one loaded program" if program is not None and fallback is None
                                    else "one program each") + ")",
                        "wall_s": round(time.monotonic() - t_win, 3), "parallelism": "GPU (runner)"})
    # the answer check once, in the first objective's folder (its windows, or the verdict kept on this operator)
    t0 = time.monotonic()
    first = jobs[0]
    ctx.out, ctx.stages, ctx.windows = first["out"], first["stages"], first["windows"]
    strict = strict_check(ctx, no_self_dot, strict_from)
    first["windows"] = ctx.windows
    call_stages.append({"stage": "answer check (" + ((strict or {}).get("how") or "none") + ")",
                        "wall_s": round(time.monotonic() - t0, 3), "parallelism": "GPU (runner)"})
    t0 = time.monotonic()
    rows, codes = [], []
    for job in jobs:
        ctx.out, ctx.stages, ctx.windows = job["out"], job["stages"], job["windows"]
        ctx.started = time.time() - sum(s["wall_s"] for s in ctx.stages)
        check = strict
        if job is not first and strict is not None and strict.get("how") != "not run":
            check = {**{k: v for k, v in strict.items() if k not in ("windows", "kept_as")}, "how": "same call",
                     "windows": [], "same_call": {"receipt": str(first["out"] / "receipt.json"),
                                                  "how_there": strict.get("how")}}
        ctx.receipt_extra = {**_speed_blocks(ctx, words, job["kept"]),
                             "call": {"objective": job["k"], "of": len(jobs), "receipt": str(top / "receipt.json")},
                             "loaded_program": (job["doc"].get("loaded_program") if program is not None
                                                else {"on": False, "why": ctx.gradient_line_off})}
        code, info = _sensitivity_answer(ctx, args, job["objective"], inputs, job["doc"], check)
        codes.append(code)
        rec = info["rec"]
        rows.append({"objective": job["k"], "folder": str(job["out"]), "name": job["objective"]["name"],
                     "metric": job["objective"]["metric"], "receipt": str(job["out"] / "receipt.json"),
                     "passed": rec.get("passed"), "valid": rec.get("valid"), "objective_how": job["kept"]["how"],
                     "window_wall_s": round(job["doc"].get("door_wall_s") or 0.0, 1),
                     "loaded_program": job["doc"].get("loaded_program")})
    call_stages.append({"stage": f"answers ({len(jobs)}: sensitivity.nc, map, receipt)",
                        "wall_s": round(time.monotonic() - t0, 3), "parallelism": "1 thread + GPU remap"})
    ctx.out = top
    call = {"schema": CALL_SCHEMA, "door": "sensitivity", "row": ctx.row_name, "steps": ctx.steps, "argv": sys.argv,
            "objectives": rows, "first_gradient": getattr(ctx, "first_gradient", None),
            "gradient_line": words["block"] if words else {"on": False, "why": ctx.gradient_line_off},
            "loaded_program": program_block or {"on": False, "why": "the row has no gradient line" if not line
                                                else "not asked"},
            "loaded_program_fallback": fallback,
            "session": ctx.session.block() if getattr(ctx, "session", None) is not None else None,
            "strict_check": {"how": (strict or {}).get("how"), "pass": (strict or {}).get("pass"),
                             "receipt": str(first["out"] / "receipt.json")},
            "stages": call_stages, "wall_s": round(time.time() - started, 1),
            "passed": all(c == 0 for c in codes)}
    _write_json(top / "receipt.json", call)
    how = ("one loaded program" if program is not None and fallback is None else "one program each")
    print(f"sensitivity: {len(jobs)} objectives on {how}: "
          + ", ".join(f"objective {r['objective']} window {r['window_wall_s']:.1f} s" for r in rows))
    print(f"wrote {top / 'receipt.json'} (the call), wall {call['wall_s']} s")
    return 0 if call["passed"] else 1


def _sensitivity_answer(ctx: "Context", args: argparse.Namespace, objective: dict, inputs: list[dict], doc: dict,
                        strict: dict | None) -> tuple[int, dict]:
    """The answer of one objective's window: the fields, sensitivity.nc, the map, the receipt and the lines."""
    import numpy as np
    split = getattr(ctx, "split", None)
    t0 = time.monotonic()
    valid = None if args.no_map else _valid_time(ctx.fmap, 0, ctx.row["dt_s"])
    ctx.stage("valid time (the plan's kept facts)", t0, "host, one kept read")
    t0 = time.monotonic()
    win = ctx.out / "window"
    # one card: every copy's gradient summed; split: every rank's copies, halo included, summed onto the global mesh
    if split:   # the ranks' copies summed back into the one-card allocations, then read as on one card
        ctx.gather_report = ctx.gather_rank_grads(win, inputs)
    grads = {f["name"]: _sum_copies(win, f) for f in inputs}
    sw = None if split else _start_winds(ctx)
    wind_term = {"folded": False, "why": ("split window: every rank rebuilds the cell winds from u at the start of "
                                          "step 1" if split else "the plan keeps no start cell winds")}
    if sw is not None and "u" in grads:   # dJ/du += R^T dJ/d(start cell winds): the forecast made them as R u
        term = _startwinds.term(sw, ctx.plan, win, grads["u"].shape[1])
        grads["u"] = grads["u"] + term
        wind_term = {"folded": True, "allocations": _startwinds.dump_allocations(sw),
                     "first_rebuild": sw["first_rebuild"], "step1_reads_before_rebuild": sw["step1_reads_before_rebuild"],
                     "term_max_abs": float(np.max(np.abs(term))),
                     "term_max_over_folded_dJ_du_max": float(np.max(np.abs(term)) / (np.max(np.abs(grads["u"])) or 1.0))}
    state = PlanState(ctx.plan)
    g_theta, g_qv = theta_qv_gradient(ctx.table, ctx.fmap, state, grads)
    units = objective["units"]
    out_fields = {"dJ_dtheta": (g_theta, f"{units} K-1", "dJ/dtheta at the initial time (qv, density, wind fixed)",
                                "cell"),
                  "dJ_dqv": (g_qv, f"{units} (kg/kg)-1", "dJ/dqv at the initial time (theta, density, wind fixed)",
                             "cell")}
    for f in inputs:
        out_fields[f"dJ_d{f['name']}_stored"] = (grads[f["name"]], f"{units} per {f['units']}",
                                                f"dJ/d{f['name']} of the stored initial state, every copy summed, "
                                                "the other stored fields held"
                                                + (" (the physics' start cell winds move with u, as the forecast "
                                                   "makes them)" if f["name"] == "u" and wind_term["folded"] else ""),
                                                f["grid"])
    nc = ctx.out / "sensitivity.nc"
    write_mesh_fields(nc, out_fields, {"title": "hex adjoint sensitivity of a forecast metric to the initial state",
                                       "metric": objective["name"], "row": ctx.row_name, "steps": ctx.steps,
                                       "plan_sha256": ctx.fmap.doc["plan_sha256"], **ctx.gradient_nc_attrs()},
                      coords=mesh_coordinates(ctx.geometry))
    ctx.stage("sensitivity.nc (write-back transpose, Rust NetCDF)", t0, "1 thread (host, 151k words)")
    drawn = map_choice(objective, g_theta, g_qv, units, args.map_level)
    level = drawn["level"]
    peak_level = int(np.argmax(np.abs(g_theta).max(axis=1))) + 1
    t0 = time.monotonic()
    rendered = None
    if not args.no_map:
        rendered = render_cell_map(ctx.out, ctx.geometry, drawn["values"][level - 1], variable=drawn["variable"],
                                   title=f"Sensitivity of {objective['metric']} at "
                                         f"+{window_length_words(ctx.steps * ctx.row['dt_s'])} "
                                         f"to {drawn['start']}, model level {level}",
                                   units=drawn["units"], valid=valid)
        ctx.stage("map (GPU remap + rw_wrfbatch)", t0, "GPU remap, Rust renderer")
    j_obj = (doc.get("objective") or {})
    summary = {f["name"]: {"max_abs": float(np.max(np.abs(grads[f["name"]]))), "units": f"{units} per {f['units']}"}
               for f in inputs}
    nonfinite = int(sum((~np.isfinite(v)).sum() for v in grads.values()))
    result = {"objective": objective["name"], "metric": objective["metric"], "sensitivity_nc": str(nc),
              "map": rendered, "map_level": level, "map_field": drawn["variable"], "map_says": drawn["said"],
              "dJ_dtheta_peak_level": peak_level,
              "dJ_dtheta_max_abs": float(np.max(np.abs(g_theta))), "dJ_dqv_max_abs": float(np.max(np.abs(g_qv))),
              "stored_fields": summary, "nonfinite_words": nonfinite, "start_cell_wind_term": wind_term,
              "reverse_wall_s": j_obj.get("reverse_wall_s"), "window_receipt": str(win / "window.json"),
              "fct_frozen": doc.get("fct_frozen")}
    rec = ctx.receipt(result, nonfinite == 0 and (strict is None or strict["pass"] or strict.get("how") == "not run"),
                      strict)
    good = rec["passed"]
    print(f"sensitivity: {objective['name']}")
    print(f"  {ctx.steps} steps ({window_length_words(ctx.steps * ctx.row['dt_s'])}) of {ctx.row_name}; reverse "
          f"{j_obj.get('reverse_wall_s', float('nan')):.1f} s")
    print(f"  max |dJ/dtheta| {result['dJ_dtheta_max_abs']:.4g} {units}/K at model level {peak_level} "
          f"({drawn['said']})   "
          f"max |dJ/dqv| {result['dJ_dqv_max_abs']:.4g}")
    if rendered:
        print(f"  map {rendered['images'][0]}")
        if rendered["cells_above_scale"]:
            print(f"  map scale tops at {rendered['scale_top']:.3g} ({rendered['scale_rule']}); "
                  f"{rendered['cells_above_scale']} cells above it at this level, peak {rendered['field_peak']:.3g}")
    print_strict(strict)
    print(f"  valid {rec['valid']}")
    print(f"wrote {nc} and {ctx.out / 'receipt.json'}")
    print(used_line(rec))
    for f in (ctx.out / "window").glob("grad-*.f32"):
        f.unlink()
    if split:
        for f in (ctx.out / "window").glob("r*/grad-*.f32"):
            f.unlink()
    return (0 if good else 1), {"out_fields": out_fields, "objective": objective, "rec": rec, "units": units}


def _sensitivity_carried(args: argparse.Namespace, route: dict) -> int:
    """sensitivity --gradient-mesh NAME: the same command on the companion mesh (its own folder made-on-NAME, its own
    receipt and answer check), then the cell fields of its sensitivity carried to the native mesh (the table's method
    and map), written as this door's sensitivity.nc and receipt.json, each saying it was carried."""
    import numpy as np
    started = time.time()
    comp, nat, made = route["companion"], route["native"], route["coarse"]
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f"woof hex sensitivity: the gradient is made on the {route['name']} companion mesh {made['mesh']} "
          f"({made['cells']:,} cells, {made['steps']} steps of {made['dt_s']:g} s) and carried to mesh {nat['mesh']} "
          f"({nat['cells']:,} cells, {nat['steps']} steps of {nat['dt_s']:g} s); {route['how']}", flush=True)
    route["coarse_args"].session = "off"   # the companion mesh's program ends with its command (no session there yet)
    ctx = Context(route["coarse_args"], "sensitivity")
    ctx.gradient_block_override = {
        "asked": route["asked"], "how": route["how"], "carried": False,
        "made_on": {"mesh": ctx.row.get("mesh"), "row": ctx.row_name}, "carried_to": None,
        "method": "none in this folder: this is the window on the companion mesh",
        "answer_check": "made on this operator (the companion mesh's)",
        "note": f"the window on the {route['name']} companion mesh of mesh {nat['mesh']}; the door's answer, carried "
                f"to that mesh, is {out / 'sensitivity.nc'} with its receipt {out / 'receipt.json'}"}
    code, info = _sensitivity(ctx, route["coarse_args"])
    t0 = time.monotonic()
    geometry = mesh_geometry(Path(nat["grid"]), nat["cells"], 0)
    inner = _gm.boundary_inner(Path(nat["grid"]), nat["cells"]) if comp.get("zero_boundary_zone", True) else None
    grids = set(comp.get("carry_grids") or ["cell"])
    carried, left = {}, {}
    try:
        for name, (data, units, what, grid) in info["out_fields"].items():
            if grid in grids:
                carried[name] = (_gm.carry_gradient(route["map"], data, ctx.geometry["areaCell"], geometry["areaCell"],
                                                    inner), units,
                                 f"CARRIED from the {route['name']} companion mesh {made['mesh']}: {what}", grid)
            else:
                left[name] = f"an {grid} field: carrying it between the meshes was not shown"
    except _gm.GradientMeshError as error:
        raise AdDoorError(f"--gradient-mesh {route['name']}: {error}") from error
    block = carried_block(route, ctx)
    block["fields_carried"], block["fields_not_carried"] = sorted(carried), left
    block["boundary_zone"] = ("set to zero on the native mesh (bdyMaskCell > 0), as the route was shown"
                              if inner is not None else "carried like the interior")
    objective, rec = info["objective"], info["rec"]
    nc = out / "sensitivity.nc"
    write_mesh_fields(nc, carried, {
        "title": f"hex adjoint sensitivity of a forecast metric to the initial state, MADE ON the {route['name']} "
                 f"companion mesh {made['mesh']} and CARRIED to mesh {nat['mesh']}",
        "metric": objective["name"], "row": ctx.row_name, "native_mesh": nat["mesh"], "steps": ctx.steps,
        "plan_sha256": ctx.fmap.doc["plan_sha256"], "gradient_mesh_asked": route["asked"],
        "gradient_made_on_mesh": made["mesh"], "gradient_carried_to_mesh": nat["mesh"],
        "gradient_carried": f"yes: made on mesh {made['mesh']}, carried to mesh {nat['mesh']}",
        "gradient_carry_method": comp["method"], "gradient_map_sha256": route["map"]["sha256"],
        "gradient_answer_check": block["answer_check"], "gradient_shown_on": block["shown_on"],
        "gradient_fields_not_carried": ", ".join(sorted(left)) or "none"},
        coords=mesh_coordinates(geometry, {g for (_, _, _, g) in carried.values()}))
    carry_stage = {"stage": f"carry to mesh {nat['mesh']} (map {route['map']['nnz']:,} weights, write NetCDF)",
                   "wall_s": round(time.monotonic() - t0, 3), "parallelism": "1 thread (host, sparse map)"}
    g_theta = carried["dJ_dtheta"][0]
    drawn = map_choice(objective, g_theta, carried["dJ_dqv"][0], info["units"], args.map_level)
    level = drawn["level"]
    peak_level = int(np.argmax(np.abs(g_theta).max(axis=1))) + 1
    rendered, stages = None, [carry_stage]
    if not args.no_map:
        t0 = time.monotonic()
        valid = _valid_time(ctx.fmap, 0, ctx.row["dt_s"])   # the plan's kept facts: no parse of plan.json
        rendered = render_cell_map(out, geometry, drawn["values"][level - 1], variable=drawn["variable"],
                                   title=f"Sensitivity of {objective['metric']} at +{window_length_words(nat['window_s'])} to "
                                         f"{drawn['start']}, model level {level}; made on the {route['name']} mesh, "
                                         "carried",
                                   units=drawn["units"], valid=valid)
        stages.append({"stage": "map (GPU remap + rw_wrfbatch)", "wall_s": round(time.monotonic() - t0, 3),
                       "parallelism": "GPU remap, Rust renderer"})
    nonfinite = int(sum((~np.isfinite(v[0])).sum() for v in carried.values()))
    top = json.loads(json.dumps(rec, default=str))
    top["gradient_mesh"] = block
    top["receipt_of_the_window"] = str(ctx.out / "receipt.json")
    top["stages"] = list(top.get("stages") or []) + stages
    top["wall_s"] = round(time.time() - started, 1)
    top["result"] = {**top["result"], "sensitivity_nc": str(nc),
                     "made_on_sensitivity_nc": rec["result"]["sensitivity_nc"], "map": rendered, "map_level": level,
                     "map_field": drawn["variable"], "map_says": drawn["said"],
                     "dJ_dtheta_peak_level": peak_level, "dJ_dtheta_max_abs": float(np.max(np.abs(g_theta))),
                     "dJ_dqv_max_abs": float(np.max(np.abs(carried["dJ_dqv"][0]))),
                     "carried_nonfinite_words": nonfinite, "carried": True}
    top["valid_rule"] = ((top.get("valid_rule") or "every window valid") + "; a gradient made on a companion mesh "
                         "and carried: valid is that of the operator it was made on (the companion mesh's), and the "
                         "carried field has no check of its own")
    if top.get("passed") is not None:
        top["passed"] = bool(top["passed"]) and nonfinite == 0
    _write_json(out / "receipt.json", top)
    print(f"  CARRIED: made on the {route['name']} companion mesh {made['mesh']} ({made['cells']:,} cells, "
          f"{ctx.steps} steps of {ctx.row['dt_s']:g} s) and carried to mesh {nat['mesh']} ({nat['cells']:,} cells) "
          f"by {comp.get('method_short') or comp['method']}")
    print(f"  the answer check was made on the {route['name']} operator, not on the native one; shown on: "
          f"{block['shown_on']}")
    print(f"  on mesh {nat['mesh']}: max |dJ/dtheta| {top['result']['dJ_dtheta_max_abs']:.4g} {info['units']}/K at "
          f"model level {peak_level}; not carried: {', '.join(sorted(left)) or 'none'}")
    if rendered:
        print(f"  map {rendered['images'][0]}")
    print(f"wrote {nc} and {out / 'receipt.json'} (the window on the companion mesh: {ctx.out})")
    return 0 if code == 0 and top.get("passed", True) else 1


def param_bundle_check(bundle: Path, pin: dict | None = None) -> dict:
    """The parameter-carrier bundle against its own SHA256SUMS, and that file's sha256 against the row's pin.

    THE BREAKAGE THIS PREVENTS: the bundle is the row's scheme bundle with four images replaced; a copy with one plain
    image left in place (or a bundle of another build chain) still loads, and its gradient would be printed under the
    name of the bundle that was proved. A bundle with no SHA256SUMS (as hex 0.5.0's pinned x1.2562 row had; 0.6.0
    ships no such row) is named as unpinned."""
    sums = bundle / "SHA256SUMS"
    if not sums.is_file():
        if pin:
            raise AdDoorError(f"parameter-carrier bundle {bundle}: no SHA256SUMS; this row pins bundle "
                              f"{pin.get('name')} (SHA256SUMS sha256 {pin.get('sha256sums_sha256')})")
        return {"path": str(bundle), "sha256sums_sha256": None, "files": None, "pinned": None}
    wrong, n = [], 0
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        want, rel = line.split(None, 1)
        rel = rel.strip().lstrip("*")
        n += 1
        if not (bundle / rel).is_file() or _sha256_file(bundle / rel) != want:
            wrong.append(rel)
    if wrong:
        raise AdDoorError(f"parameter-carrier bundle {bundle}: {len(wrong)} of {n} files are missing or differ from "
                          f"its SHA256SUMS ({', '.join(wrong[:6])})")
    sha = _sha256_file(sums)
    if pin and pin.get("sha256sums_sha256") and pin["sha256sums_sha256"] != sha:
        raise AdDoorError(f"parameter-carrier bundle {bundle}: its SHA256SUMS is {sha}, and this row pins bundle "
                          f"{pin.get('name')} ({pin['sha256sums_sha256']}): another bundle's gradient is not this "
                          "row's")
    return {"path": str(bundle), "name": (pin or {}).get("name"), "sha256sums_sha256": sha, "files": n,
            "pinned": bool(pin)}


def run_param_grad(args: argparse.Namespace) -> int:
    plant_request(args, load_table(), "param-grad")   # a planting word in the environment: refused before any download
    route = gradient_route(args, "param-grad")   # a coarse mesh is refused by name here
    say_storm_measures(load_table(), args, "param-grad", [getattr(args, "metric", None)])
    ctx = Context(args, "param-grad")
    ctx.gradient_route = route
    import struct
    from .ad import params as _params
    t0 = time.monotonic()
    bundle = (param_bundle_check(ctx.assets["bundle_params"], ctx.row.get("param_bundle"))
              if ctx.assets.get("bundle_params") is not None and Path(ctx.assets["bundle_params"]).is_dir() else None)
    objective = metric_objective(ctx.table, ctx.fmap, ctx.geometry, args.metric, args.levels, args.box, args.circle,
                                 ctx.row)
    obj_path = ctx.out / "objective.json"
    _write_json(obj_path, objective)
    if args.params:
        params_path = Path(args.params)
    elif ctx.row.get("param_names"):   # the settings the row's carrier bundle carries, at the shipped values
        params_path = ctx.out / "params.json"
        _params.write_carrier_document(params_path, ctx.row["param_names"])
    else:
        params_path = ctx.assets["params"]
    params = json.loads(params_path.read_text(encoding="utf-8"))
    # a table that moves a setting off its shipped value: the forward then runs at that value, and no recording
    # exists to check it against (the runner needs --params-unchecked 1); the receipt says so and is not "valid"
    moved = sorted(p["name"] for p in params["params"] if p["name"] in _params.BY_NAME
                   and int(str(p["p0_bits"]), 16) != int(_params.BY_NAME[p["name"]].bits, 16))
    ctx.stage("inputs (objective, parameter table)", t0, "1 thread (setup, small)")
    extra = ["--objective", str(obj_path), "--params", str(params_path)]
    if moved:
        extra += ["--params-unchecked", "1"]
        strict = {"how": "not run", "pass": False, "moved": moved,
                  "what": f"moved point: the forward runs with {', '.join(moved)} off the shipped value, so it is "
                          "not the recorded forecast and nothing checks it; this receipt's validity is not checked"}
    else:
        strict = strict_check(ctx, getattr(args, "no_self_dot", False))
    if args.check:
        n = len(params["params"])
        scale = ctx.table["doors"]["param-grad"].get("param_tl_scale", 1e-12)
        doc = ctx.window(extra + ["--param-tl", ",".join(repr(float(scale)) for _ in range(n))],
                         door_args_key="check_args")
    else:
        doc = ctx.window(extra)
    rows = []
    p0 = {p["name"]: struct.unpack("<f", struct.pack("<I", int(p["p0_bits"], 16)))[0] for p in params["params"]}
    what = {p["name"]: p.get("what", "") for p in params["params"]}
    for g in doc.get("param_gradients") or []:
        name = g.get("name")
        rows.append({"name": name, "family": g.get("family"), "p0": g.get("p0", p0.get(name)), "dJ_dlnp": g.get("dJdlnp"),
                     "dJ_dp": g.get("dJdp"), "nonzero_columns": g.get("nonzero_columns"), "what": what.get(name, "")})
    pdot = doc.get("param_dot")
    passed = bool(rows) and all(r["dJ_dlnp"] is not None and math.isfinite(r["dJ_dlnp"]) for r in rows)
    if args.check:
        passed = passed and bool(pdot) and pdot.get("rel", 1.0) <= ctx.table["thresholds"]["param_dot_rel"]
    values = doc.get("objective_values") or []
    j_value = values[0].get("J") if values and isinstance(values[0], dict) else None
    result = {"objective": objective["name"], "J": j_value, "params_table": str(params_path), "moved": moved,
              "param_bundle": bundle, "operator_what": (ctx.row.get("param_bundle") or {}).get("instead_of_row"),
              "gradients": rows, "param_dot": pdot,
              "window_receipt": str(ctx.out / "window" / "window.json"), "fct_frozen": doc.get("fct_frozen")}
    rec = ctx.receipt(result, passed, strict)
    passed = rec["passed"]
    units = objective["units"]
    print(f"param-grad: {objective['name']}")
    print(f"  {ctx.steps} steps ({window_length_words(ctx.steps * ctx.row['dt_s'])}) of {ctx.row_name}")
    if j_value is not None:
        print(f"  J = {j_value!r} {units}" + (f"   (moved point: {', '.join(moved)} off the shipped value)" if moved else ""))
    if bundle:
        print(f"  parameter-carrier bundle {bundle.get('name') or ''} {bundle['path']} (SHA256SUMS sha256 "
              f"{bundle['sha256sums_sha256']}, {bundle['files']} files equal)")
    if result["operator_what"]:
        print(f"  operator: {result['operator_what']}")
    refused = ctx._refused_names()
    print(f"  refused by name (identity both ways): {len(refused)} launch kinds: {', '.join(refused) or 'none'}")
    if moved:
        print("  whole-state dot  not run (moved point): this receipt's validity is not checked")
    else:
        print_strict(strict)
    print(f"  {'constant':<26} {'p0':>12} {'dJ/dln p':>14} {'dJ/dp':>14}")
    for r in rows:
        print(f"  {r['name']:<26} {r['p0']:>12.5g} {r['dJ_dlnp']:>14.5g} {r['dJ_dp']:>14.5g}   ({units})")
    if pdot:
        print(f"  parameter dot: tangent {pdot.get('tl_dJ'):.6g} vs adjoint {pdot.get('adjoint_sum_c_dJdlnp'):.6g}, "
              f"rel {pdot.get('rel'):.3e}  {'PASS' if pdot.get('pass') else 'FAIL'}")
    print(f"{'PASS' if passed else 'FAIL'}  (receipt {ctx.out / 'receipt.json'})")
    print(used_line(rec))
    return 0 if passed else 1


# -------------------------------------------------------------------------------------------------------- parsers
def _common(parser: argparse.ArgumentParser, steps_help: str) -> None:
    parser.add_argument("--inner-model", default="full", choices=INNER_MODELS,
                        help="the linear model the windows run: full (default) is the row's own recorded derivative; "
                             "reduced is that derivative with pieces left out by name (--inner-model-keep); simple is "
                             "the purpose-built linear model a variational system's inner loop runs (exact dynamics; "
                             "the boundary layer as a linear diffusion with MYNN's own diffusivities from the "
                             "trajectory held; cloud, radiation, drag, land and sea ice held at the trajectory's "
                             "tendencies). reduced and simple are an inner loop's linear model, not the derivative of "
                             "the forecast")
    parser.add_argument("--inner-model-keep", default=None, metavar="PIECES",
                        help="with --inner-model reduced: the pieces kept, a comma list of "
                             + ", ".join(INNER_MODEL_PIECES) + " (default: boundary-layer; the dynamics and the "
                             "surface layer are always kept; every piece not listed is held at the trajectory's "
                             "tendencies)")
    parser.add_argument("--row", default=None, help="operator row of the door table or the row cache (default: the "
                                                     "table's default row)")
    made = parser.add_argument_group("a row for any mesh (made on demand and cached; --init records its plan)")
    made.add_argument("--mesh", default=None,
                      help="mesh name (found in WOOF_HEX_AD_MESH_PATH or the hex mesh cache), grid file, or directory")
    made.add_argument("--suite", default="hrrrv5", help="physics suite of the made row (default hrrrv5)")
    made.add_argument("--oro", type=Path, default=None,
                      help="GSL-drag orography file (default: <mesh>.ugwp_oro_data.nc beside the grid)")
    made.add_argument("--dt", type=float, default=None,
                      help="timestep in s (default: hex's rule, 6 s per km of nominalMinDc dividing the radiation step)")
    parser.add_argument("--need-mib", type=int, default=None,
                        help="free card memory to wait for before the window (default: the row's need)")
    parser.add_argument("--need-host-mib", type=int, default=None,
                        help="host memory the window needs (default: the row's measured need; a row with none is not "
                             "checked). With less available the door stops before any download, exit code "
                             f"{HOST_TOO_SMALL_EXIT}")
    src = parser.add_argument_group("plan package (the recorded forecast the window replays)")
    src.add_argument("--plan", type=Path, default=None, help="a recorded plan package directory")
    src.add_argument("--init", type=Path, default=None,
                     help="record a plan package from this init first (needs the recorder); not supported in WOOF 1.1.7: use --row's example plans")
    src.add_argument("--record-route", choices=("derivative", "forecast"), default="derivative",
                     help="with --init: the forecast's kernel forms the recording runs (derivative, the default: the "
                          "forms this release has derivative rows for; forecast: the forecast's own defaults, the "
                          "same bits). A plan with a kernel kind that has no derivative is refused when it is sealed")
    src.add_argument("--cumulus", choices=CUMULUS_SCHEMES, default=None,
                     help="with --init: record the forecast with this cumulus scheme on (config_convection_scheme; "
                          "default: the suite's own deck for the mesh, which for hrrrv5 away from NOAA's 12 km rows "
                          "runs none). The window holds the scheme's derivative by name: the linear model keeps the "
                          "forward run's cumulus tendency")
    parser.add_argument("--tempo-image", choices=("shipped",) + tuple(TEMPO_IMAGES), default="shipped",
                        help="the taped TEMPO derivative image (default shipped: the row's own). f08 adds the PBL "
                             "subgrid-radii path NOAA's hrrrv5 deck runs at every radiation step (bl held as "
                             "constants), so a window can cross a radiation step")
    src.add_argument("--case", type=Path, default=None,
                     help="for a limited-area mesh: the regional door's work directory (woof hex regional ... "
                          "--stage mesh..lbc), or a LAM_* case file; gives the recording its static file, boundaries, "
                          "start time and contract deck (default: the work directory --init lies in)")
    parser.add_argument("--physics-mode", default=None, choices=("exact", "correct"),
                        help="the forecast the tangent and adjoint are of (default exact: every plan is recorded in "
                             "exact mode and the derivative rows are of that arithmetic). `woof hex forecast` runs "
                             "correct mode by default; this build has no derivative rows of it, and correct is "
                             "refused with that sentence before anything is downloaded")
    parser.add_argument("--steps", type=int, default=None, help=steps_help)
    parser.add_argument("--out", type=Path, required=True, help="output directory (receipt.json and the door's files)")
    parser.add_argument("--card", type=int, default=None,
                        help="GPU index (default: the CUDA_VISIBLE_DEVICES a scheduler set, else 0)")
    parser.add_argument("--cards", default=None,
                        help="GPU indices A,B,... for a row whose window runs on several cards (a mesh above the "
                             "one-card size); the first card holds the dycore. On a row with a rank split "
                             "(adjoint-test, sensitivity) N cards run the window as N ranks, one on each card; EXPERIMENTAL: a "
                             "split gradient is not yet shown equal to the one-card gradient, and the door says so "
                             "before it runs and in the receipt")
    parser.add_argument("--rank-plans", type=Path, default=None,
                        help="with --cards on a row with a rank split: the rank plan package PREFIX (PREFIX-r<k>, each "
                             "with its rank-gather.json) split from the one-card plan given with --plan (default: the "
                             "table's package under the asset root)")
    parser.add_argument("--record-ranks", type=int, default=1, metavar="N",
                        help="with --init: record the forecast as N rank threads on the one card and cut the sealed "
                             "plan into N rank plans, OUT/plan-r0 .. OUT/plan-r<N-1>, the plans a window split over N "
                             "cards replays (a mesh above the one-card window classes, such as 12 km CONUS, is admitted "
                             "by its cells per rank). The door stops after the split and prints the plans. Default 1: "
                             "the one-card recording, unchanged")
    parser.add_argument("--wait-card", type=float, default=0.0,
                        help="seconds to wait for the row's free memory on the card (default: refuse at once)")
    parser.add_argument("--runner", type=Path, default=None, help="window runner binary (default: the installed one)")
    parser.add_argument("--assets", type=Path, default=None,
                        help="asset root of the operator rows (default: the installed runner package's)")
    parser.add_argument("--grid", type=Path, default=None, help="mesh grid file (default: the row's)")
    g = parser.add_argument_group("the mesh the gradient is taken on")
    g.add_argument("--gradient-mesh", default=GRADIENT_MESH_DEFAULT,
                   help="native: on the mesh itself; NAME (such as 6km): on the table's coarser companion mesh of "
                        "the same region, carried back to this mesh (sensitivity only; the receipt and the files say "
                        "so); auto (default): this mesh's own default in the table")
    g.add_argument("--gradient-plan", type=Path, default=None,
                   help="the companion mesh's plan package (a recording of the same start) in place of the table's")
    from . import ad_radiation_seed as _radseed
    parser.add_argument("--radiation-seed", choices=_radseed.CHOICES, default=None, help=_radseed.HELP)


def _perturbation_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("initial perturbation (a potential-temperature bump)")
    g.add_argument("--at", default=None, help="bump centre LAT,LON in degrees (default 20.4,-26.2 on a globe; on a "
                   "limited-area mesh that does not reach it, the cell nearest the mesh's centre)")
    g.add_argument("--level", type=int, default=32, help="1-based model level of the bump centre (default 32)")
    g.add_argument("--amplitude", type=float, default=None, help="peak in K (default 1)")
    g.add_argument("--radius-km", type=float, default=None, help="Gaussian radius in km (default 1200)")
    g.add_argument("--level-width", type=float, default=None, help="Gaussian width in levels (default 4)")


def _metric_args(parser: argparse.ArgumentParser, default_region: str) -> None:
    g = parser.add_argument_group("forecast metric J (a mean over a region and levels at the window end)")
    g.add_argument("--metric", default="theta_m", help="metric of the table (default theta_m): " +
                   ", ".join(sorted(load_table()["metrics"])))
    g.add_argument("--box", default=None, help="region S,N,W,E in degrees")
    g.add_argument("--circle", default=None, help=f"region LAT,LON,RADIUS_KM (default: {default_region})")
    g.add_argument("--levels", default=None, help="1-based model levels, 'a-b' or 'a,b' (default: every level)")


def _strict_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("strict check (the whole-state dot that makes the receipt valid)")
    g.add_argument("--strict-from", type=Path, default=None,
                   help="bind the strict check of a passing adjoint-test receipt.json of the same plan, row, runner, "
                        "steps, kernel set and refuse list instead of running it (default: run the self-dot and its "
                        "NaN-poison proof windows here)")
    g.add_argument("--no-self-dot", action="store_true",
                   help="skip the strict check: the receipt's valid reads null (not checked)")
    g.add_argument("--self-dot", action="store_true",
                   help="run the strict check here even when an earlier command on this plan (same runner, kernel "
                        "set, card family, steps and row) passed it and its verdict is kept (default: that verdict "
                        "is taken and the check is not run again; the first command on a plan always runs it)")
    g.add_argument("--strict", dest="self_dot", action="store_true",
                   help="the same as --self-dot: run the answer check on this card, even where a verdict was kept "
                        "here or shipped with the packed example (made by the publisher on its card family)")
    g.add_argument("--seed", type=int, default=None,
                   help="test direction of the whole-state dot (the runner's --seeds; default: the table's, 1). One "
                        "direction proves little near the bar: run a second with --seed 2")
    _dot_bar_arg(g)


def _dot_bar_arg(parser: Any) -> None:
    parser.add_argument("--dot-bar", default=None, choices=_dot_bar.SETTINGS,
                        help="the whole-state dot's bar: roundings (default: within 4 single-precision roundings of "
                             "the dot, and with 8 or more directions no one-signed drift) or strict (1e-07 relative, "
                             "0.84 of one rounding); WOOF_HEX_DOT_BAR sets it too")


def refuse_strict_words(args: argparse.Namespace) -> None:
    """--self-dot asks for the strict check here; --no-self-dot and --strict-from ask for it not to run here."""
    if getattr(args, "self_dot", False):
        if getattr(args, "no_self_dot", False):
            raise AdDoorError("--self-dot and --no-self-dot are opposites: give one")
        if getattr(args, "strict_from", None) is not None:
            raise AdDoorError("--self-dot and --strict-from are alternatives: run the check here, or bind a receipt's")


def self_dot_args_seeded(args: list, seed: int | None, directions: int = 1) -> list:
    """The self-dot window's runner words with the test direction set to ``seed`` (None keeps the table's), or with
    ``directions`` consecutive test directions from it (--dot-directions N: the runner's --seeds list, one tangent and
    one reverse each, in one window).
    THE BREAKAGE THIS PREVENTS: a door's validity could be checked at one direction only (seed 1); a four-direction
    table reads 2.54e-07 at seed 3 on the same 120 km operator that reads 8.1e-09 at seed 1."""
    if seed is None and directions <= 1:
        return list(args)
    if seed is not None and seed < 1:
        raise AdDoorError(f"--seed {seed}: the runner's test directions are numbered from 1")
    if directions < 1:
        raise AdDoorError(f"--dot-directions {directions}: give 1 or more")
    out = list(args)
    if "--seeds" not in out:
        raise AdDoorError("--seed: the table's self_dot_args name no --seeds word to set")
    first = seed if seed is not None else int(str(out[out.index("--seeds") + 1]).split(",")[0])
    out[out.index("--seeds") + 1] = ",".join(str(first + i) for i in range(directions))
    return out


def _exit_codes(handler):
    """The door's handler with its own exit code for a host that is too small (HOST_TOO_SMALL_EXIT); every other
    refusal keeps the command line's code 2."""
    def run(args: argparse.Namespace) -> int:
        try:
            return handler(args)
        except HostMemoryError as error:
            print(f"woof hex: {error}", file=sys.stderr)
            return error.exit_code
    run.__name__, run.__doc__, run.__wrapped__ = handler.__name__, handler.__doc__, handler
    return run


def _route_arg(p: argparse.ArgumentParser, door: str) -> None:
    """--boundary-layer on a door that runs the row's route (ROUTE_DOORS) besides sensitivity."""
    p.add_argument("--boundary-layer", default=None, choices=("taped", "tape-free", "guarded", "floored"),
                   help=f"on a row that names boundary-layer routes: the linear model {door} runs, as sensitivity's "
                        "--boundary-layer (default: the row's default route, the one a plain sensitivity runs: "
                        "tape-free on the 3 km storm row and its 12 km cut; taped is the exact route of the held "
                        "boundary layer)")


def add_ad_parsers(commands: argparse._SubParsersAction) -> None:
    t = commands.add_parser("tangent", help="tangent-linear steps of a row from a perturbed initial state; writes the "
                                            "tangent state at the window end")
    _common(t, "model steps (default: the row's)")
    _perturbation_args(t)
    t.add_argument("--fields", default=None, help="comma list of output fields (default: the table's)")
    _strict_args(t)
    _route_arg(t, "tangent")
    t.set_defaults(handler=_exit_codes(run_tangent))

    a = commands.add_parser("adjoint-test", help="dot, finite-difference and Taylor checks of the tangent and the "
                                                 "adjoint against the recorded thresholds (PASS/FAIL)")
    _common(a, "model steps (default: the row's; on the 120 km rows this command runs 1 step unless told)")
    _perturbation_args(a)
    _metric_args(a, "a circle 1.25 bump radii around the bump")
    a.add_argument("--taylor", default=None, help="comma list of Taylor step sizes h (default: the table's)")
    a.add_argument("--directions", default=None,
                   help="comma list of the table's test directions, or all (default: the row's list): "
                        + ", ".join(load_table()["doors"]["adjoint-test"]["directions"]["order"])
                        + ". The bump always runs; each further direction is one more window. A PASS needs at "
                          "least two directions resolved, each inside the bar")
    a.add_argument("--no-self-dot", action="store_true",
                   help="skip the whole-state dot window (the objective-direction dot, FD and Taylor checks only)")
    a.add_argument("--seed", type=int, default=None,
                   help="test direction of the whole-state dot (the runner's --seeds; default: the table's, 1). One "
                        "direction proves little near the bar: run a second with --seed 2")
    a.add_argument("--dot-directions", type=int, default=None, metavar="N",
                   help="run the whole-state dot on N test directions (from --seed, default 1) in one window and "
                        "judge them together: each within the bar, and with 8 or more no one-signed drift (the mean "
                        "within 2 standard errors of zero). The release check runs 8. (--directions names the "
                        "gradient test's directions, a different list)")
    _dot_bar_arg(a)
    _route_arg(a, "adjoint-test")
    a.add_argument("--plant-error", default=None, metavar="KERNEL:SCALE",
                   help="break the derivative ON PURPOSE to watch the transpose check catch it (default off): in the "
                        "backward (adjoint) code of KERNEL only, each change is scaled by SCALE (1.01: a 1 per cent "
                        "error); the forward model and the tangent are untouched. KERNEL: "
                        + ", ".join(load_table()["doors"]["adjoint-test"]["plant_error"]["kernels"])
                        + ". The verdict line says PLANTED, the receipt is never valid, and the exit status is 4 when "
                          "the check caught the error, 5 when it did not")
    a.set_defaults(handler=_exit_codes(run_adjoint_test))

    s = commands.add_parser("sensitivity", help="adjoint of a scalar forecast metric to the initial state; writes "
                                                "the sensitivity fields and a map")
    _common(s, "model steps (default: the row's)")
    _metric_args(s, "every cell")
    s.add_argument("--map-level", type=int, default=None,
                   help="1-based model level of the map (default: the level that carries most of the drawn field: "
                        "dJ/dqv for a qv measure, dJ/dtheta for any other)")
    s.add_argument("--no-map", action="store_true", help="write the fields only")
    s.add_argument("--plant-error", default=None, help=argparse.SUPPRESS)   # refused in one sentence (plant_request)
    fast = s.add_argument_group("what a gradient costs (the table's gradient line and first-gradient routes)")
    fast.add_argument("--objectives", type=Path, default=None,
                      help="a JSON list of objectives ({metric, levels, box, circle, name}; a word an entry leaves "
                           "out is this command's): every one in this command, each in its own folder oNN under "
                           "--out with its own sensitivity.nc, map and receipt; on a row with a gradient line they "
                           "run on ONE loaded program (the second and later objectives skip the program's start)")
    fast.add_argument("--checked-forward", action="store_true",
                      help="run the gradient window with every step boundary of the forward compared with the "
                           "recording, as the answer check's windows do (slower; one program per objective; default: "
                           "the row's gradient line where it has one)")
    fast.add_argument("--boundary-layer", default=None, choices=("taped", "tape-free", "guarded", "floored"),
                      help="on a row with a gradient line: the boundary layer's backward without its tape "
                           "(tape-free: faster; from this build a different linear model from the exact route, which "
                           "differentiates the turbulence closure, the plume's transport and the boundary-layer "
                           "height that the exact route holds, so its answer differs by design and is not "
                           "byte-equal; the size of the difference was not measured on this build) or from its tape (taped: the exact route, for windows up to about 30 "
                           "minutes at 3 km, measured on the example; a longer window runs with a one-line "
                           "warning), or, on the 3 km storm row, from its tape with the boundary layer's mixing "
                           "differentiated under a weak-shear guard (guarded: for a near-ground wind question over "
                           "a short window; slower; the command says what was shown and what was not); default: "
                           "the row's (tape-free on the 3 km storm row)")
    fast.add_argument("--keep-forward", default=None, choices=("on", "off"),
                      help="on a row and route that name it (the 3 km storm row's default route): a further target "
                           "in the same command and a later command that attaches take the forward of the gradient "
                           "before, when their start is the same, and go straight to the backward sweep; the "
                           "loaded program holds the forward's checkpoints in host memory between gradients (about "
                           "91,000 MiB at 59 steps; the command counts it). off runs every forward. Default: on "
                           "where the route names it")
    cds = sorted({n for r in load_table()["rows"].values()
                  for n in (((r.get("gradient_line") or {}).get("cloud_derivative") or {}).get("routes") or {})})
    fast.add_argument("--cloud-derivative", default=None, choices=cds or ("shipped",),
                      help="on a row whose gradient line names the option (the 3 km storm row): the cloud scheme's "
                           "derivative with rain and droplet number following the forecast (number), the same "
                           "form with both numbers held (gated), or as the earlier builds of this release ran it "
                           "(shipped); default: the row's (number on the 3 km storm row)")
    fg = load_table()["doors"]["sensitivity"].get("first_gradient") or {}
    fast.add_argument("--first-gradient", default=None, choices=sorted(fg.get("routes") or {"recorded": {}}),
                      help="how a first gradient of a new forecast is made: recorded (record the forecast first, "
                           "then every gradient on it replays the recording) or direct (no recording first; not in "
                           "this build: the door says why and where it pays); default: "
                           f"{fg.get('default', 'recorded')}")
    _strict_args(s)
    from . import ad_session as _session
    _session.add_session_args(s)
    s.set_defaults(handler=_exit_codes(run_sensitivity))

    p = commands.add_parser("param-grad", help="gradient of a forecast metric to the physics constants of the row's "
                                               "parameter table; prints the table")
    _common(p, "model steps (default: the row's)")
    _metric_args(p, "every cell")
    p.add_argument("--params", type=Path, default=None, help="parameter table JSON (default: the row's)")
    p.add_argument("--check", action="store_true",
                   help="also run the parameter tangent and report the parameter dot test (tangent vs adjoint)")
    p.add_argument("--no-self-dot", action="store_true",
                   help="skip the strict whole-state check of the parameter-carrier operator: the receipt's valid "
                        "reads null (not checked)")
    p.add_argument("--self-dot", action="store_true",
                   help="run the strict check here even when an earlier param-grad on this plan and operator passed "
                        "it and its verdict is kept (default: that verdict is taken and not run again)")
    p.add_argument("--strict", dest="self_dot", action="store_true",
                   help="the same as --self-dot: run the answer check on this card")
    p.add_argument("--seed", type=int, default=None,
                   help="test direction of the whole-state dot (the runner's --seeds; default: the table's, 1)")
    _dot_bar_arg(p)
    p.set_defaults(handler=_exit_codes(run_param_grad))

