"""``woof hex global --members``: a global ensemble, resident on the card.

    woof hex global --row global-120km --source gefs --members 1-30 --cycle latest --work-dir RUN

WHAT WAS MISSING (B5, the 2026-10-16 global operations gap list): the global
door ran one state.  An ensemble of global members meant one door command per
member, each paying its own process, imports, mesh bind, kernel load and
device-stack construction (measured on the 120 km row, warm kernel cache:
22.5 s of the 32 s forecast door), and one init after another.

WHAT THIS DOES, composing doors that exist:

  init      every member's ``init-source --member N`` runs as its own process,
            ``--init-jobs`` at a time (downloads and the Rust init overlap);
  forecast  members run in GROUPS RESIDENT ON ONE CARD through the forecast
            door's own roster (``forecast --members``, :mod:`woof.hex.multi_member`):
            one process binds the mesh and loads the kernels once, every member
            of the group keeps private state, a private stream and private
            history, and the card interleaves their kernels.  Groups run back to
            back on each card, one queue per card over ``--devices`` (default:
            the visible card).  The group size is ``--members-per-card``, or
            the smaller of the co-tenancy table's limit for this mesh and card
            (``device_admission.CO_TENANCY_ROWS``) and how many members the
            card's free memory holds, priced as the roster admission prices
            them (full solo floor plus stochastic arrays per member); a group
            the card still refuses before construction is split in two and retried.
            Groups on one card are pipelined (:mod:`woof.hex.card_handoff`): the
            next group's process does its imports, init reads and host
            preparation while the current one integrates, and builds on the card
            the moment the current one releases it, before that one's host-only tail.
  the rest  products, export and score run per member on its own history.  Each member's maps are
            drawn hour by hour while its group integrates (a first look per member, first_look.py),
            and its full products pass then draws only what the live draws left.

Each member's history lands in ``RUN/ensemble/gNN/memNNN/`` and the plan
(``global-plan.json``) lists it; ``ensemble.json`` maps every member to its
init, group, card, forecast folder and outcome.

A member resident with others writes the same bytes it writes alone: the
roster gives each member private module state (the forecast door's own
resident-member proof); the one-member command, ``--member N``, is the
reference this is checked against.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Mapping, Sequence

from .card_handoff import GATE_ENV, TICKET_ENV
from .errors import MpasPortError

SCHEMA = "gpuwm-hex.global-ensemble/v1"
REFUSED_GROUP = "co-tenancy refuses"
#: Members' live draws (products processes) that run at once on the host; auto is one per 8 cores.
LIVE_DRAWS = os.environ.get("WOOF_HEX_ENSEMBLE_LIVE_DRAWS", "auto")


class EnsembleDoorError(MpasPortError):
    """The ensemble path refuses, and says why."""


# -- graceful recovery (WOOF 2026-10-26, run_2de68ba40c93de79: one member's products refused at f15 and the door threw
# away the nine good members). Each per-member stage is retried once; a member that still fails is recorded in
# ensemble.json "missing" with its own reason and the ensemble goes on with the others. The door refuses only when
# no member is left. WOOF_FAULT_PLAN (test-only; the WOOF worker passes it for injected runs) fails named units.
FAULT_ENV = "WOOF_FAULT_PLAN"
MEMBER_RETRIES = 1
_FAULT_PLAN = None
_FAULT_TEXT = {
    "exit": "FAULT INJECTION: exited 1",
    "transient": "FAULT INJECTION: Connection reset by peer",
    "nonfinite": "FAULT INJECTION: history q2 is not finite at f{hour:03d}",
    "hang": "FAULT INJECTION: no progress (stall)",
    "http500": "FAULT INJECTION: HTTP 500",
    "gpu": "FAULT INJECTION: CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered (Xid 13)",
}


def member_number(run_id) -> int | None:
    digits = "".join(ch for ch in str(run_id)[-4:] if ch.isdigit())
    return int(digits) if digits else None


def fault_point(unit: str, run_id, hour=None) -> str | None:
    """The injected failure for this unit, or None. Each plan item fails its unit ``times`` times."""
    global _FAULT_PLAN
    if _FAULT_PLAN is None:
        try:
            plan = json.loads(os.environ.get(FAULT_ENV) or "[]")
        except ValueError:
            plan = []
        _FAULT_PLAN = [item for item in (plan if isinstance(plan, list) else []) if isinstance(item, dict)]
    number = member_number(run_id)
    for item in _FAULT_PLAN:
        want = item.get("member")
        if item.get("unit") != unit or int(item.get("times") or 0) <= 0:
            continue
        if want not in (None, "*") and str(want) != str(number):
            continue
        item["times"] = int(item["times"]) - 1
        at = item.get("hour") if isinstance(item.get("hour"), int) else (hour or 0)
        return _FAULT_TEXT.get(str(item.get("kind") or "exit"), _FAULT_TEXT["exit"]).format(hour=at)
    return None


def failure_reason(text: str) -> str:
    """A stage's own reason: its last line that refuses, errs or fails, else its last line."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        low = line.lower()
        if any(word in low for word in ("refused", "error", "not finite", "failed", "fault injection", "exited")):
            return line[:400]
    return lines[-1][:400] if lines else ""


def _log_reason(path) -> str:
    try:
        return failure_reason(Path(path).read_text(encoding="utf-8", errors="replace")[-20000:])
    except (OSError, TypeError):
        return ""


def _captured_door(argv, env, log: Path) -> int:
    """_run_door with the stage's output (its subprocesses' too) captured at the file-descriptor level into ``log``,
    then echoed to the door's own output, so a failed member stage has its own reason."""
    from .global_door import _run_door

    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_bytes(b"")
    sys.stdout.flush()
    sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    py = sys.stdout, sys.stderr
    try:
        # Both the descriptors (subprocesses, native code) and Python's own streams append to the one log.
        with open(log, "ab") as stream, open(log, "a", encoding="utf-8", buffering=1) as text:
            os.dup2(stream.fileno(), 1)
            os.dup2(stream.fileno(), 2)
            sys.stdout = sys.stderr = text
            try:
                return _run_door(argv, env)
            except Exception as error:  # a stage that raises is a failed stage with that reason
                print(f"{argv[0]} failed: {type(error).__name__}: {error}", flush=True)
                return 1
            finally:
                text.flush()
                sys.stdout, sys.stderr = py
    finally:
        sys.stdout, sys.stderr = py
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        os.close(saved[0])
        os.close(saved[1])
        try:
            sys.stdout.write(log.read_text(encoding="utf-8", errors="replace")[-200000:])
            sys.stdout.flush()
        except OSError:
            pass


def history_hours(folder) -> int:
    """History frames a member's forecast folder holds (its hours made, the start included)."""
    try:
        return sum(1 for _ in Path(folder).rglob("*history*.nc"))
    except (OSError, TypeError):
        return 0


def forecast_devices(requested: str | None, environment=None, error=EnsembleDoorError) -> list[str]:
    """Resolve logical device selections inside the inherited CUDA allocation.

    Child processes receive physical identifiers, since setting their CUDA
    visibility to a logical index would discard the parent's pool allocation.
    """
    environment = os.environ if environment is None else environment
    mask = environment.get("CUDA_VISIBLE_DEVICES")
    visible = None if mask is None else [item.strip() for item in mask.split(",")]
    if visible is not None and (not all(visible) or "-1" in visible or len(set(visible)) != len(visible)):
        raise error("CUDA_VISIBLE_DEVICES exposes no valid unique GPU allocation")
    selected = [item.strip() for item in requested.split(",")] if requested else visible or ["0"]
    cards = []
    for item in selected:
        if visible is None or not requested:
            card = item
        elif item.isdecimal() and int(item) < len(visible):
            card = visible[int(item)]
        elif item.startswith(("GPU-", "MIG-")) and item in visible:
            card = item
        else:
            raise error(f"--devices {item!r} is outside CUDA_VISIBLE_DEVICES={mask!r}; "
                        "launching it would escape the assigned GPU allocation")
        if not card or card == "-1" or card in cards:
            raise error("--devices must select valid unique GPUs within the assigned allocation")
        cards.append(card)
    return cards


def validate_member_bindings(members, error=EnsembleDoorError):
    """Shared inputs require distinct model realizations, not duplicate controls."""
    bindings = {}
    for member in members:
        p = member["provenance"]
        key = (p["source"], p["source_cycle"], p["source_member"], p["lead_hours"])
        bindings.setdefault(key, []).append(member)
    for group in bindings.values():
        if sum(not (m["stochastic"]["sppt"] or m["stochastic"]["skeb"]) for m in group) > 1:
            raise error("repeated source input needs independent SPPT/SKEB members; multiple unperturbed "
                        "members would repeat one forecast and understate spread")


def parse_members(spec: str) -> list[int]:
    """``1-30``, ``0,1,2`` or a mix; every member once, in order."""

    found: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            low, high = (int(x) for x in part.split("-", 1))
            if high < low:
                raise EnsembleDoorError(f"--members range {part!r} runs backwards")
            found.extend(range(low, high + 1))
        else:
            found.append(int(part))
    if not found or len(set(found)) != len(found):
        raise EnsembleDoorError(f"--members {spec!r} must name each member once")
    return found


def member_plan(value: Mapping[str, Any], members: Sequence[int], lags=None) -> list[dict[str, Any]]:
    """Source cycles and leads are distinct inputs at one common valid time."""
    from .init_sources import Cycle, load_table, resolve_source
    from .lagged_members import expand_lags, forecast_window

    work = Path(value["paths"]["work"])
    base = list(value["commands"]["init"])
    source = resolve_source(load_table(), value["source"], value.get("accepted_licences", ()))
    selected = members if source.get("members") else [None]
    expanded = expand_lags(source, Cycle.parse(value["cycle"]), selected, lags or [0])
    if not source.get("members"):
        # Generated identities name independent stochastic model realizations,
        # never provider members which the deterministic row does not publish.
        expanded = [{**p, "generated_member": number} for p in expanded for number in members]
    rows = []
    for index, provenance in enumerate(expanded, 1):
        number = provenance["source_member"]
        identity = provenance.get("generated_member", number)
        run_id = f"mem{identity:03d}" if lags is None else f"mem{index:03d}"
        init_dir = work / "members" / run_id / "init"
        if not source.get("members") and len(members) > 1:
            stamp = Cycle.parse(provenance["source_cycle"]).when.strftime("%Y%m%d%H")
            init_dir = work / "inits" / f"{stamp}-f{provenance['lead_hours']:03d}"
        if value.get("vertical", {}).get("explicit_binding"):
            init_dir /= value["vertical"]["sha256"][:12]
        argv = list(base)
        argv[argv.index("--out-dir") + 1] = str(init_dir)
        argv[argv.index("--cycle") + 1] = provenance["source_cycle"]
        if number is not None:
            argv += ["--member", str(number)]
        if provenance["lead_hours"]:
            argv += ["--lead-hours", str(provenance["lead_hours"])]
        rows.append({"member": number, "run_id": run_id, "init_dir": str(init_dir),
                     "init": str(init_dir / "init.nc"), "init_argv": argv, "provenance": provenance,
                     **forecast_window(provenance, value.get("hours", 24))})
    return rows


#: Run in a child process, so the planning process never holds a CUDA context on a card its groups need.
#: It reads what the forecast door's roster admission reads: memGetInfo with a context open, and the
#: card shape (SM count, resident threads) the footprint row is priced from.
_CARD_PROBE = r"""
import json, cupy
r = cupy.cuda.runtime
cards = []
for d in range(r.getDeviceCount()):
    with cupy.cuda.Device(d):
        free, total = r.memGetInfo()
    p = r.getDeviceProperties(d)
    name = p.get("name", b"")
    name = name.decode("utf-8", "replace") if isinstance(name, (bytes, bytearray)) else str(name)
    cards.append({"name": name, "free": int(free), "total": int(total), "sms": int(p["multiProcessorCount"]),
                  "threads": int(r.deviceGetAttribute(r.cudaDevAttrMaxThreadsPerMultiProcessor, d))})
print(json.dumps(cards))
"""


def _probe_cards(devices: str | None) -> list[dict[str, Any]]:
    env = dict(os.environ)
    if devices:
        env["CUDA_VISIBLE_DEVICES"] = devices
    try:
        out = subprocess.run([sys.executable, "-c", _CARD_PROBE], capture_output=True, text=True,
                             timeout=120, env=env)
        cards = json.loads(out.stdout.strip().splitlines()[-1]) if out.returncode == 0 else []
    except (OSError, subprocess.SubprocessError, ValueError, IndexError, TypeError, AttributeError):
        return []  # no readable card here: the forecast door admits at bind
    return cards if isinstance(cards, list) else []


def _forecast_flag(argv: Sequence[str], flag: str) -> str | None:
    for at, token in enumerate(argv):
        if token == flag and at + 1 < len(argv):
            return argv[at + 1]
        if token.startswith(flag + "="):
            return token.split("=", 1)[1]
    return None


def memory_group_limit(value: Mapping[str, Any], args: argparse.Namespace) -> int | None:
    """Members per group the smallest visible card holds, priced exactly as the roster admission prices them.

    Same sum as the roster admissions (``multi_member._run_members``, ``multi_member.admit_suite_group``): each
    member's solo floor on the route it runs (``device_admission.route_required_free_bytes`` on this card's
    footprint row and the run's headroom; the MPAS route never below its measured whole runs) plus its
    stochastic arrays (``forecast_door.stochastic_array_bytes`` of the member's own options), against the
    card's free bytes, through ``device_admission.co_tenancy_limits``.  None when no card can be read here
    (the forecast door still admits, and a refused group still splits).
    """

    from . import forecast_door as door
    from .device_admission import (card_profile_from_attributes, co_tenancy_limits, model_for_card,
                                   route_required_free_bytes)
    from .multi_member import argv_physics_route

    cards = _probe_cards(getattr(args, "devices", None))
    if not cards:
        return None
    argv = list(value["commands"]["forecast"])
    # Each member is priced on the route it runs: the MPAS route (the hex default) holds far more than the
    # ArWen-route rows (U-H4: 30 km MPAS members planned 2 per RTX PRO 6000 at the ArWen price and ran out of memory).
    route = argv_physics_route(argv)
    headroom = _forecast_flag(argv, "--headroom-mib")
    margin = None if headroom is None else int(round(float(headroom) * door.MIB))
    cells = int(value["mesh"]["cells"])
    levels = int(value.get("vertical", {}).get("n_vert_levels", 55))
    additional = [door.stochastic_array_bytes(cells, member.get("stochastic"), levels=levels)
                  for member in value.get("members") or []]
    limits = []
    for card in cards:
        profile = card_profile_from_attributes(card["name"], {"MultiProcessorCount": card["sms"],
                                                              "MaxThreadsPerMultiProcessor": card["threads"]})
        model = model_for_card(profile, "global", rollback_armed="--stop-on-refusal" in argv)
        table, memory = co_tenancy_limits(
            mesh_kind="global", cells=cells, levels=levels, card_bytes=int(card["total"]),
            free_bytes=int(card["free"]),
            solo_required_bytes=route_required_free_bytes(cells, model, margin, route=route),
            member_additional_bytes=additional)
        limits.append(min(table, memory))
    return max(1, min(limits))


def group_limit(value: Mapping[str, Any], args: argparse.Namespace) -> int:
    """``--members-per-card``, else the smaller of the co-tenancy table's limit and what the card's memory holds.

    THE BREAKAGE THIS PREVENTS (RTX 5090, 60 km GEFS members 1-2, 2026-10-20): the table alone gave 3
    per 32 GB card, the roster admission charged 19.65 GB per member (solo floor + stochastic arrays) and
    refused every first group, which was split and rerun.
    """

    if getattr(args, "members_per_card", None) not in (None, "auto"):
        limit = int(args.members_per_card)
        if not 1 <= limit <= 4:
            raise EnsembleDoorError("--members-per-card is 1 to 4: the forecast door's roster holds 1 to 4 members")
        return limit
    from .device_admission import CO_TENANCY_ROWS
    from .global_door import _visible_cards

    memory = _visible_cards()
    if not memory:
        return 1
    rows = [row for row in CO_TENANCY_ROWS if row.mesh_kind == "global"
            and int(value["mesh"]["cells"]) <= row.maximum_cells
            and int(value.get("vertical", {}).get("n_vert_levels", 55)) <= row.maximum_levels
            and min(memory) >= row.minimum_card_bytes]
    table = max((row.maximum_members for row in rows), default=1)
    if table == 1:
        return 1
    priced = memory_group_limit(value, args)
    return table if priced is None else max(1, min(table, priced))


#: Extra columns one rank of a member split over several cards holds beyond its share of the mesh (its halo
#: rings and the seam packets), as a fraction of that share.  Priced high on purpose until the measured rows
#: below replace it: a rank that does not fit fails at its first step, a rank priced a little high only costs
#: a card.
RANK_HALO_SHARE = 0.10


def cards_per_member(value: Mapping[str, Any], args: argparse.Namespace,
                     cards: list[dict[str, Any]] | None = None) -> int:
    """Cards one member runs on: ``--cards-per-member``, else the fewest cards whose per-rank price fits the
    smallest visible card.

    THE BREAKAGE THIS PREVENTS (8x RTX 5090 fit, 2026-10-27): a 30 km MPAS-route member peaks at 75,069 MiB
    on one card (``device_admission.MPAS_ROUTE_RUN_PEAKS``), more than twice a 32 GB card, and the ensemble
    door only ever put a member on ONE card (``forecast_argv`` strips ``--samehost-devices``), so a 30 km
    global ensemble (and the synthetic-radar DA cycle's member forecasts, which run through this door) could
    not run on any 32 GB card at all.  The forecast door already runs one member over several cards in one
    process (the same-host rank threads, ``mpasphys.samehost_v842`` on the MPAS route); this picks how many.

    The price of one rank is the member's own route price (``device_admission.route_required_free_bytes``, the
    MPAS route never below its measured whole runs) at the rank's share of the columns plus
    :data:`RANK_HALO_SHARE`, against the card's TOTAL memory: the card count is a property of the shape and
    the card model, not of what another lease holds on the box at the moment (the forecast door's own
    admission still checks the free memory when it binds).  1 whenever a member fits one card.
    """

    requested = getattr(args, "cards_per_member", None)
    resident = getattr(args, "members_per_card", None)
    if requested in (None, "", "auto") and resident not in (None, "", "auto") and int(resident) > 1:
        return 1  # several members resident on ONE card were asked for: a member fits one card
    if cards is None:
        cards = _probe_cards(getattr(args, "devices", None))
    visible = max(1, len(cards)) if cards else len(forecast_devices(getattr(args, "devices", None)))
    if requested not in (None, "", "auto"):
        count = int(requested)
        if not 1 <= count <= visible:
            raise EnsembleDoorError(f"--cards-per-member is 1 to the {visible} cards this door can use, got {count}")
        return count
    if not cards:
        return 1
    from . import forecast_door as door
    from .device_admission import card_profile_from_attributes, model_for_card, route_required_free_bytes
    from .multi_member import argv_physics_route

    argv = list(value["commands"]["forecast"])
    route = argv_physics_route(argv)
    headroom = _forecast_flag(argv, "--headroom-mib")
    margin = None if headroom is None else int(round(float(headroom) * door.MIB))
    cells = int(value["mesh"]["cells"])
    smallest = min(cards, key=lambda card: int(card["total"]))
    profile = card_profile_from_attributes(smallest["name"], {"MultiProcessorCount": smallest["sms"],
                                                               "MaxThreadsPerMultiProcessor": smallest["threads"]})
    model = model_for_card(profile, "global", rollback_armed="--stop-on-refusal" in argv)
    total = int(smallest["total"])
    if route_required_free_bytes(cells, model, margin, route=route) <= total:
        return 1
    for count in range(2, len(cards) + 1):
        share = -(-int(cells * (1.0 + RANK_HALO_SHARE)) // count)
        if route_required_free_bytes(share, model, margin, route=route) <= total:
            return count
    # Not even every visible card holds it: one card each, and the forecast door's own admission refuses by name.
    return 1


def card_slots(cards: Sequence[str], per_member: int) -> list[str]:
    """The card sets members run on: consecutive runs of ``per_member`` cards (``"0,1,2"``), one queue each.
    Cards left over after the last whole set run nothing (named in the forecast stage's record)."""

    if per_member <= 1:
        return list(cards)
    return [",".join(cards[at:at + per_member]) for at in range(0, len(cards) - per_member + 1, per_member)]


def _set_flag(argv: list[str], flag: str, value: str) -> None:
    if flag in argv:
        argv[argv.index(flag) + 1] = value
    else:
        argv += [flag, value]


def member_forecast_argv(value: Mapping[str, Any], member: Mapping[str, Any], out: Path, scratch: Path,
                         ranks: int) -> list[str]:
    """One member alone, over ``ranks`` cards (the forecast door's same-host ranks), with exactly the arguments its
    roster record would give it (``multi_member.roster_members``): init, start, length, history start, source
    label and the member's own stochastic identity, so the member is the same member it is on one card."""

    argv = list(value["commands"]["forecast"])
    if "--samehost-devices" in argv:
        at = argv.index("--samehost-devices")
        del argv[at:at + 2]
    _set_flag(argv, "--init", str(member["init"]))
    _set_flag(argv, "--out", str(out / member["run_id"]))
    if "--icechunk-store" in argv:
        at = argv.index("--icechunk-store") + 1
        argv[at] = argv[at].rstrip("/") + "/" + member["run_id"]
    _set_flag(argv, "--scratch", str(scratch / member["run_id"]))
    _set_flag(argv, "--start-time", str(member["start_time"]))
    _set_flag(argv, "--hours", f"{float(member['hours']):g}")
    _set_flag(argv, "--history-from-minutes", f"{float(member.get('history_from_minutes') or 0):g}")
    _set_flag(argv, "--init-source", json.dumps(member["provenance"], sort_keys=True))
    _set_flag(argv, "--stochastic", json.dumps(member["stochastic"], sort_keys=True))
    return [*argv, "--samehost-devices", ",".join(str(index) for index in range(ranks))]


#: Host bytes one member's forecast process holds, per mesh cell, by route, with headroom.  Measured (hex fit/hex-5090
#: = rel/hex-0.4.9 + this branch, RTX 5090, MPAS physics convection_permitting, 2026-10-27): whole-run VmHWM 4.22 GB
#: at 40,962 cells (120 km, one rank) and 8.46 GB at 99,898 cells (multi-zoom, fixed step) = 72 KB/cell + 1.3 GB.
#: Used only to hold back the PIPELINED next process on a card set while the host lacks the memory
#: (see ``worker`` in run_forecasts).
HOST_BYTES_PER_CELL = {"mpas": 80 * 1024, "arwen": 60 * 1024}
#: The interpreter, CUDA context and kernel images of one forecast process.
HOST_BYTES_FLOOR = 2 << 30
#: Each same-host rank past the first: 6.68 GB for 2 ranks against 4.22 GB for 1 at 120 km (measured as above).
HOST_BYTES_PER_EXTRA_RANK = 5 << 29


def member_host_bytes(value: Mapping[str, Any], ranks: int = 1) -> int:
    from .multi_member import argv_physics_route
    route = argv_physics_route(list(value["commands"]["forecast"]))
    cells = int(value["mesh"]["cells"])
    return (HOST_BYTES_FLOOR + cells * HOST_BYTES_PER_CELL.get(route, HOST_BYTES_PER_CELL["mpas"])
            + max(0, int(ranks) - 1) * HOST_BYTES_PER_EXTRA_RANK)


def _init_done(member: Mapping[str, Any], value: Mapping[str, Any]) -> bool:
    try:
        receipt = json.loads((Path(member["init_dir"]) / "init-source-receipt.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    expected = member.get("provenance", {})
    return (Path(member["init"]).is_file() and (receipt.get("init") or {}).get("path") == member["init"]
            and receipt.get("member") == member["member"] and str(receipt.get("source")) == str(value["source"])
            and receipt.get("cycle") == expected.get("source_cycle")
            and receipt.get("lead_hours", 0) == expected.get("lead_hours", 0)
            and receipt.get("valid_time", receipt.get("cycle")) == expected.get("valid_time"))


def _cli(argv: Sequence[str], log: Path, env: Mapping[str, str]) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        return subprocess.run([sys.executable, "-m", "woof.hex.cli", *argv], stdout=stream,
                              stderr=subprocess.STDOUT, env={**os.environ, **env}).returncode


def _signal_on_ticket(ticket: Path, started: threading.Event, done: threading.Event) -> None:
    """Set ``started`` once the group's process holds its card (its ticket exists) or has exited."""
    while not done.is_set() and not ticket.is_file():
        done.wait(0.25)
    started.set()


#: Bytes one prefetch read takes, and the threads one member's init is read with.
PREFETCH_CHUNK = 16 << 20
PREFETCH_THREADS = 8


def warm_page_cache(paths: Sequence[str | Path]) -> threading.Thread:
    """Read files into the page cache on several threads, in the background.

    THE COST THIS REMOVES, MEASURED (RTX 5090, global-60km, 1.5 GB GEFS init.nc, cold cache, 2026-10-20): the
    forecast's serial netCDF reads of the init spent 60.4 s of wall on 7.4 CPU-s (prep.init_vertical) and
    48.9 s on 12.2 CPU-s (prep.init_state), waiting on small random reads while the card sat at 0 %.  The
    reads stay serial netCDF (libnetcdf is not thread safe); the bytes they need are already in memory.
    """

    def read(path: Path, start: int, stop: int) -> None:
        try:
            with path.open("rb", buffering=0) as stream:
                offset = start
                while offset < stop:
                    data = os.pread(stream.fileno(), min(PREFETCH_CHUNK, stop - offset), offset)
                    if not data:
                        return
                    offset += len(data)
        except OSError:
            return  # a cache warm-up only; the reader reports any real error itself

    def run() -> None:
        jobs = []
        for name in paths:
            path = Path(name)
            try:
                size = path.stat().st_size
            except OSError:
                continue
            step = max(PREFETCH_CHUNK, -(-size // PREFETCH_THREADS))
            jobs += [(path, start, min(size, start + step)) for start in range(0, size, step)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=PREFETCH_THREADS,
                                                   thread_name_prefix="hex-prefetch") as pool:
            list(pool.map(lambda job: read(*job), jobs))

    thread = threading.Thread(target=run, name="hex-prefetch", daemon=True)
    thread.start()
    return thread


#: Peak resident bytes of one member's rw_mpas_init per mesh cell (55 levels), with headroom.
#: Measured 2026-10-16 (unified image): x1.40962 (120 km GEFS) peak RSS 1,223 MiB (30 KiB per cell; 60 km x1.163842 4,857 MiB).
INIT_BYTES_PER_CELL = 64 * 1024


def _available_host_bytes() -> int | None:
    try:
        with open("/proc/meminfo", encoding="ascii") as stream:
            for line in stream:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def init_jobs(requested: Any, value: Mapping[str, Any], todo: int) -> int:
    """Member inits built at once: ``--init-jobs N``, or ``auto`` = bounded by cores and host RAM.

    THE BREAKAGE THIS PREVENTS, MEASURED (home RTX 5090 host, unified image 2026-10-16, 6 GEFS members at
    120 km): a fixed 4 left two of six members waiting on a 24-core box.  Each rw_mpas_init runs
    multi-threaded phases (rayon) beside single-threaded ones, so ``auto`` allows one init per four
    cores, and no more than host RAM holds at INIT_BYTES_PER_CELL per mesh cell each.
    """

    if requested not in (None, "", "auto"):
        jobs = int(requested)
        if jobs < 1:
            raise EnsembleDoorError("--init-jobs is a positive integer or auto")
        return jobs
    try:
        cores = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cores = os.cpu_count() or 1
    jobs = max(1, min(todo, max(1, cores // 4)))
    available = _available_host_bytes()
    cells = int((value.get("mesh") or {}).get("cells") or 0)
    if available is not None and cells > 0:
        levels = int(value.get("vertical", {}).get("n_vert_levels", 55))
        bytes_per_cell = (INIT_BYTES_PER_CELL * max(55, levels) + 54) // 55
        jobs = max(1, min(jobs, available // (cells * bytes_per_cell)))
    return jobs


def run_inits(value: Mapping[str, Any], members: list[dict[str, Any]], args, status, env) -> dict[str, Any]:
    unique = {m["init"]: m for m in members}
    todo = [m for m in unique.values() if args.rebuild_init or not _init_done(m, value)]
    pending = {m["init"] for m in todo}
    for member in members:
        member["init_skipped"] = member["init"] not in pending
    status.stage("init", step_count=len(todo))
    if not todo:
        status.note(f"every member's init is already built for {value['source']} {value['cycle']}")
    done = 0
    failed = []
    jobs = init_jobs(getattr(args, "init_jobs", "auto"), value, len(todo))
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(_cli, m["init_argv"], Path(m["init_dir"]).parent / "init.log", env): m for m in todo}
        for future in concurrent.futures.as_completed(futures):
            member = futures[future]
            rc = future.result()
            for bound in members:
                if bound["init"] == member["init"]:
                    bound["init_returncode"] = rc
            done += 1
            status.step(done, note=f"{member['run_id']} init exited {member['init_returncode']}")
            if member["init_returncode"] != 0:
                failed.append(member["run_id"])
    retried = []
    for run_id in sorted(failed):
        member = next(m for m in todo if m["run_id"] == run_id)
        log = Path(member["init_dir"]).parent / "init.log"
        first = _log_reason(log)
        for attempt in range(2, 2 + MEMBER_RETRIES):
            status.note(f"{run_id} init retry {attempt - 1} of {MEMBER_RETRIES} after: {first}")
            began = time.monotonic()
            rc = _cli(member["init_argv"], log, env)
            retried.append({"run_id": run_id, "stage": "init", "attempt": attempt, "of": 1 + MEMBER_RETRIES, "rc": rc,
                            "reason": first, "seconds": round(time.monotonic() - began, 1), "recovered": rc == 0})
            for bound in members:
                if bound["init"] == member["init"]:
                    bound["init_returncode"] = rc
            if rc == 0:
                break
    lost = sorted({m["run_id"] for m in members if m.get("init_returncode", 0) != 0})
    for member in members:
        if member.get("init_returncode", 0) != 0:
            member["missing"] = {"stage": "init", "reason": _log_reason(Path(member["init_dir"]).parent / "init.log")
                                 or f"init exited {member['init_returncode']}", "attempts": 1 + MEMBER_RETRIES}
    if lost and len(lost) == len(members):
        raise EnsembleDoorError(
            f"the init of every member failed ({', '.join(lost)}; logs: members/<run_id>/init.log)")
    return {"built": len(todo), "skipped": len(unique) - len(todo), "jobs": jobs,
            "source_inputs": len(unique), "bound_members": len(members), "retried": retried, "missing": lost}


def member_case(value: Mapping[str, Any], member: Mapping[str, Any]) -> str:
    """The case folder a member's maps go into: the products door's own name (mesh row and start), named on both the
    first look and the full pass so the two write one tree."""
    from .global_door import runtime_mesh_name
    from .products_door import case_token
    return case_token(None, Path(value["paths"]["grid"]), str(member["start_time"]),
                      runtime_mesh_name(value["mesh"]["id"]))


def member_labels(member: Mapping[str, Any], count: int, args) -> list[str]:
    """The label flags of a member's maps. Every member draws the same titles; the credit names the member so two
    members' maps differ on the picture, not only by folder."""
    from .products_door import label_argv
    return ["--source-label", f"Hex forecast, member {member['run_id']} of {count}", *label_argv(args)]


def live_draw_gate(count: int) -> threading.Semaphore:
    """How many members' live draws run at once: WOOF_HEX_ENSEMBLE_LIVE_DRAWS, else one per 8 host cores."""
    if LIVE_DRAWS not in ("", "auto"):
        return threading.Semaphore(max(1, int(LIVE_DRAWS)))
    return threading.Semaphore(max(1, min(count, (os.cpu_count() or 4) // 8)))


def member_looks(value: Mapping[str, Any], group: Sequence[dict[str, Any]], out: Path, count: int, args,
                 gate: threading.Semaphore) -> dict[str, Any]:
    """One first look per member of a group (first_look.py), started before the group's forecast process.

    THE BREAKAGE THIS PREVENTS (a hex global-120km ensemble, 2 members, GFS 2026-10-24 00Z,
    6 h): the ensemble door ran no first look, so no member had a map until the whole forecast and every member's
    full products pass had ended; the site's live revision stayed empty and all 610 maps arrived at the end. Each
    member's look follows its own forecast folder (``<out>/<run_id>``) and draws into the tree and case of the
    member's full pass, with that pass's labels; its receipt and logs live in the member's own folder, since the
    members of a group share one forecast parent."""
    from .first_look import FirstLook
    work = Path(value["paths"]["work"])
    looks = {}
    for m in group:
        own = work / "members" / m["run_id"]
        own.mkdir(parents=True, exist_ok=True)
        looks[m["run_id"]] = FirstLook(out / m["run_id"], grid=value["paths"]["grid"], init=m["init"],
                                       start_time=str(m["start_time"]), out=own / "products",
                                       case=member_case(value, m), extra=member_labels(m, count, args),
                                       log=own / "first-look.log", receipt=own / "live-frames.json", gate=gate)
        looks[m["run_id"]].start()
    return looks


def forecast_argv(value: Mapping[str, Any], roster: Path, out: Path, scratch: Path, first_init: str) -> list[str]:
    argv = list(value["commands"]["forecast"])
    argv[argv.index("--init") + 1] = first_init
    argv[argv.index("--out") + 1] = str(out)
    argv[argv.index("--scratch") + 1] = str(scratch)
    if "--samehost-devices" in argv:  # a card per group here, never one member over several cards
        at = argv.index("--samehost-devices")
        del argv[at:at + 2]
    return [*argv, "--members", str(roster)]


def run_forecasts(value: Mapping[str, Any], members: list[dict[str, Any]], args, status, env,
                  live: bool = False, ensemble: list[dict[str, Any]] | None = None, group_size: int | None = None,
                  prefix: str = "") -> dict[str, Any]:
    """``live``: the products stage follows, so each member's maps are drawn while its group integrates.
    ``ensemble``: the whole ensemble when ``members`` is a retry of some of them (each member's stochastic identity is
    sized by the whole ensemble, so a retried member is the same member); ``group_size`` and ``prefix`` (its group
    folders) for a retry, which runs each member alone so one bad member cannot fail the others again."""
    work = Path(value["paths"]["work"])
    whole = ensemble or members
    gate = live_draw_gate(len(members)) if live else None
    devices = forecast_devices(args.devices)
    ranks = cards_per_member(value, args)
    limit = 1 if ranks > 1 else (group_size or group_limit(value, args))
    cards = card_slots(devices, ranks)
    if not cards:
        raise EnsembleDoorError(f"a member needs {ranks} cards and this door can use {len(devices)}")
    host_need = member_host_bytes(value, ranks)
    pending: "queue.Queue[list[dict[str, Any]]]" = queue.Queue()
    for start in range(0, len(members), limit):
        pending.put(members[start:start + limit])
    groups: list[dict[str, Any]] = []
    lock = threading.Lock()
    counter = [0]
    status.stage("forecast", step_count=len(members))
    finished = [0]

    def run_group(card: str, group: list[dict[str, Any]], started: threading.Event) -> None:
        try:
            _run_group(card, group, started)
        except BaseException as error:
            # A group that cannot even be launched still finishes, failed, so the card's queue drains.
            with lock:
                unfinished = [member for member in group if "forecast_returncode" not in member]
                for member in unfinished:
                    member.update(card=card, forecast_returncode=None,
                                  forecast_error=f"{type(error).__name__}: {error}")
                finished[0] += len(unfinished)
                groups.append({"members": [m["run_id"] for m in group], "card": card,
                               "error": f"{type(error).__name__}: {error}"})
        finally:
            started.set()

    def _run_group(card: str, group: list[dict[str, Any]], started: threading.Event) -> None:
        with lock:
            counter[0] += 1
            name = f"{prefix}g{counter[0]:02d}"
        warm_page_cache([m["init"] for m in group])
        out, scratch = work / "ensemble" / name, work / "scratch" / name
        roster = work / "ensemble" / f"{name}.roster.json"
        roster.parent.mkdir(parents=True, exist_ok=True)
        scratch.parent.mkdir(parents=True, exist_ok=True)
        from .stochastic import member_number, member_options
        selected = getattr(args, "stochastic", None)
        overrides = {} if selected is None else json.loads(selected)
        if "member" in overrides:
            raise EnsembleDoorError('ensemble stochastic member identity comes from the source roster')
        stochastic_seed = overrides.pop("seed", getattr(args, "stochastic_seed", 53))
        for m in group:
            identity = m["member"] if m["member"] is not None and not m["provenance"]["lag_hours"] else member_number(m["run_id"])
            m["stochastic"] = member_options(identity, seed=stochastic_seed,
                source_members=len({item["member"] for item in whole}), ensemble_size=len(whole), **overrides)
        if ranks > 1:
            # One member over the card set: the forecast door's same-host ranks (a roster cannot span cards).
            argv = member_forecast_argv(value, group[0], out, scratch, ranks)
        else:
            roster.write_text(json.dumps([
                {"run_id": m["run_id"], "init": m["init"], "stochastic": m["stochastic"],
                 **{key: m[key] for key in ("start_time", "hours", "history_from_minutes")},
                 "init_source": json.dumps(m["provenance"], sort_keys=True)}
                for m in group], indent=2) + "\n", encoding="utf-8")
            argv = forecast_argv(value, roster, out, scratch, group[0]["init"])
        log = work / "ensemble" / f"{name}.log"
        ticket = work / "ensemble" / f"{name}.card-ticket.json"
        ticket.unlink(missing_ok=True)
        began = time.monotonic()
        done = threading.Event()
        threading.Thread(target=_signal_on_ticket, args=(ticket, started, done), daemon=True).start()
        looks = member_looks(value, group, out, len(members), args, gate) if gate is not None else {}
        try:
            injected = next((text for m in group for text in [fault_point("forecast", m["run_id"])] if text), None)
            if injected:
                log.write_text(injected + "\n", encoding="utf-8")
                rc = 1
            else:
                rc = _cli(argv, log, {**env, "CUDA_VISIBLE_DEVICES": card,
                                      "WOOF_HEX_MEMBER_MEMORY": "shared-card" if len(group) > 1 else "single",
                                      GATE_ENV: str(gate_path(card)),
                                      TICKET_ENV: str(ticket)})
        finally:
            done.set()
            # The forecast has ended: each look lets a draw in progress land; what it rendered, its member's full
            # pass keeps (products --live-frames).
            for m in group:
                look = looks.get(m["run_id"])
                if look is not None:
                    m["first_look"] = look.finish()
                    receipt = look.receipt()
                    m["live_frames"] = str(receipt) if receipt else None
        row = {"group": name, "card": card, "cards_per_member": ranks, "members": [m["run_id"] for m in group],
               "returncode": rc,
               "seconds": round(time.monotonic() - began, 1), "log": str(log), "out": str(out)}
        try:
            row["card_handoff"] = json.loads(ticket.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            row["card_handoff"] = None
        text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
        if rc != 0 and REFUSED_GROUP in text and len(group) > 1:
            # Refused before construction (the roster's folders are removed): split and queue again.
            row["split"] = True
            half = len(group) // 2
            pending.put(group[:half])
            pending.put(group[half:])
            with lock:
                status.note(f"{name}: card {card} refused {len(group)} resident members; split in two")
        else:
            for member in group:
                member.update(group=name, card=card, forecast=str(out / member["run_id"]), forecast_returncode=rc,
                              forecast_log=str(log))
            with lock:
                finished[0] += len(group)
                status.step(finished[0], note=f"{name} on card {card}: {', '.join(row['members'])} exited {rc} "
                                              f"in {row['seconds']} s")
        with lock:
            groups.append(row)

    def gate_path(card: str) -> Path:
        return work / "ensemble" / f".card-{card}.gate"

    def worker(card: str) -> None:
        # Groups on one card are PIPELINED (woof.hex.card_handoff): the next group's process starts as soon
        # as this one holds the card, does its imports and host preparation while this one integrates, and
        # builds on the device only once this one has released the card.  At most one group waits.
        running: list[threading.Thread] = []
        try:
            while True:
                try:
                    group = pending.get_nowait()
                except queue.Empty:
                    with lock:
                        if finished[0] >= len(members):
                            return
                    time.sleep(0.5)  # a split group may still be queued by another card
                    continue
                # Host-RAM admission for the PIPELINED start: the next group's process prepares on the host while
                # this card's current group still holds its own host memory.  THE BREAKAGE THIS PREVENTS (V8X,
                # 8x RTX 5090, 251 GB host, 2026-10-27): hex 60 km forecast children at 21-29 GB each were the
                # cgroup OOM killer's victims 30 times in one night.  Waits only while another group of this card
                # is still running (a first group always starts) and never longer than that group runs.
                while any(item.is_alive() for item in running):
                    available = _available_host_bytes()
                    if available is None or available >= host_need:
                        break
                    time.sleep(2.0)
                started = threading.Event()
                thread = threading.Thread(target=run_group, args=(card, group, started),
                                          name=f"hex-card-{card}", daemon=True)
                thread.start()
                running.append(thread)
                started.wait()
        finally:
            for thread in running:
                thread.join()

    threads = [threading.Thread(target=worker, args=(card,), daemon=True) for card in cards]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    failed = [m["run_id"] for m in members if m.get("forecast_returncode") != 0]
    return {"members_per_group": limit, "cards_per_member": ranks, "cards": cards,
            "cards_unused": [card for card in devices if not any(card in slot.split(",") for slot in cards)],
            "host_bytes_per_member": host_need, "groups": groups, "failed": failed}


def restore_forecasts(value, members):
    """A postprocessing-only invocation uses the recorded member outputs, never init folders."""
    work = Path(value["paths"]["work"])
    try:
        prior = json.loads((work / "ensemble.json").read_text(encoding="utf-8"))
        if any(prior.get(key) != value[key] for key in ("source", "cycle")) or prior.get("row") != value["row"]["id"]:
            raise ValueError("the recorded ensemble belongs to a different row, source or cycle")
        saved = {m["run_id"]: m for m in prior["members"]}
        for member in members:
            row = saved[member["run_id"]]
            forecast = Path(row["forecast"])
            if (row["member"] != member["member"] or row.get("provenance") != member["provenance"]
                    or any(row.get(key) != member.get(key) for key in ("start_time", "hours", "history_from_minutes"))
                    or not forecast.is_dir() or not forecast.resolve().is_relative_to((work / "ensemble").resolve())):
                raise ValueError(f"no matching forecast folder for {member['run_id']}")
            member.update({key: row[key] for key in ("forecast", "group", "card", "forecast_returncode", "live_frames")
                           if key in row})
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise EnsembleDoorError(f"cannot resume member postprocessing from {work / 'ensemble.json'}: {error}; run the forecast stage first") from None


def run_ensemble(value: dict[str, Any], args: argparse.Namespace, stages: list[str], status,
                 env: Mapping[str, str]) -> dict[str, Any]:
    from .global_door import _run_door

    work = Path(value["paths"]["work"])
    members = value["members"]
    if "forecast" not in stages and any(name in stages for name in ("products", "export", "score")):
        restore_forecasts(value, members)
    record: dict[str, Any] = {"schema": SCHEMA, "row": value["row"]["id"], "source": value["source"],
                              "cycle": value["cycle"], "members": members, "stages": {}, "missing": [],
                              "recovery": []}

    def lose(member, stage, reason, **extra):
        member.setdefault("missing", {"stage": stage, "reason": reason, **extra})
        record["missing"].append({"run_id": member["run_id"], "stage": stage, "reason": reason, **extra})

    def save() -> None:
        (work / "ensemble.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    try:
        for name in stages:
            started = time.monotonic()
            if name == "init":
                record["stages"]["init"] = run_inits(value, members, args, status, env)
                record["recovery"] += record["stages"]["init"].get("retried", [])
                for member in members:
                    if member.get("missing"):
                        record["missing"].append({"run_id": member["run_id"], **member["missing"]})
            elif name == "forecast":
                ready = [m for m in members if not m.get("missing")]
                result = run_forecasts(value, ready, args, status, env, live="products" in stages, ensemble=members)
                record["stages"]["forecast"] = result
                for attempt in range(2, 2 + MEMBER_RETRIES):
                    failed = [m for m in ready if m.get("forecast_returncode") != 0]
                    if not failed:
                        break
                    reasons = {m["run_id"]: _log_reason(m.get("forecast_log")) or m.get("forecast_error")
                               or f"forecast exited {m.get('forecast_returncode')}" for m in failed}
                    for m in failed:
                        for key in ("forecast_returncode", "forecast_error", "group", "card", "forecast"):
                            m.pop(key, None)
                    status.note(f"forecast retry {attempt - 1} of {MEMBER_RETRIES}, each member alone: "
                                + ", ".join(m["run_id"] for m in failed))
                    began = time.monotonic()
                    again = run_forecasts(value, failed, args, status, env, live="products" in stages, ensemble=members,
                                          group_size=1, prefix=f"r{attempt - 1}")
                    seconds = round(time.monotonic() - began, 1)
                    record["stages"].setdefault("forecast_retries", []).append(again)
                    for m in failed:
                        record["recovery"].append({"run_id": m["run_id"], "stage": "forecast", "attempt": attempt,
                                                   "of": 1 + MEMBER_RETRIES, "rc": m.get("forecast_returncode"),
                                                   "reason": reasons[m["run_id"]], "seconds": seconds,
                                                   "recovered": m.get("forecast_returncode") == 0})
                for m in ready:
                    if m.get("forecast_returncode") != 0:
                        lose(m, "forecast", _log_reason(m.get("forecast_log")) or m.get("forecast_error")
                             or f"forecast exited {m.get('forecast_returncode')}",
                             attempts=1 + MEMBER_RETRIES, hours=history_hours(m.get("forecast")))
                result["failed"] = [m["run_id"] for m in members if m.get("missing")]
                if all(m.get("missing") for m in members):
                    raise EnsembleDoorError(
                        f"the forecast of every member failed (logs: ensemble/gNN.log): "
                        + "; ".join(f"{x['run_id']}: {x['reason']}" for x in record["missing"][:4]))
            elif name in ("products", "export", "score"):
                status.stage(name, step_count=len(members))
                for index, member in enumerate(members, 1):
                    if member.get("missing") and member["missing"].get("stage") in ("init", "forecast"):
                        status.step(index, note=f"{member['run_id']} {name} skipped: its {member['missing']['stage']} is missing")
                        continue
                    forecast = member["forecast"]
                    out = work / "members" / member["run_id"]
                    if name == "products":
                        # The case is named, so the member's first look and this pass write one tree; the hours
                        # the look rendered are kept (--live-frames) and this pass draws the rest.
                        argv = ["products", forecast, "--out", str(out / "products"),
                                "--case", member_case(value, member), *member_labels(member, len(members), args)]
                        if member.get("live_frames"):
                            argv += ["--live-frames", member["live_frames"]]
                    elif name == "export":
                        argv = ["export", forecast, "--out", str(out / "export"), "--window", "global",
                                "--field-set", args.export_field_set, "--overwrite"]
                    else:
                        config = out / "observation-scorecard.config.json"
                        config.write_text(json.dumps({**value["score_config"], "init": member["init"]}, indent=2)
                                          + "\n", encoding="utf-8")
                        argv = ["score", forecast, "--config", str(config)]
                    stage_log = out / f"{name}.log"
                    first = None
                    for attempt in range(1, 2 + MEMBER_RETRIES):
                        began = time.monotonic()
                        injected = fault_point(name, member["run_id"])
                        if injected:
                            stage_log.parent.mkdir(parents=True, exist_ok=True)
                            stage_log.write_text(injected + "\n", encoding="utf-8")
                            print(f"{member['run_id']} {name}: {injected}", flush=True)
                            rc = 1
                        else:
                            rc = _captured_door(argv, env, stage_log)
                        if name == "score" and rc == 3:
                            from .observation_scorecard import completed_scorecard
                            completed = completed_scorecard(forecast, config)
                            if completed is not None:
                                member["scorer_returncode"] = rc
                                member["score_status"] = completed["status"]
                                member["measured_metrics"] = completed["measured_metrics"]
                                status.note(completed["note"])
                                rc = 0
                        reason = _log_reason(stage_log) if rc != 0 else ""
                        if attempt > 1:
                            record["recovery"].append({"run_id": member["run_id"], "stage": name, "attempt": attempt,
                                                       "of": 1 + MEMBER_RETRIES, "rc": rc, "reason": first,
                                                       "seconds": round(time.monotonic() - began, 1),
                                                       "recovered": rc == 0})
                        if rc == 0:
                            break
                        first = first or reason or f"{name} exited {rc}"
                        status.note(f"{member['run_id']} {name} exited {rc}: {reason}"
                                    + (f"; retry {attempt} of {MEMBER_RETRIES}" if attempt <= MEMBER_RETRIES else ""))
                    member[f"{name}_returncode"] = rc
                    status.step(index, note=f"{member['run_id']} {name} exited {rc}")
                    if rc != 0:
                        # The member's other outputs stay: its forecast (and, for products, its export) is still
                        # in the ensemble's products unless its own reason says its data is bad from some hour.
                        lose(member, name, reason or f"{name} exited {rc}", attempts=1 + MEMBER_RETRIES)
                if name != "score" and all(m.get(f"{name}_returncode", 1) != 0 for m in members):
                    raise EnsembleDoorError(f"{name} failed for every member: "
                                            + "; ".join(f"{x['run_id']}: {x['reason']}" for x in record["missing"][:4]))
            else:
                continue  # mesh and static are the one-state stages, run by the caller
            record["stages"].setdefault(name, {})["seconds"] = round(time.monotonic() - started, 1)
            save()
    finally:
        save()
    return record


def add_ensemble_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lagged", default=None, help="earlier source-cycle offsets in hours, e.g. 0,6,12,18; Cartesian product with --members")
    parser.add_argument("--stochastic", default=None, metavar="JSON",
                        help="member options; default SPPT+SKEB on when fewer than 10 source members")
    parser.add_argument("--stochastic-seed", type=int, default=53)
    parser.add_argument("--members", default=None, metavar="SPEC",
                        help="source members, or generated stochastic identities for a deterministic source, e.g. 1-30: each member's init, "
                             "then the members resident on the card in groups (see --members-per-card)")
    parser.add_argument("--members-per-card", default="auto", metavar="N|auto",
                        help="resident members per group, 1 to 4 (default: the co-tenancy table's limit, capped by what the card's memory holds)")
    parser.add_argument("--cards-per-member", default="auto", metavar="N|auto",
                        help="cards one member runs on (the forecast door's same-host ranks; default auto: 1 when a "
                             "member fits one card, else the fewest cards whose per-rank price fits)")
    parser.add_argument("--init-jobs", default="auto", metavar="N|auto",
                        help="member inits built at once (default auto: one per four cores, bounded by host RAM)")
