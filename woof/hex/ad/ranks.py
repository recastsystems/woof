# SPDX-License-Identifier: Apache-2.0
"""The rank route of the adjoint doors: one window split over N cards as N rank processes.

The engine already runs a window as N ranks (``ad-binder ranks --ranks N --cards LIST --pkg PREFIX -- window ...``):
rank k replays its own rank plan ``PREFIX-r<k>`` on its own card, and halos cross between ranks in the runner's
exchange. The runner does not put the ranks' answers back on the global mesh: each rank writes its own files under
``--out X/r<k>`` and the launcher writes ``X/ranks.json``. This module is the door's part, all of it table data and
small functions:

* which rows can run as ranks, and with what (the table's ``rank_splits``: a rank plan package per N, the memory a
  rank needs on its card, on the host and in /dev/shm, the exchange word, cards_max);
* the runner's line for N ranks;
* the cut of a global objective or direction into each rank's words, and the gather of each rank's gradient back
  onto the global mesh, through the rank package's map (``rank-gather.json`` in every rank plan; format below);
* the one window receipt the door reads, made from the N rank receipts.

``rank-gather.json`` (schema hexad.rank-gather/v1), one per rank plan: its format, its one writer and its one
reader are ``woof.hex.ad.rank_gather`` (the same file as the box chain's ``rank_gather.py`` beside split_plan.py, which
``tc3_views.py gather`` reads through), so the door and the chain cut and gather with one rule.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from . import rank_gather as _rg

GATHER_SCHEMA = _rg.SCHEMA
GATHER_FILE = _rg.FILE

# one error for the split route and its maps: a map the reader refuses is a split the door refuses
RankSplitError = _rg.RankGatherError


def split_of(table: dict, row_name: str) -> dict | None:
    """The row's rank split (table ``rank_splits``), or None: a row without one runs as before."""
    entry = (table.get("rank_splits") or {}).get(row_name)
    return dict(entry) if isinstance(entry, dict) else None


MADE = "made"   # table rank_splits.made: the split of a row made for a mesh (--mesh) whose rank plans are given


def made_split(table: dict, row: dict) -> dict | None:
    """The rank split of a row made for a mesh (``--mesh``, ``row["generated"]``) when the user gives its rank plans
    (``--rank-plans PREFIX``): the table's ``rank_splits.made`` entry with one plan entry per rank count up to its
    cards_max, each asking every card for the row's own one-card need (an upper bound: a rank holds 1/N of the columns
    and its halo). None when the table has no such entry.

    THE BREAKAGE THIS PREVENTS: a row made for a mesh never had a rank split (rank_splits names shipped rows only), so
    ``--cards 0,1,2,3 --rank-plans PREFIX`` on a made mesh (the 96-to-24 km zoom globe, 103,950 columns; anything over
    the 200,000-cell one-card limit) ran the window on card 0 alone and never read the rank plans."""
    entry = (table.get("rank_splits") or {}).get(MADE)
    if not isinstance(entry, dict) or not row.get("generated"):
        return None
    out = dict(entry)
    need = int(row.get("need_free_mib") or 0)
    out["plans"] = {str(n): {"pkg": "(given with --rank-plans)", "need_free_mib_rank": need,
                             "need_host_mib_rank": None, "shm_mib": None,
                             "measured": "not measured per rank: the made row's one-card need, an upper bound"}
                    for n in range(2, int(entry.get("cards_max") or 1) + 1)}
    return out


def choose(split: dict, row_name: str, n_cards: int, door: str) -> dict:
    """The split for N cards (one rank a card): the plan package and the memory each rank needs.

    THE BREAKAGE THIS PREVENTS: before this, ``--cards 0,1`` on a row without a column-sharded class used card 0 only
    and said nothing, so a user who asked for two cards got a one-card window and a one-card receipt."""
    doors = split.get("doors") or []
    if door not in doors:
        raise RankSplitError(f"row {row_name}: the {door} door does not run as ranks yet (the rank route is wired for "
                             f"{', '.join(doors) or 'no door'}); run it on one card (--card N)")
    cards_max = int(split.get("cards_max") or 1)
    if n_cards > cards_max:
        raise RankSplitError(f"row {row_name}: {n_cards} cards given; this row splits over at most {cards_max}")
    plans = split.get("plans") or {}
    entry = plans.get(str(n_cards))
    if entry is None:
        have = sorted(int(n) for n in plans)
        raise RankSplitError(f"row {row_name}: no {n_cards}-rank plan package for this row (it has "
                             f"{', '.join(str(n) for n in have) or 'none'}); give {' or '.join(str(n) for n in have) or 'one'} "
                             f"card{'s' if have else ''} with --cards, or one card with --card")
    word = entry.get("exchange") or split.get("exchange") or "file"
    exchange = (split.get("exchanges") or {}).get(word)
    if exchange is None:
        raise RankSplitError(f"row {row_name}: exchange word {word!r} is not in the row's exchanges")
    return {"ranks": n_cards, "pkg": entry["pkg"], "gather": split.get("gather") or GATHER_FILE,
            "need_free_mib_rank": int(entry["need_free_mib_rank"]),
            "need_host_mib_rank": int(entry["need_host_mib_rank"]) if entry.get("need_host_mib_rank") else None,
            "shm_mib": int(entry["shm_mib"]) if entry.get("shm_mib") else None,
            "exchange": word, "exchange_env": dict(exchange.get("env") or {}),
            "exchange_runner_label": exchange.get("runner_label"),
            "env_drop": list(split.get("env_drop") or []), "window_args": list(split.get("window_args") or []),
            "window_args_drop": list(split.get("window_args_drop") or []),
            "assets": dict(split.get("assets") or {}), "row_overrides": dict(split.get("row_overrides") or {}),
            "ownerreduce": dict(split["ownerreduce"]) if split.get("ownerreduce") else None,
            "measured": entry.get("measured"), "what": split.get("what"),
            "experimental": dict(split["experimental"]) if split.get("experimental") else None}


def shm_free_mib(path: str = "/dev/shm") -> int | None:
    try:
        st = os.statvfs(path)
    except OSError:
        return None
    return int(st.f_bavail * st.f_frsize // (1 << 20))


def refuse_shm(row_name: str, need: int | None, free: int | None) -> None:
    """THE BREAKAGE THIS PREVENTS: the ranks' halo messages live in /dev/shm for the whole window (two 120 km ranks
    held about 24 GB there on 9 Nov); a tmpfs that fills ends a rank part way through with a bus error."""
    if need is None or free is None or free >= need:
        return
    raise RankSplitError(f"/dev/shm has {free:,} MiB free and row {row_name}'s ranks keep about {need:,} MiB of halo "
                         "messages there; free it or enlarge it (mount -o remount,size=...). Nothing was downloaded")


def rank_dirs(prefix: Path, n: int) -> list[Path]:
    return [Path(f"{prefix}-r{k}") for k in range(n)]


def refuse_rank_plans_missing(prefix: Path, n: int, gather: str = GATHER_FILE) -> None:
    """Every rank plan, and its map back to the global mesh, before the card work starts."""
    lacking = []
    for k, d in enumerate(rank_dirs(prefix, n)):
        if not (d / "plan.json").is_file():
            lacking.append(f"{d}/plan.json")
        elif not (d / gather).is_file():
            lacking.append(f"{d}/{gather}")
    if lacking:
        raise RankSplitError(f"the {n}-rank plan package is not complete: {lacking[0]}"
                             + (f" and {len(lacking) - 1} more" if len(lacking) > 1 else "")
                             + " missing. A rank package is a decomposed recording cut per rank (split_plan.py), with "
                               "each rank's map back to the global mesh (" + gather + ")")


def ranks_argv(window_argv: list[str], n: int, cards: list[str], prefix: Path) -> list[str]:
    """``runner window ARGS`` -> ``runner ranks --ranks N --cards LIST --pkg PREFIX -- window ARGS`` (rank k on card
    k; the launcher gives --out, --bstore and AD_RESULT_DUMP a per-rank /r<k> and replaces {rank} in every word)."""
    if window_argv[1] != "window":
        raise RankSplitError(f"not a window line: {window_argv[1]}")
    if len(cards) != n:
        raise RankSplitError(f"{n} ranks need {n} cards, one each (given {','.join(cards)})")
    return [window_argv[0], "ranks", "--ranks", str(n), "--cards", ",".join(cards), "--pkg", str(prefix), "--",
            *window_argv[1:]]


# ---------------------------------------------------------------------------------------------- the rank maps
# One reader (woof.hex.ad.rank_gather): the door, tc3_views.py gather and grad_ranks.py read the same file the same way.
def load_gather(rank_dir: Path, name: str = GATHER_FILE) -> dict:
    return _rg.load(rank_dir, name)


cut_terms = _rg.cut_terms            # objective to the owner only, direction to every copy
gather_allocs = _rg.gather_allocs    # a gradient: every rank's copies, owned, halo and replicated, summed
local_allocs = _rg.local_allocs      # a rank's dump list for gather_allocs
global_words = _rg.global_words
refuse_other_plan = _rg.refuse_other_plan


# ----------------------------------------------------------------------------------------- the rank receipts
AGREE_KEYS = (("tangent_dir", "w_Mx"), ("tangent_dir", "x_MTw"), ("tangent_dir", "rel"))


def exchange_seen(peer) -> str:
    """The exchange a rank's runner reports: its peer block names the shared slots when it used them."""
    if not isinstance(peer, dict):
        return "no peer block"
    return str(peer.get("transport") or "file mailbox")


def compose_window_doc(out: Path, n: int, exchange_label: str | None = None) -> dict:
    """The one window receipt the doors read, from the N rank receipts under out/r<k> and the launcher's ranks.json.

    The global numbers (w_Mx, x_MTw, the dot) are reduced over ranks by the runner, so every rank reports them;
    they must agree, or the door does not report them. Rank 0's receipt carries them. Each rank's own receipt stays
    beside it (``ranks``)."""
    launcher = out / "ranks.json"
    if not launcher.is_file():
        raise RankSplitError(f"the rank launcher wrote no {launcher}")
    lr = json.loads(launcher.read_text(encoding="utf-8"))
    docs = []
    for k in range(n):
        p = out / f"r{k}" / "window.json"
        if not p.is_file():
            raise RankSplitError(f"rank {k} wrote no {p}")
        docs.append(json.loads(p.read_text(encoding="utf-8")))
    for outer, inner in AGREE_KEYS:
        values = [(d.get(outer) or {}).get(inner) for d in docs]
        if len({json.dumps(v) for v in values}) > 1:
            raise RankSplitError(f"the ranks disagree on {outer}.{inner} ({values}): the runner did not reduce it "
                                 "over ranks, so the door has no global value to report")
    seen = [exchange_seen(d.get("peer")) for d in docs]
    if exchange_label is not None and any(s != exchange_label for s in seen):
        raise RankSplitError(f"the window asked for exchange {exchange_label!r} and the ranks report {seen}: this "
                             "runner does not have that exchange")
    doc = dict(docs[0])
    doc["ranks"] = {"n": n, "launcher": str(launcher), "exchange_seen": seen,
                    "rank_rows": lr.get("rank_rows"), "cards": lr.get("cards"), "transport": lr.get("transport"),
                    "wall_s": lr.get("wall_s"),
                    "rank_receipts": [str(out / f"r{k}" / "window.json") for k in range(n)]}
    return doc
