# SPDX-License-Identifier: Apache-2.0
"""rank-gather.json: each rank plan's map back to the global mesh, written once and read by everyone.

One file, one reader. The front door (``woof.hex.ad.ranks``) and the box chain (``tc3_views.py gather``,
``grad_ranks.py``) both read a rank plan's ``rank-gather.json`` through this module, and ``write`` here is the only
thing that makes one. The same source file sits in two trees: ``tree/src/hexcore/ad/rank_gather.py`` (the package)
and ``tree/ad/phys/p1/tools/multidevice/rank_gather.py`` (beside split_plan.py); the two copies are byte-equal.
It needs numpy and the standard library only.

THE BREAKAGE THIS PREVENTS: before it, no tool wrote a rank plan's map back to the global mesh, so the door refused
``--cards`` by name; and the box chain gathered a split gradient from owned cells only, which drops the gradient the
halo copies hold at boundary 0 (75,992 nonzero halo words of rho at 2 ranks at 120 km; the split-equal check).

Format (schema hexad.rank-gather/v1), one per rank plan ``PREFIX-r<k>``::

    {"schema": "hexad.rank-gather/v1", "rank": k, "ranks": N,
     "global_plan": {"plan_sha256": "...", "path": "..."},           # the one-card plan the global copies name
     "entities": {"cell": {"l2g": "l2g-cell.i32", "owned": n_own, "local": n_local}, "edge": {...}, "vertex": {...}},
     "fields": {NAME: {"grid": "cell" | "edge" | "vertex" | null,
                       "global": [COPY, ...], "local": [COPY, ...]}}}

    COPY = {"alloc": A, "offset": BYTES, "shape": [...],
            "strides": [BYTES, ...],   # optional: C order of 4-byte words when absent
            "axis": i,                 # optional: the entity axis (default the last); absent for a replicated copy
            "count": n,                # local copies only, optional: entries of the entity axis that are mesh
                                       #   entities (default the entity's local count); a padding entry past them
                                       #   (MPAS's nCells + 1 on a limited-area mesh) maps to nothing
            "step": s}                 # optional: the step boundary the copy is the field at

``l2g-<entity>.i32`` holds, for each local index, its global index (little-endian int32); the first ``owned`` are the
entities this rank owns, the rest are its halo copies (the ownership files of make_owners.py, same order). ``global``
and ``local`` list a field's copies in the same order: copy i of the one-card plan is copy i of the rank plan.

The rules:
  * a gradient is every rank's local copies, owned and halo alike, SUMMED into the global entity (a halo copy's
    cotangent is part of the owner's gradient: the reverse of the forward's copy into it); a replicated copy (no
    entity axis) is summed over the ranks the same way (each rank's copy is an input of that rank's program);
  * a state or tangent field is taken from the owner (the first ``owned`` local entries);
  * an objective term goes to the owner only (each global word counted once in J); a direction term goes to every
    local copy, halo included.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = "hexad.rank-gather/v1"
FILE = "rank-gather.json"
ENTITIES = ("cell", "edge", "vertex")
_PLURAL = {"cell": "cells", "edge": "edges", "vertex": "vertices"}


class RankGatherError(Exception):
    pass


# --------------------------------------------------------------------------------------------------- words
def view_words(copy: dict) -> np.ndarray:
    """The 4-byte word index (inside its allocation) of every element of a copy, in the copy's shape."""
    shape = [int(s) for s in copy["shape"]]
    strides = copy.get("strides")
    if strides is None:   # C order of 4-byte words
        strides, acc = [], 4
        for s in reversed(shape):
            strides.insert(0, acc)
            acc *= s
    idx = np.full(shape, int(copy["offset"]), np.int64)
    for d, (n, s) in enumerate(zip(shape, strides)):
        sh = [1] * len(shape)
        sh[d] = n
        idx = idx + (np.arange(n, dtype=np.int64) * int(s)).reshape(sh)
    if np.any(idx % 4):
        raise RankGatherError(f"a copy of allocation {copy['alloc']} is not on 4-byte words")
    return idx // 4


def extent_bytes(copy: dict) -> int:
    """Bytes from the copy's first word to one past its last."""
    w = view_words(copy)
    return 0 if w.size == 0 else int(w.max() + 1) * 4 - int(copy["offset"])


def _axis(copy: dict) -> int:
    return int(copy.get("axis", len(copy["shape"]) - 1))


def copy_pairs(gmap: dict, name: str, i: int, owner_only: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """(local words, global words) of copy i of a field, element for element: every mesh entity this rank holds
    (owner_only: only the ones it owns), each level. A replicated copy pairs word for word."""
    f = gmap["fields"][name]
    loc, glo = f["local"][i], f["global"][i]
    lw, gw = view_words(loc), view_words(glo)
    if f.get("grid") is None:
        if lw.shape != gw.shape:
            raise RankGatherError(f"{name} copy {i}: a replicated copy of shape {list(lw.shape)} against "
                                  f"{list(gw.shape)}")
        return lw.ravel(), gw.ravel()
    e = gmap["entities"][f["grid"]]
    la, ga = _axis(loc), _axis(glo)
    n = int(loc.get("count", e["local"]))
    if owner_only:
        n = min(n, int(e["owned"]))
    if n > lw.shape[la] or n > int(e["local"]):
        raise RankGatherError(f"{name} copy {i}: {n} entities on an axis of {lw.shape[la]} ({e['local']} local)")
    lw = np.moveaxis(np.take(lw, np.arange(n), axis=la), la, -1)
    gw = np.moveaxis(np.take(gw, e["_l2g"][:n].astype(np.int64), axis=ga), ga, -1)
    if lw.shape != gw.shape:
        raise RankGatherError(f"{name} copy {i}: local {list(lw.shape)} and global {list(gw.shape)} do not pair")
    return lw.ravel(), gw.ravel()


# ---------------------------------------------------------------------------------------------- load, find
def load(rank_dir, name: str = FILE) -> dict:
    """A rank plan's map, with each entity's local-to-global index read (``_l2g``)."""
    rank_dir = Path(rank_dir)
    doc = json.loads((rank_dir / name).read_text(encoding="utf-8"))
    if doc.get("schema") != SCHEMA:
        raise RankGatherError(f"{rank_dir / name}: schema {doc.get('schema')!r}, not {SCHEMA}")
    for entity, e in doc["entities"].items():
        l2g = np.fromfile(rank_dir / e["l2g"], "<i4")
        if l2g.size != int(e["local"]) or not 0 < int(e["owned"]) <= l2g.size:
            raise RankGatherError(f"{rank_dir / e['l2g']}: {l2g.size} entries for {e['local']} local {entity} "
                                  f"({e['owned']} owned)")
        e["_l2g"] = l2g
    return doc


def find_copy(gmap: dict, alloc: int, offset: int, step=None) -> tuple[str, int] | None:
    """(field, copy) whose global copy starts at this allocation and byte (and step, when given)."""
    for name, f in gmap["fields"].items():
        for i, c in enumerate(f["global"]):
            if int(c["alloc"]) == int(alloc) and int(c["offset"]) == int(offset) and (
                    step is None or str(c.get("step")) == str(step)):
                return name, i
    return None


def _require(gmap: dict, alloc: int, offset: int, step=None) -> tuple[str, int]:
    hit = find_copy(gmap, alloc, offset, step)
    if hit is None:
        raise RankGatherError(f"rank {gmap.get('rank')}: allocation {alloc} at byte {offset}"
                              + (f" (step {step})" if step is not None else "") + " is in no field of the rank map")
    return hit


# ----------------------------------------------------------------------------------------------- the readers
def gather_copy(maps: list[dict], alloc: int, offset: int, read_alloc, step=None) -> np.ndarray:
    """dJ/d(one global copy), in that copy's shape (a padding entry stays 0): every rank's local copy, halo
    included, summed. read_alloc(rank, alloc) returns that rank allocation's float32 words from word 0."""
    out = None
    for r, gmap in enumerate(maps):
        name, i = _require(gmap, alloc, offset, step)
        glo = gmap["fields"][name]["global"][i]
        if out is None:
            gw0 = view_words(glo)
            base = int(gw0.min())
            pos = np.full(int(gw0.max()) + 1 - base, -1, np.int64)
            pos[(gw0 - base).ravel()] = np.arange(gw0.size)
            out = np.zeros(gw0.size, np.float64)
            shape = list(gw0.shape)
        lw, gw = copy_pairs(gmap, name, i)
        src = np.asarray(read_alloc(r, int(gmap["fields"][name]["local"][i]["alloc"])))
        np.add.at(out, pos[gw - base], src[lw].astype(np.float64))
    return out.reshape(shape)


def owned_copy(maps: list[dict], alloc: int, offset: int, read_alloc, step=None) -> np.ndarray:
    """A state or tangent copy on the global mesh: each entity from the rank that owns it (padding entries NaN)."""
    out = None
    for r, gmap in enumerate(maps):
        name, i = _require(gmap, alloc, offset, step)
        if out is None:
            gw0 = view_words(gmap["fields"][name]["global"][i])
            base = int(gw0.min())
            pos = np.full(int(gw0.max()) + 1 - base, -1, np.int64)
            pos[(gw0 - base).ravel()] = np.arange(gw0.size)
            out = np.full(gw0.size, np.nan, np.float64)
            shape = list(gw0.shape)
        lw, gw = copy_pairs(gmap, name, i, owner_only=gmap["fields"][name].get("grid") is not None)
        src = np.asarray(read_alloc(r, int(gmap["fields"][name]["local"][i]["alloc"])))
        out[pos[gw - base]] = src[lw]
    return out.reshape(shape)


def gather_allocs(maps: list[dict], allocs, read_alloc, alloc_words: dict, step=None) -> tuple[dict, dict]:
    """The split gradient written back into one-card allocations (for the one-card writer): for each global
    allocation, every word any rank's copy names gets the sum of every rank's copies of it (owned, halo and
    replicated alike). Copies that alias the same words are counted once (pairs are made unique per rank).
    Returns ({alloc: float32 words}, {alloc: report})."""
    allocs = [int(a) for a in allocs]
    out = {a: np.zeros(int(alloc_words[a]), np.float64) for a in allocs}
    covered = {a: np.zeros(int(alloc_words[a]), bool) for a in allocs}
    halo_nonzero = {a: 0 for a in allocs}
    copies = {a: 0 for a in allocs}
    for r, gmap in enumerate(maps):
        per = {}   # (global alloc, local alloc) -> list of (lw, gw, owned mask)
        for name, f in gmap["fields"].items():
            for i, glo in enumerate(f["global"]):
                ga = int(glo["alloc"])
                if ga not in out or (step is not None and str(glo.get("step")) != str(step)):
                    continue
                lw, gw = copy_pairs(gmap, name, i)
                if f.get("grid") is None:
                    own = np.ones(lw.size, bool)
                else:
                    lo, _ = copy_pairs(gmap, name, i, owner_only=True)
                    own = np.isin(lw, lo)
                per.setdefault((ga, int(f["local"][i]["alloc"])), []).append((lw, gw, own))
                if r == 0:
                    copies[ga] += 1
        for (ga, la), parts in per.items():
            lw = np.concatenate([p[0] for p in parts])
            gw = np.concatenate([p[1] for p in parts])
            own = np.concatenate([p[2] for p in parts])
            keep = np.unique((lw.astype(np.int64) << 32) | gw.astype(np.int64), return_index=True)[1]
            lw, gw, own = lw[keep], gw[keep], own[keep]
            if np.any(gw >= out[ga].size):
                raise RankGatherError(f"rank {r}: a copy of allocation {ga} names word {int(gw.max())} of "
                                      f"{out[ga].size}")
            src = np.asarray(read_alloc(r, la))
            vals = src[lw].astype(np.float64)
            np.add.at(out[ga], gw, vals)
            covered[ga][gw] = True
            halo_nonzero[ga] += int(np.count_nonzero(vals[~own]))
    rep = {a: {"words": int(out[a].size), "covered_words": int(covered[a].sum()), "copies": copies[a],
               "halo_words_nonzero": halo_nonzero[a]} for a in allocs}
    return {a: out[a].astype(np.float32) for a in allocs}, rep


def local_allocs(gmap: dict, global_allocs, step=None) -> list[int]:
    """The rank allocations that hold a copy of any of these global allocations (at this step, when given): the
    rank's dump list for gather_allocs, which reads exactly these."""
    want = {int(a) for a in global_allocs}
    out = set()
    for f in gmap["fields"].values():
        for glo, loc in zip(f["global"], f["local"]):
            if int(glo["alloc"]) in want and (step is None or str(glo.get("step")) == str(step)):
                out.add(int(loc["alloc"]))
    return sorted(out)


def global_words(maps: list[dict], alloc: int) -> int:
    """One past the last word of a global allocation that any rank map names (its size when no catalog is at hand)."""
    top = 0
    for gmap in maps:
        for f in gmap["fields"].values():
            for glo in f["global"]:
                if int(glo["alloc"]) == int(alloc):
                    w = view_words(glo)
                    if w.size:
                        top = max(top, int(w.max()) + 1)
    return top


def cut_terms(terms: list, gmap: dict, owner_only: bool) -> list:
    """A global objective's or direction's terms [alloc, word, weight] as this rank's terms.

    owner_only: an objective term goes to the rank that owns its entity only (each global word counted once in J);
    a direction term goes to every local copy, halo included (a direction must agree where ranks overlap). A term
    the map cannot place is refused (a word silently dropped would change J or the direction), and so is an
    objective term on a replicated copy (it has no one owner)."""
    if not terms:
        return []
    ta = np.asarray([int(t[0]) for t in terms], np.int64)
    tw = np.asarray([int(t[1]) for t in terms], np.int64)
    pieces = []   # (term index, local alloc, local word)
    for alloc in np.unique(ta).tolist():
        sel = np.nonzero(ta == alloc)[0]
        words = tw[sel]
        placed = np.zeros(sel.size, bool)
        gws, las, lws = [], [], []
        for name, f in gmap["fields"].items():
            for i, glo in enumerate(f["global"]):
                if int(glo["alloc"]) != alloc:
                    continue
                inside = np.isin(words, view_words(glo).ravel())
                placed |= inside
                if f.get("grid") is None and owner_only:
                    if inside.any():
                        w = int(words[np.argmax(inside)])
                        raise RankGatherError(f"objective term at allocation {alloc} word {w} lies in a replicated "
                                              "copy: it has no one owner")
                    continue
                lw, gw = copy_pairs(gmap, name, i, owner_only=owner_only)
                gws.append(gw)
                lws.append(lw)
                las.append(np.full(lw.size, int(f["local"][i]["alloc"]), np.int64))
        if not placed.all():
            w = int(words[np.argmin(placed)])
            raise RankGatherError(f"term at allocation {alloc} word {w} is in no field of the rank map")
        if not gws:
            continue
        gw, la, lw = np.concatenate(gws), np.concatenate(las), np.concatenate(lws)
        trip = np.unique(np.stack([gw, la, lw], axis=1), axis=0)   # aliasing copies place a word once
        gw, la, lw = trip[:, 0], trip[:, 1], trip[:, 2]
        lo, hi = np.searchsorted(gw, words, "left"), np.searchsorted(gw, words, "right")
        cnt = hi - lo
        if cnt.sum():
            rep = np.repeat(np.arange(sel.size), cnt)
            at = np.repeat(lo - np.concatenate([[0], np.cumsum(cnt)[:-1]]), cnt) + np.arange(int(cnt.sum()))
            pieces.append(np.stack([sel[rep], la[at], lw[at]], axis=1))
    if not pieces:
        return []
    allp = np.concatenate(pieces)
    allp = allp[np.lexsort((allp[:, 2], allp[:, 1], allp[:, 0]))]
    return [[int(la), int(lw), terms[int(t)][2]] for t, la, lw in allp.tolist()]


# ------------------------------------------------------------------------------------------------- the writer
def plan_sha256(plan_dir) -> str | None:
    p = Path(plan_dir) / "plan.json"
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def refuse_other_plan(maps: list[dict], plan_dir) -> str:
    """The one-card plan the maps' global copies name must be the plan the door cuts the objective on.

    THE BREAKAGE THIS PREVENTS: rank maps made against another recording (another start, another window length) name
    other allocation numbers; the objective would be cut onto the wrong words and the gradient gathered into the wrong
    cells, with no error. Returns 'equal' or 'not recorded in the maps' (a map written before global_plan existed)."""
    have = plan_sha256(plan_dir)
    named = {(m.get("global_plan") or {}).get("plan_sha256") for m in maps} - {None}
    if not named:
        return "not recorded in the maps"
    if named != {have}:
        raise RankGatherError(f"the rank maps were made against one-card plan sha256 {sorted(named)[0][:16]}, and the "
                              f"plan given is {str(have)[:16]} ({plan_dir}): give the one-card plan the rank package "
                              "was split from")
    return "equal"


def _strip(view: dict, step) -> dict:
    out = {"alloc": int(view["allocation"]), "offset": int(view["offset"]), "shape": [int(s) for s in view["shape"]],
           "strides": [int(s) for s in view["strides"]]}
    if step is not None:
        out["step"] = int(step)
    return out


def from_views(views: dict, owners: list[dict], steps=None, global_plan: dict | None = None) -> list[dict]:
    """The N rank maps from tc3_views.py's aligned views (one-card view against rank view, each with its entity
    axis [axis, entity, 'local'|'owned', pad] or None) and the ownership files (make_owners.py rank-<r>.npz, as
    dicts). Only float32 views; a view key is one field, with one copy per step asked."""
    n = len(views["ranks"])
    if len(owners) != n:
        raise RankGatherError(f"{n} ranks in the views and {len(owners)} ownership files")
    steps = sorted({str(s) for s in (steps or views.get("steps") or [])}, key=int)
    common = None
    for rk in views["ranks"]:
        keys = {(s, k) for s in steps for k, v in (rk["boundaries"].get(s) or {}).items()
                if v["native"].get("dtype") in ("float32", "<f4", "f4")}
        common = keys if common is None else common & keys
    docs = []
    for r, rk in enumerate(views["ranks"]):
        z = owners[r]
        ents = {}
        for e in ENTITIES:
            l2g = np.asarray(z[f"{e}_l2g"], np.int64)
            ents[e] = {"l2g": f"l2g-{e}.i32", "owned": int(z[f"n_owned_{_PLURAL[e]}"]), "local": int(l2g.size),
                       "_l2g": l2g}
        fields = {}
        for s, key in sorted(common, key=lambda x: (x[1], int(x[0]))):
            pair = rk["boundaries"][s][key]
            nv, rv, ax = pair["native"], pair["rank"], pair.get("entity_axis")
            g, loc = _strip(nv, s), _strip(rv, s)
            if ax is not None:
                axis, ent, extent, pad = int(ax[0]), ax[1], ax[2], int(ax[3]) if len(ax) > 3 else 0
                g["axis"], loc["axis"] = axis, axis
                loc["count"] = ents[ent]["local"] if extent == "local" else ents[ent]["owned"]
                if loc["shape"][axis] != loc["count"] + pad:
                    raise RankGatherError(f"rank {r} {key} step {s}: entity axis {loc['shape'][axis]} is not "
                                          f"{loc['count']} + pad {pad}")
            else:
                ent = None
            f = fields.setdefault(key, {"grid": ent, "global": [], "local": []})
            if f["grid"] != ent:
                raise RankGatherError(f"{key}: entity {ent} at step {s}, {f['grid']} at another step")
            if any(c["alloc"] == g["alloc"] and c["offset"] == g["offset"] for c in f["global"]):
                continue   # the same copy at another step boundary
            f["global"].append(g)
            f["local"].append(loc)
        doc = {"schema": SCHEMA, "rank": r, "ranks": n, "entities": ents, "fields": fields,
               "steps": [int(s) for s in steps], "made_by": "rank_gather.py from tc3_views views + make_owners.py"}
        if global_plan:
            doc["global_plan"] = dict(global_plan)
        docs.append(doc)
    return docs


def save(doc: dict, rank_dir) -> Path:
    rank_dir = Path(rank_dir)
    rank_dir.mkdir(parents=True, exist_ok=True)
    out = {k: v for k, v in doc.items() if k != "entities"}
    out["entities"] = {}
    for e, x in doc["entities"].items():
        np.asarray(x["_l2g"], "<i4").tofile(rank_dir / x["l2g"])
        out["entities"][e] = {k: v for k, v in x.items() if k != "_l2g"}
    p = rank_dir / FILE
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(p)
    return p


def _owners(d) -> list[dict]:
    paths = sorted(Path(d).glob("rank-*.npz"), key=lambda p: int(p.stem.split("-")[1]))
    if not paths:
        raise RankGatherError(f"no rank-<r>.npz ownership files in {d}")
    out = []
    for p in paths:
        z = np.load(p)
        out.append({k: z[k] for k in z.files})
    return out


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="rank_gather.py", description=__doc__.split("\n\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    w = sp.add_parser("write", help="write PREFIX-r<k>/rank-gather.json from aligned views and ownership files")
    w.add_argument("--views", required=True)
    w.add_argument("--owners", required=True)
    w.add_argument("--rank-prefix", required=True)
    w.add_argument("--steps", default=None, help="step boundaries to map, comma list (default: the views' steps)")
    w.add_argument("--native-plan", default=None, help="the one-card plan the global copies name (its sha256 is kept)")
    g = sp.add_parser("gather", help="the split gradient back into one-card allocations (grad-<alloc>.f32)")
    g.add_argument("--rank-prefix", required=True)
    g.add_argument("--ranks", type=int, required=True)
    g.add_argument("--rank-dirs", required=True, help="each rank's window folder holding grad-<alloc>.f32")
    g.add_argument("--allocs", required=True)
    g.add_argument("--native-catalog", required=True, help="catalog-a.json of the one-card plan (allocation bytes)")
    g.add_argument("--step", default="0")
    g.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "write":
        views = json.loads(Path(a.views).read_text())
        gp = None
        if a.native_plan:
            gp = {"path": str(a.native_plan), "plan_sha256": plan_sha256(a.native_plan)}
        docs = from_views(views, _owners(a.owners), a.steps.split(",") if a.steps else None, gp)
        for k, doc in enumerate(docs):
            save(doc, f"{a.rank_prefix}-r{k}")
        print(json.dumps({"ranks": len(docs), "fields": len(docs[0]["fields"]),
                          "copies": sum(len(f["global"]) for f in docs[0]["fields"].values()),
                          "owned": [d["entities"]["cell"]["owned"] for d in docs],
                          "local": [d["entities"]["cell"]["local"] for d in docs]}))
        return 0
    maps = [load(f"{a.rank_prefix}-r{k}") for k in range(a.ranks)]
    dirs = [Path(d) for d in a.rank_dirs.split(",")]
    if len(dirs) != a.ranks:
        raise SystemExit(f"{a.ranks} ranks and {len(dirs)} rank folders")
    cat = json.loads(Path(a.native_catalog).read_text())["allocations"]
    nbytes = {int(x["id"]): int(x["bytes"]) for x in (cat.values() if isinstance(cat, dict) else cat)}
    allocs = [int(x) for x in a.allocs.split(",")]
    grads, rep = gather_allocs(maps, allocs, lambda r, al: np.fromfile(dirs[r] / f"grad-{al}.f32", "<f4"),
                               {al: nbytes[al] // 4 for al in allocs}, a.step)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for al, g in grads.items():
        g.tofile(out / f"grad-{al}.f32")
    (out / "gather.json").write_text(json.dumps({"rule": "every rank's copies summed (owned, halo, replicated)",
                                                 "allocs": {str(k): v for k, v in rep.items()}}, indent=1) + "\n")
    print(json.dumps({str(k): v for k, v in rep.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
