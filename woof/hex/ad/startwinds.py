# SPDX-License-Identifier: Apache-2.0
"""The physics' start cell winds of a one-card plan, folded back into the edge-wind sensitivity.

THE DEFECT THIS FIXES (hex 0.6.0). A one-card plan starts after the forecast has already rebuilt the physics' cell
winds (uReconstructZonal / uReconstructMeridional, kernel ``reconstruct_winds_v841_f32``) from the start edge wind u,
and its first step runs the physics on those cell winds BEFORE it rebuilds them. In the recorded program they are start
words of their own, so the window's gradient of the first step's physics with respect to wind lands on those words and
dJ/du left it out. A rank plan rebuilds the cell winds at the start of step 1, so its dJ/du carries the term: on the 3 km
storm box the two-card and one-card wind gradients differed by 7.4 per cent of the largest dJ/du word over 12 steps,
and with R^T (dJ/d start cell winds) added to the one-card dJ/du they agree to 3.7e-6.

The start cell winds are R u_start (R is that kernel's linear map from edge-normal wind to zonal and meridional cell
wind), so dJ/du += R^T [dJ/d zonal; dJ/d meridional]. On a plan whose step 1 rebuilds them before any read, their start
gradient is zero and the fold adds nothing.

Everything is read from the plan itself, for any mesh, regional or global, and for packed plans whose arrays sit at
offsets inside shared allocations: the window's first rebuild (its mesh tables by the kernel's argument order), and the
copies of its zonal and meridional outputs into the physics' arrays (whole-array copies only: same source pointer,
nlev rows).

  plan_info(plan_dir) -> dict, JSON-safe ('pools', 'mesh', 'step1_reads_before_rebuild', ...)
  term(info, plan_dir, win_dir, nedges_out) -> [nlev, nedges_out] float64 = R^T [g_zonal; g_meridional]
  start_check(info, plan_dir, u_start) -> relative max of (start pool words - R u_start), per pool
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

KERNEL = "reconstruct_winds_v841_f32"
SCHEMA = "gpuwm-hex.start-cell-winds/v1"
# argument order of reconstruct_winds_v841_f32 (tree/ad/specs/hex/dycore/reconstruct_winds_v841_f32.json)
ARGS = {"u": 4, "edges_on_cell": 5, "n_edges_on_cell": 6, "coeffs": 7, "lat": 8, "lon": 9,
        "x": 15, "y": 16, "z": 17, "u_p": 18, "v_p": 19}
MESH = ("edges_on_cell", "n_edges_on_cell", "coeffs", "lat", "lon")


def _targets(arg) -> list[tuple[int, int]]:
    if not isinstance(arg, dict):
        return []
    if isinstance(arg.get("target"), dict):
        return [(int(arg["target"]["allocation"]), int(arg["target"].get("offset", 0)))]
    return [(int(q["target"]["allocation"]), int(q["target"].get("offset", 0))) for q in (arg.get("pointers") or [])]


def _pointers(node):
    """Every (allocation, offset) a command names, wherever it sits (kernel arguments, copy-plan rows, uploads)."""
    if isinstance(node, dict):
        if "allocation" in node and "offset" in node and not isinstance(node["allocation"], (dict, list)):
            yield int(node["allocation"]), int(node["offset"])
        for v in node.values():
            if isinstance(v, (dict, list)):
                yield from _pointers(v)
    elif isinstance(node, list):
        for v in node:
            yield from _pointers(v)


def _i32(arg) -> int:
    return int.from_bytes(bytes.fromhex(arg["bytes_hex"])[:4], "little", signed=True)


def _array(arg) -> dict | None:
    """A cupy array argument: pointer target, element count, shape and byte strides (8-byte words after the pointer)."""
    t = _targets(arg)
    hx = bytes.fromhex(arg.get("bytes_hex", "")) if isinstance(arg, dict) else b""
    if len(t) != 1 or len(hx) < 32:
        return None
    words = [int.from_bytes(hx[i:i + 8], "little") for i in range(8, len(hx) - 7, 8)]
    ndim = (len(words) - 1) // 2
    if ndim < 1:
        return None
    shape, strides = words[1:1 + ndim], words[1 + ndim:1 + 2 * ndim]
    return {"target": t[0], "size": words[0], "shape": shape, "strides": strides}


def _start_bytes(plan_dir: Path, entry: dict) -> bytes:
    p = Path(plan_dir) / entry["path"]
    return gzip.open(p).read() if entry.get("encoding") == "gzip" else p.read_bytes()


def plan_info(plan_dir, plan_doc: dict | None = None) -> dict:
    """Where the plan's physics keeps the start cell winds and how its first rebuild makes them (JSON-safe)."""
    plan_dir = Path(plan_dir)
    d = plan_doc if plan_doc is not None else json.loads((plan_dir / "plan.json").read_text(encoding="utf-8"))
    al = {int(a["id"]): a for a in d["allocations"]}
    blocks = d["blocks"]
    steps = sorted(k for k in blocks if k.startswith("step_"))
    first = None
    for s in steps:
        hit = [i for i, c in enumerate(blocks[s]) if c.get("function") == KERNEL]
        if hit:
            first = (s, hit[0])
            break
    base = {"schema": SCHEMA, "needed": False, "pools": []}
    if first is None:
        return {**base, "why": f"the plan never runs {KERNEL}: no physics cell winds to fold"}
    blk = blocks[first[0]]
    cm = blk[first[1]]
    args = cm["arguments"]
    nlev, ncells, nedges, maxedges = (_i32(args[k]) for k in range(4))
    t = {n: _targets(args[k])[0] for n, k in ARGS.items()}
    # the physics' copies of the rebuild's zonal (u_p) and meridional (v_p) outputs: whole arrays, nlev rows
    pools = {}
    for c in blk[first[1] + 1:first[1] + 24]:
        if c.get("function") == KERNEL:
            break
        if not str(c.get("function", "")).startswith("cupy_copy__float32"):
            continue
        a = c.get("arguments") or []
        if len(a) < 2:
            continue
        src, dst = _array(a[0]), _array(a[1])
        if not src or not dst or len(src["shape"]) != 2 or len(dst["shape"]) != 2 or dst["shape"][0] != nlev:
            continue
        for name, out in (("zonal", "u_p"), ("meridional", "v_p")):
            if src["target"] == t[out] and dst["target"] != t[out] and name not in pools:
                width, row = int(dst["shape"][1]), int(dst["strides"][0]) // 4
                if width > ncells or row < width:
                    raise ValueError(f"start cell winds: {plan_dir}: the copy of {out} into allocation "
                                     f"{dst['target'][0]} has {width} columns a row of {row} words; not a cell field")
                pools[name] = {"alloc": dst["target"][0], "offset": dst["target"][1], "width": width, "row": row}
    if set(pools) != {"zonal", "meridional"}:   # a plan whose physics reads the rebuild's own outputs
        pools = {name: {"alloc": t[out][0], "offset": t[out][1], "width": ncells, "row": ncells}
                 for name, out in (("zonal", "u_p"), ("meridional", "v_p"))}
    pools = [{"name": n, **pools[n]} for n in ("zonal", "meridional")]

    def span(p):
        return p["offset"], p["offset"] + 4 * ((nlev - 1) * p["row"] + p["width"])

    # who reads them in step 1 before its first rebuild (on a rank plan nobody does: the fold then adds zero)
    s1 = blocks[steps[0]]
    stop = first[1] if first[0] == steps[0] else len(s1)
    readers = [i for i, c in enumerate(s1[:stop])
               if any(a == p["alloc"] and span(p)[0] <= o < span(p)[1] for a, o in _pointers(c) for p in pools)]
    missing = [p["alloc"] for p in pools if not al.get(p["alloc"], {}).get("initial")]
    mesh = {}
    for n in MESH:
        a, o = t[n]
        if not al.get(a, {}).get("initial"):
            raise ValueError(f"start cell winds: {plan_dir}: the rebuild's {n} (allocation {a}) has no start payload")
        mesh[n] = {"alloc": a, "offset": o, "initial": al[a]["initial"]}
    for p in pools:
        if not missing:
            p["initial"] = al[p["alloc"]]["initial"]
    return {**base, "needed": not missing, "why": (f"allocations {missing} have no start words" if missing else
                                                   "the start cell winds are start words of the plan"),
            "kernel_ncells": ncells, "kernel_nedges": nedges, "nlev": nlev, "maxedges": maxedges,
            "u": list(t["u"]), "pools": pools, "mesh": mesh, "first_rebuild": [first[0], first[1]],
            "step1_reads_before_rebuild": len(readers), "step1_first_readers": readers[:6]}


def dump_allocations(info: dict) -> list[int]:
    return sorted({p["alloc"] for p in info["pools"]}) if info.get("needed") else []


def matrix(info: dict, plan_dir):
    """R: edge-normal wind [nedges] -> [zonal; meridional] cell wind [2 ncells], as the kernel computes it."""
    import numpy as np
    import scipy.sparse as sp
    nc, ne, me = info["kernel_ncells"], info["kernel_nedges"], info["maxedges"]
    cache: dict[int, bytes] = {}

    def arr(name, dtype, count):
        m = info["mesh"][name]
        if m["alloc"] not in cache:
            cache[m["alloc"]] = _start_bytes(plan_dir, m["initial"])
        return np.frombuffer(cache[m["alloc"]], dtype, count=count, offset=m["offset"])
    eoc = arr("edges_on_cell", "<i4", nc * me).reshape(nc, me).astype(np.int64)
    neo = arr("n_edges_on_cell", "<i4", nc).astype(np.int64)
    co = arr("coeffs", "<f4", nc * me * 3).reshape(nc, me, 3).astype(np.float64)
    lat = arr("lat", "<f4", nc).astype(np.float64)
    lon = arr("lon", "<f4", nc).astype(np.float64)
    slat, clat, slon, clon = np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)
    rows, cols, vals = [], [], []
    for j in range(me):   # the kernel skips slots past n_edges_on_cell and edges outside [0, nedges)
        c = np.nonzero(j < neo)[0]
        e = eoc[c, j]
        ok = (e >= 0) & (e < ne)
        c, e = c[ok], e[ok]
        cx, cy, cz = co[c, j, 0], co[c, j, 1], co[c, j, 2]
        rows += [c, c + nc]
        cols += [e, e]
        vals += [-slon[c] * cx + clon[c] * cy, -slat[c] * (clon[c] * cx + slon[c] * cy) + clat[c] * cz]
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(2 * nc, ne))


def _pool_words(info: dict, p: dict, flat):
    """[nlev, ncells] float64 from a pool's words (its offset, row stride and width), zero past its width."""
    import numpy as np
    nl, nc = info["nlev"], info["kernel_ncells"]
    base = p["offset"] // 4
    idx = base + np.arange(nl)[:, None] * p["row"] + np.arange(p["width"])[None, :]
    out = np.zeros((nl, nc))
    out[:, :p["width"]] = flat[idx]
    return out


def term_from_flat(info: dict, plan_dir, flats: dict, nedges_out: int):
    """R^T [g_zonal; g_meridional] -> [nlev, nedges_out]; ``flats`` maps allocation -> its float32 gradient words."""
    import numpy as np
    R = matrix(info, plan_dir)
    z, m = (_pool_words(info, p, flats[p["alloc"]]) for p in info["pools"])
    T = (R.T @ np.concatenate([z, m], axis=1).T).T
    return T[:, :nedges_out]


def term(info: dict, plan_dir, win_dir, nedges_out: int):
    """R^T of the window's gradient of the start cell winds, [nlev, nedges_out] (None when the plan needs no fold)."""
    import numpy as np
    if not info.get("needed"):
        return None
    win_dir = Path(win_dir)
    flats = {}
    for a in dump_allocations(info):
        f = win_dir / f"grad-{a}.f32"
        if not f.exists():
            raise FileNotFoundError(f"the window {win_dir} did not dump the gradient of the start cell winds "
                                    f"(allocation {a}); it must be in --dump-grad")
        flats[a] = np.fromfile(f, "<f4").astype(np.float64)
    return term_from_flat(info, plan_dir, flats, nedges_out)


def start_check(info: dict, plan_dir, u_start) -> dict:
    """Relative max of (start pool words - R u_start) per pool, for u_start [nlev, >= nedges or the mesh's edges]."""
    import numpy as np
    R = matrix(info, plan_dir)
    nl, nc, ne = info["nlev"], info["kernel_ncells"], info["kernel_nedges"]
    u = np.zeros((nl, ne))
    n = min(ne, u_start.shape[1])
    u[:, :n] = u_start[:, :n]
    r = (R @ u.T).T
    out = {}
    for k, p in enumerate(info["pools"]):
        words = np.frombuffer(_start_bytes(plan_dir, p["initial"]), "<f4").astype(np.float64)
        have = _pool_words(info, p, words)[:, :p["width"]]
        want = r[:, k * nc:k * nc + p["width"]]
        out[p["name"]] = float(np.abs(have - want).max() / (np.abs(want).max() or 1.0))
    return out
