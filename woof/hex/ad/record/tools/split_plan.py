#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The multi-device runner: split a sealed N-rank-thread plan into one plan per rank.

The decomposed forward recorded as N rank threads on one card (multirank observer, deterministic baton) is ONE command
stream. Every recorded command carries `recorded_sequence`, the driver-trace sequence number of the call; the trace
records the host thread that made it. The rank threads are the N threads that launch kernels (the observer names them
hex-rank-<r>; with MR_RANK_POOLS=1 each has its own CuPy pool). So:

  * a command with a sequence belongs to the rank of its thread; a command with none (event and stream syncs, host
    boundaries, RK scope, host branches) is kept in every rank, in place;
  * an allocation is OWNED by a rank when only that rank's commands name it, and SHARED when several ranks' commands do
    (one card's per-device module caches and scratch that both rank threads reuse in turn: pkg-p2b has 447 such blocks
    with no cross-rank copy into them). A shared allocation is kept in every rank plan and dropped from the reference
    check (each process holds only its own rank's writes there), so every OWNED allocation's per-step hash still proves
    the rank's forward;
  * a device copy whose source and destination are owned by different ranks is the halo DMA: it becomes PeerSend in the
    source rank's plan and PeerRecv in the destination rank's plan under one channel id.

Refusals (each names its breakage): a command of one rank that names an allocation another rank OWNS (the data would
not be there in the other process); a host allocation written from one rank's device data (host_publish) that another
rank reads (it would read its own stale value); a block that is empty on some rank. --analyze prints the census only.

Usage: split_plan.py PLAN_DIR --ranks N --trace DRIVER.jsonl.gz --out OUT_PREFIX [--global-arena] [--analyze]
  OUT_PREFIX-r<k>/ gets plan.json, reference.jsonl and a symlink for every other entry of PLAN_DIR (payloads shared).
"""
import argparse, collections, gzip, hashlib, json, os, re, sys


def load(p):
    return json.load(gzip.open(p) if p.endswith(".gz") else open(p))


def slices(o, out):
    if isinstance(o, dict):
        if "allocation" in o and "offset" in o and isinstance(o["allocation"], int):
            out.append(o["allocation"])
        for v in o.values():
            slices(v, out)
    elif isinstance(o, list):
        for v in o:
            slices(v, out)


def slice_pairs(o, out):
    if isinstance(o, dict):
        if "allocation" in o and "offset" in o and isinstance(o["allocation"], int):
            out.append((o["allocation"], o["offset"]))
        for v in o.values():
            slice_pairs(v, out)
    elif isinstance(o, list):
        for v in o:
            slice_pairs(v, out)


def op_ids(op):
    out = []
    slices(op, out)
    if op["kind"] in ("allocate", "free"):
        out.append(op["id"])
    return out


# Kernels that read a table of device addresses another command built (the RRFS copy plan rrfs_copy_plan_upload writes,
# the TEMPO pointer tables): replayed in another rank's process they dereference the issuing rank's memory. THE BREAKAGE
# THIS PREVENTS: rank 1 stopped at step 3 with CUDA 700 after replaying rank 0's rrfs_glue_copy_plan on block 786.
# Two cards at 3 km (2026-11-13): ruc_land_tile_preflight (RUC land tiles, [1,1,1]) reads the tile's column pointer table in a
# shared block (103+3262464 on the d4-15 storm box): replayed in rank 1's process it dereferenced rank 0's columns and
# stopped rank 1 with CUDA 700 at its first step. Each rank runs its own.
POINTER_TABLE_KERNELS = {"rrfs_glue_copy_plan", "ruc_land_tile_preflight"}


def thread_of_sequence(trace):
    pat = re.compile(rb'"sequence":(\d+).*?"thread_id":(\d+)')
    seq = {}
    with gzip.open(trace) as f:
        for line in f:
            m = pat.search(line)
            if m:
                seq[int(m.group(1))] = int(m.group(2))
    return seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan_dir")
    ap.add_argument("--ranks", type=int, required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out")
    ap.add_argument("--plan-name", default="plan.json")
    ap.add_argument("--global-arena", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--ownership-from", help="the partial plan of the same recording (every command keeps recorded_sequence); the sealed plan drops it from kernel launches")
    a = ap.parse_args()
    src = os.path.join(a.plan_dir, a.plan_name)
    plan = load(src)
    nr = a.ranks
    dev = {x["id"]: x["bytes"] for x in plan["allocations"]}
    for ops in plan["blocks"].values():
        for op in ops:
            if op["kind"] == "allocate":
                dev[op["id"]] = max(dev.get(op["id"], 0), op["bytes"])
    host = {x["id"] for x in plan.get("host_allocations", [])}
    th = thread_of_sequence(a.trace)
    own_plan = load(a.ownership_from) if a.ownership_from else plan
    # Rank threads: the N threads with the most kernel launches, numbered by their first launch.
    launches, first = collections.Counter(), {}
    for bn in sorted(own_plan["blocks"]):
        for op in own_plan["blocks"][bn]:
            if op["kind"] == "kernel" and "recorded_sequence" in op:
                t = th.get(op["recorded_sequence"])
                if t is not None:
                    launches[t] += 1
                    first.setdefault(t, op["recorded_sequence"])
    ranks = sorted([t for t, _ in launches.most_common(nr)], key=lambda t: first[t])
    if len(ranks) < nr:
        sys.exit(f"refused: {len(ranks)} launching threads for {nr} ranks")
    rank_of_thread = {t: r for r, t in enumerate(ranks)}
    other_threads = {t: n for t, n in launches.items() if t not in rank_of_thread}

    def op_rank(op):
        s = op.get("recorded_sequence")
        if s is None:
            return None
        t = th.get(s)
        if t is None:
            sys.exit(f"refused: sequence {s} ({op['kind']}) not in the trace")
        if t not in rank_of_thread and op["kind"] == "synchronization" and not any(i in dev for i in op_ids(op)):
            return None  # a context or event sync from another thread (the transport): kept in every rank
        return rank_of_thread.get(t, -1)

    users = collections.defaultdict(set)
    uses = collections.defaultdict(collections.Counter)      # block -> rank -> commands (not copies) naming it
    named = collections.defaultdict(lambda: collections.defaultdict(set))  # block -> offset -> ranks naming that offset
    unranked = collections.Counter()
    for ops in own_plan["blocks"].values():
        for op in ops:
            if op["kind"] in ("device_copy", "device_to_host"):
                # a halo DMA is issued by the receiving rank and names the sending rank's packet, and a device-to-host read
                # only reads: neither makes the block the reader's. THE BREAKAGE THIS PREVENTS: rank 0's edge field 784
                # (allocated before step 1, written by rank 0's recover_edges_f32) counted as shared because rank 1 pulls
                # its halo out of it (device_copy 784+0) and reads it to the host, so rank 1 pulled its own stale copy.
                continue
            r = op_rank(op)
            if r == -1:
                unranked[op["kind"] if op["kind"] != "kernel" else "kernel:" + op["function"]] += 1
                continue
            if r is None:
                continue
            for i in op_ids(op):
                if i in dev:
                    users[i].add(r)
            pairs = []
            slice_pairs(op, pairs)
            for i, off in pairs:
                if i in dev:
                    uses[i][r] += 1
                    named[i][off].add(r)
    owner = {i: next(iter(u)) for i, u in users.items() if len(u) == 1 and i in dev}
    # A block the forward allocates during the steps (cuMemAlloc by a rank thread's own CuPy pool, MR_RANK_POOLS=1) is
    # that rank's memory whoever reads it: the other rank reads a halo packet out of it (a two-rank plan: rank 1 copies
    # 836+2181632 -> 882 from rank 0's pool block 836) or reads it to the host (TEMPO words). Counting those readers made
    # 836 "shared", so each process kept its own stale copy of the packet and 41 allocations per rank differed after step
    # 1. The allocating thread decides; a read by another rank becomes a peer transfer below.
    alloc_owner = {}
    for bn in sorted(own_plan["blocks"]):
        for op in own_plan["blocks"][bn]:
            if op["kind"] == "allocate" and op["id"] in dev:
                rr = op_rank(op)
                if rr is not None and rr >= 0:
                    alloc_owner[op["id"]] = rr
    for i, rr in alloc_owner.items():
        owner[i] = rr
    shared = {i for i in dev if len(users.get(i, ())) != 1 and i not in alloc_owner}
    seq_rank = op_rank
    # Rank recordings (2026-11-09). A block that exists before step 1 has no allocate command, so the allocating thread is not
    # known; when another rank's command also names it (a per-card table that rank reads) it lands in `shared`. It is still
    # ONE rank's CuPy pool block (MR_RANK_POOLS=1: every rank thread allocates from its own pool, also during init), and
    # that rank keeps putting new arrays into it during the steps: on pk120-r2, block 758 (62.8 MB) is named by 1,000+
    # commands of rank 0 and by 8 table reads of rank 1, and rank 0's halo send packet lives in it (758+31811072). The
    # receiving rank's halo DMA out of that packet was kept as a local device_copy, so the rank-1 process copied its own
    # never-written copy of the block (zeros) into its halo: rho = 0 in 644 halo columns after the first unpack, 23
    # derivative guards fired at step 1 (edge_fields_from_saved_u_f32 "rho" first), and the forward itself differed from
    # the recording in 43 + 63 allocations at boundary 1 (120 km, two ranks). The pool owner
    # is the rank whose commands name the block most; a copy out of the block issued by ANOTHER rank is a peer transfer.
    pool_owner = {}  # filled below, once the commands of the sealed plan have their ranks

    def pool_copy(op):
        """(sender, receiver) when `op` is a copy out of another rank's pool block that lies in `shared`, else None."""
        if op["kind"] != "device_copy":
            return None
        s, d = op["source"]["allocation"], op["destination"]["allocation"]
        if s not in shared or s not in pool_owner:
            return None
        io = op_rank(op)  # the final op_rank (by sequence where kept, else by its rank-thread run), bound when this is called
        if io is None or io < 0:
            return None
        q, n, tot = pool_owner[s]
        if q == io:
            return None
        lo, hi = op["source"]["offset"], op["source"]["offset"] + op["bytes"]
        mine = sum(1 for off, rs in named[s].items() if lo <= off < hi and io in rs)
        theirs = sum(1 for off, rs in named[s].items() if lo <= off < hi and q in rs)
        # THE BREAKAGE THIS PREVENTS: guessing the wrong side. The issuing rank must never name the range itself, and the
        # pool owner must name it (it wrote the packet) or own nearly every reference to the block.
        if mine or not (theirs or n >= 0.9 * tot):
            sys.exit(f"refused: copy out of shared block {s}+{lo} ({op['bytes']} bytes) issued by rank {io}: the range is named by rank {io} "
                     f"{mine} times and by the pool owner (rank {q}, {n} of {tot} references to the block) {theirs} times; cannot tell whose bytes it reads")
        # the destination is the issuing rank's: an allocation it owns, or a shared block of its own pool (pk120-r8: rank 7
        # receives rank 6's packet 2250+0 into its own pool block 2995, which rank 6's table reads made "shared" too)
        if owner.get(d) != io and not (d in shared and pool_owner.get(d, (None,))[0] == io):
            sys.exit(f"refused: copy out of rank {q}'s pool block {s}+{lo} by rank {io} lands in allocation {d}, which rank {io} does not own "
                     f"(owner {owner.get(d)}, pool owner {pool_owner.get(d)})")
        return q, io

    # The sealed plan drops recorded_sequence from kernels and rewrites some uploads and library calls, so a command
    # with no sequence takes the rank of the rank-thread run it sits in: the recording's baton lets one rank thread
    # issue commands at a time, and the sealed plan keeps the recorded order, so the nearest earlier command that
    # kept its sequence (event and stream syncs, clears, copies) names the thread. A command that names owned
    # allocations must agree with that run (counted; a disagreement takes the owner). Host boundaries stay in every
    # rank. THE BREAKAGE THIS PREVENTS: a kernel naming only shared blocks (rank 1's mcica_gpoint on 786/809/782) was
    # kept in every rank process, and after step 1 41 allocations per rank differed from the one-process replay.
    run_rank, run_stats = {}, collections.Counter()
    # First, kernels and uploads matched to the partial plan (same recording, every command sequenced): the sealer keeps
    # their order and rewrites or drops only some of them, so an in-order alignment of (kind, function / destination,
    # bytes) recovers each one's own sequence, hence its thread. Only commands that do not match fall back to the run.
    import difflib
    def sig(o):
        if o["kind"] == "kernel": return ("k", o.get("function"))
        if "destination" in o and isinstance(o["destination"], dict): return (o["kind"].split("_")[-1], o["destination"].get("allocation"), o.get("bytes"))
        return (o["kind"],)
    if a.ownership_from:
        for bn in sorted(plan["blocks"]):
            A = [o for o in plan["blocks"][bn] if "recorded_sequence" not in o and o["kind"] in ("kernel", "upload")]
            Bo = [o for o in own_plan["blocks"].get(bn, []) if o["kind"] in ("kernel", "upload")]
            sm = difflib.SequenceMatcher(None, [sig(o) for o in A], [sig(o) for o in Bo], autojunk=False)
            for blk in sm.get_matching_blocks():
                for j in range(blk.size):
                    rr = seq_rank(Bo[blk.b + j])
                    if rr is not None and rr >= 0:
                        run_rank[id(A[blk.a + j])] = rr; run_stats["aligned:" + A[blk.a + j]["kind"]] += 1
    for bn in sorted(plan["blocks"]):
        last = None
        for op in plan["blocks"][bn]:
            if id(op) in run_rank:
                rs = {owner[i] for i in op_ids(op) if i in owner}
                if rs and rs != {run_rank[id(op)]}:
                    run_stats["aligned_disagrees_with_owner:" + op["kind"]] += 1
                last = run_rank[id(op)]
                continue
            if "recorded_sequence" in op:
                rr = seq_rank(op)
                if rr is not None and rr >= 0:
                    last = rr
                continue
            if op["kind"] in ("host_boundary",):
                continue
            rs = {owner[i] for i in op_ids(op) if i in owner}
            if len(rs) == 1:
                o = next(iter(rs))
                if last is not None and o != last:
                    run_stats["owner_disagrees_with_run:" + op["kind"]] += 1
                run_rank[id(op)] = o
            elif len(rs) > 1:
                run_rank[id(op)] = -2
            elif last is not None:
                run_rank[id(op)] = last; run_stats["by_run:" + op["kind"]] += 1
            else:
                run_stats["no_run_kept_everywhere:" + op["kind"]] += 1

    def op_rank(op):  # noqa: F811 - the sealed plan: by sequence where kept, else by its rank-thread run
        if "recorded_sequence" in op or not a.ownership_from:
            return seq_rank(op)
        return run_rank.get(id(op))
    # Rank recordings: the sealed plan names more than the partial one (the sealer resolves the pointer words of a table upload,
    # and a halo pack names its send packet only through such a table: pk120-r4 block 1127, offsets 0, 58528, ...), so
    # the references that decide a pool owner are counted over the sealed plan too.
    for bn in sorted(plan["blocks"]):
        for op in plan["blocks"][bn]:
            if op["kind"] in ("device_copy", "device_to_host"):
                continue
            rr = op_rank(op)
            if rr is None or rr < 0:
                continue
            pairs = []
            slice_pairs(op, pairs)
            for i, off in pairs:
                if i in dev:
                    uses[i][rr] += 1
                    named[i][off].add(rr)
    for i in shared:
        c = uses.get(i)
        if c:
            (q, n), tot = c.most_common(1)[0], sum(c.values())
            pool_owner[i] = (q, n, tot)
    # Cross-rank copies and commands that name another rank's owned allocation.
    cross, foreign = 0, collections.Counter()
    pool_cross = 0
    # A kernel of rank r that names rank o's block reads a per-card table rank o made (a two-rank plan: rank 1's
    # mpas_lw_mcica_gpoint reads the gpoint table rank 0 uploaded into its pool block 809). Rank r's process then gets a
    # MIRROR of that block: the allocation, and every upload into it in recorded order (host bytes, so they replay in any
    # process), so the table rank r reads is the bytes the one-process forward holds. The mirror is not hash-checked in
    # rank r (rank o checks the block). THE BREAKAGE THIS PREVENTS: the split refused, "commands name another rank's
    # owned allocation: kernel:mpas_lw_mcica_gpoint 4".
    mirror, mirror_why = collections.defaultdict(set), collections.defaultdict(set)
    for ops in plan["blocks"].values():
        for op in ops:
            if op["kind"] == "device_copy":
                s, d = owner.get(op["source"]["allocation"]), owner.get(op["destination"]["allocation"])
                if s is not None and d is not None and s != d:
                    cross += 1
                elif pool_copy(op):
                    cross += 1; pool_cross += 1
                continue
            r = op_rank(op)
            if r == -2:
                foreign["ties two ranks: " + (op["kind"] if op["kind"] != "kernel" else "kernel:" + op["function"])] += 1
                continue
            if r is None or r == -1:
                continue
            for i in op_ids(op):
                if i in owner and owner[i] != r:
                    if op["kind"] == "kernel":
                        mirror[i].add(r); mirror_why[i].add(op["function"])
                    else:
                        foreign[op["kind"]] += 1
    # Rank recordings (2026-11-09): one-time module initializers. A kernel the whole plan launches ONCE, on one thread, with a
    # [1,1,1] grid and only plain arguments, is a table builder that fills its module's device globals at first use
    # (mynnsfc_tables: the MYNN surface-layer lookup tables; tempo_warp_sizes: an image constant). In the recording one
    # rank thread ran it and every rank thread then used the module; with one process per rank each process loads its
    # own copy of the module, and only the launching rank's copy had its tables. On pk120-r2 the kernel's scratch
    # argument is a block only rank 0 names (122+512), so the shared-block rule below did not replicate it: every
    # column of rank 1 differed from the recording after one step (surface fluxes from zero tables)
    # and rank 0 differed near the seam. Such a kernel runs in every rank process, in
    # place; the other ranks get a mirror of the block it names (never hash-checked there).
    by_fn = collections.defaultdict(list)
    for bn in sorted(plan["blocks"]):
        for op in plan["blocks"][bn]:
            if op["kind"] == "kernel":
                by_fn[op["function"]].append(op)
    init_once = {}
    for fn, lst in by_fn.items():
        op = lst[0]
        if len(lst) != 1 or op.get("grid") != [1, 1, 1] or fn in POINTER_TABLE_KERNELS:
            continue
        rr = op_rank(op)
        if rr is None or rr < 0:
            continue
        ids = [i for i in op_ids(op) if i in dev]
        if any(g.get("kind") not in ("device", "bytes") for g in op.get("arguments", [])):
            continue  # frames and bound tables are not plain memory: left to the rules above
        if any(i in owner and owner[i] != rr for i in ids) or any(i not in owner and i not in shared for i in ids):
            continue
        init_once[id(op)] = fn
        for i in ids:
            if i in owner:
                for q in range(nr):
                    if q != rr:
                        mirror[i].add(q); mirror_why[i].add(fn)
    # Host dependencies across ranks.
    writers, readers = collections.defaultdict(set), collections.defaultdict(set)
    for ops in plan["blocks"].values():
        for op in ops:
            r = op_rank(op)
            rs = set(range(nr)) if r is None or r == -1 else {r}
            for h in (i for i in op_ids(op) if i in host):
                (writers if op["kind"] == "host_publish" else readers)[h] |= rs
    host_cross = {h: {"writers": sorted(writers[h]), "readers": sorted(readers[h])} for h in writers if readers[h] - writers[h]}
    host_users = collections.defaultdict(set)
    for ops in plan["blocks"].values():
        for op in ops:
            rr = op_rank(op)
            if rr is not None and rr >= 0:
                for h in (i for i in op_ids(op) if i in host):
                    host_users[h].add(rr)
    per = [sum(dev[i] for i in owner if owner[i] == r) for r in range(nr)]
    census = {"allocations": len(dev), "rank_threads": ranks, "other_launching_threads": other_threads,
              "commands_on_other_threads": dict(unranked), "owned_allocations": [sum(1 for i in owner if owner[i] == r) for r in range(nr)],
              "owned_mib": [round(b / 2**20, 1) for b in per], "shared_allocations": len(shared),
              "shared_mib": round(sum(dev[i] for i in shared) / 2**20, 1), "cross_rank_copies": cross,
              "cross_rank_copies_out_of_a_pool_block": pool_cross,
              "commands_naming_another_ranks_allocation": dict(foreign.most_common(20)), "host_cross_rank": host_cross}
    census["rank_separable"] = not foreign and not host_cross and not unranked and cross > 0
    print(json.dumps({"event": "split_census", "plan": src, "ranks": nr, **census}, indent=1))
    if a.analyze:
        return
    if foreign:
        sys.exit(f"refused: commands name another rank's owned allocation: {dict(foreign.most_common(5))}")
    if host_cross:
        sys.exit(f"refused: host allocations written from one rank's device data and read by another: {host_cross}")
    if unranked:
        sys.exit(f"refused: commands on threads that are not rank threads: {dict(unranked)}")
    if not a.out:
        sys.exit("--out needed")
    # Events by creator. A CUDA event handle is reused once destroyed, and the rank threads create, record and destroy
    # their own; a wait (StreamWaitEvent / EventSynchronize) or record of an event another rank created ties two ranks.
    # In one process per rank that event does not exist, and the only data that crosses ranks crosses through a
    # PeerSend/PeerRecv pair (the receive blocks until the sender's bytes, produced after its own context sync, exist),
    # so the op is kept only in the creating rank. THE BREAKAGE THIS PREVENTS: both rank plans refused at step 1,
    # "unknown event 125449055086192" (rank 0 waited on a handle rank 1 created).
    ev_keep, ev_owner, ev_cross = {}, {}, collections.Counter()
    for e in plan.get("events", []):
        ev_owner[e["id"]] = None
    for bn in sorted(plan["blocks"]):
        for k, op in enumerate(plan["blocks"][bn]):
            h = op.get("event_handle_recorded") if op["kind"] == "synchronization" else None
            if h is None:
                continue
            orank = op_rank(op)
            orank = None if orank in (-1, -2) else orank
            if op.get("operation") == "EventCreate":
                ev_owner[h] = orank
                continue
            o = ev_owner.get(h)
            if o is not None:
                if orank is not None and orank != o:
                    ev_cross[op.get("operation")] += 1
                    # a destroy by another rank thread (CuPy frees an event on whichever thread drops it) still ends the
                    # creator's event, so the handle can be created again there; a cross-rank wait is dropped
                    ev_keep[(bn, k)] = {o} if op.get("operation") == "EventDestroy" else set()
                else:
                    ev_keep[(bn, k)] = {o}
            if op.get("operation") == "EventDestroy":
                ev_owner.pop(h, None)
    copies = [(bn, k) for bn in sorted(plan["blocks"]) for k, op in enumerate(plan["blocks"][bn]) if op["kind"] == "device_copy"]
    channel = {c: n for n, c in enumerate(copies)}
    ref = plan.get("reference")
    refrows = [json.loads(l) for l in open(os.path.join(a.plan_dir, ref)) if l.strip()] if ref else []
    # Rank recordings at 60 km (fix E). The forecast's TEMPO pointer-table dictionary is one per PROCESS and empties itself at 64
    # entries (tempo_v303.py _pointer_array); the runner copies that per process (tempo.rs Cache, main.rs bind_tempo).
    # A rank process alone never reaches 64 at the recorded moment, so it reuses, or collides with, tables the recording
    # had dropped and uploaded again. THE BREAKAGE THIS PREVENTS: a plan of x1.163842 on 8 ranks, run as 8 rank processes,
    # step 2: CUDA 700 in tempo_from (ranks 0, 3, 4) and "TEMPO target overlaps a live binding" (ranks 5, 7), on
    # 60 km at eight ranks. Each kept table therefore carries the number of tables the OTHER
    # ranks put into the dictionary around it (foreign_before, foreign_after); the runner counts them as entries.
    # Here: the one-process dictionary replayed over the unsplit plan (key = the table's views), so that only tables
    # that really entered it (misses) are counted.
    def tempo_key(op):
        # the runner keys the dictionary by the device and the table words, which are the addresses of the views
        return (op.get("device", 0), tuple(None if e is None else (e["target"]["allocation"], e["target"]["offset"])
                                           for e in op["entries"]))
    tempo_miss = {}
    if nr > 1:
        entries = set()
        for bn in sorted(plan["blocks"]):
            seen = set()
            for k, op in enumerate(plan["blocks"][bn]):
                kd = op["kind"]
                if kd in ("tempo_named_table_bind", "tempo_table_release") or (
                        kd == "kernel" and any(g.get("kind") == "tempo_table" for g in op.get("arguments", []))):
                    # THE BREAKAGE THIS PREVENTS: a count applied after a bind could empty the dictionary while a
                    # named binding of this rank is still in use ("unknown live TEMPO table binding"); no package has one.
                    sys.exit(f"refused: {bn}/{k} {kd}: named TEMPO bindings are not carried across rank processes")
                if kd != "tempo_pointer_table_bind":
                    continue
                if op["binding"] in seen:
                    sys.exit(f"refused: {bn}: TEMPO binding {op['binding']} is made twice in one block")
                seen.add(op["binding"])
                key = tempo_key(op)
                miss = key not in entries
                if miss:
                    if len(entries) >= 64:
                        entries.clear()
                    entries.add(key)
                tempo_miss[(bn, op["binding"])] = miss
    # Two cards at 3 km (2026-11-13): the functions each rank launches itself. A kernel of rank q that names only shared blocks
    # is copied into rank r below only when rank r never launches that function: then it is a table builder whose output
    # rank r reads (mynnsfc_tables). When rank r launches it too, rank q's launch is rank q's own work on rank q's columns
    # through descriptors in a shared block. THE BREAKAGE THIS PREVENTS: on the 3 km storm box (regional, 2 ranks) the
    # solve-region scrub, halo-copy and CuPy index kernels keep their descriptors in blocks both rank threads used (103,
    # 117); 1,516 of rank 0's launches were copied into rank 1, and rank 1 stopped with CUDA 700 at its first step in
    # rank 0's ruc_land_tile_preflight and then its solve_region_scrub_columns_v841.
    own_fns = collections.defaultdict(set)
    for ops in plan["blocks"].values():
        for op in ops:
            if op["kind"] == "kernel":
                rr = op_rank(op)
                if rr is not None and rr >= 0:
                    own_fns[rr].add(op["function"])
    for r in range(nr):
        mirrored = {i for i, rs in mirror.items() if r in rs}
        mine = {i for i in owner if owner[i] == r} | shared | mirrored
        out = dict(plan)
        # Pointer words of a kept allocation (initial relocations) or held upload (pointers) that name an allocation this
        # rank does not hold are dropped: the word keeps its recorded value, which this rank never dereferences (no command
        # of this rank names that allocation; the check above refuses one that does). THE BREAKAGE THIS PREVENTS: the
        # runner refused both rank plans, "relocation target has no recorded base" (shared TEMPO scratch 569 holds
        # rank-1 pointer words; 486/198 hold the other rank's tables and no command reads them).
        dropped = collections.Counter()
        out["allocations"] = []
        for x in plan["allocations"]:
            if x["id"] not in mine:
                continue
            if x.get("relocations"):
                keepr = [rl for rl in x["relocations"] if rl["target"]["allocation"] in mine]
                if len(keepr) != len(x["relocations"]):
                    dropped["relocation:" + ("owned" if x["id"] in owner else "shared")] += len(x["relocations"]) - len(keepr)
                    x = dict(x); x["relocations"] = keepr
            out["allocations"].append(x)
        blocks, stats = {}, collections.Counter()
        for bn, ops in plan["blocks"].items():
            keep = []
            for k, op in enumerate(ops):
                if op["kind"] == "device_copy":
                    s, d = op["source"]["allocation"], op["destination"]["allocation"]
                    rs, rd = owner.get(s), owner.get(d)
                    pc = pool_copy(op)
                    if pc:
                        q, io = pc
                        if q == r:
                            keep.append({"kind": "peer_send", "stream": op["stream"], "channel": channel[(bn, k)], "peer": io,
                                         "source": op["source"], "bytes": op["bytes"]}); stats["peer_send_pool"] += 1
                        elif io == r:
                            keep.append({"kind": "peer_recv", "stream": op["stream"], "channel": channel[(bn, k)], "peer": q,
                                         "destination": op["destination"], "bytes": op["bytes"]}); stats["peer_recv_pool"] += 1
                        continue
                    if rs is not None and rd is not None and rs != rd:
                        if rs == r:
                            keep.append({"kind": "peer_send", "stream": op["stream"], "channel": channel[(bn, k)], "peer": rd,
                                         "source": op["source"], "bytes": op["bytes"]}); stats["peer_send"] += 1
                        elif rd == r:
                            keep.append({"kind": "peer_recv", "stream": op["stream"], "channel": channel[(bn, k)], "peer": rs,
                                         "destination": op["destination"], "bytes": op["bytes"]}); stats["peer_recv"] += 1
                        continue
                    if rs is not None and d in shared:
                        # A copy from rank rs's data into a shared block (a per-card staging or full-mesh buffer the rank
                        # threads take in turn; a two-rank plan: 874 -> 836 issued by rank 0 from rank 1's packet, 794 -> 900
                        # issued by rank 1). The bytes exist only in rank rs's process, and every rank may read that
                        # range afterwards (836 -> 882 on rank 1). So rank rs makes the copy itself and sends the range to
                        # every other rank, which receives it into its own copy of the shared block: every process holds
                        # the block as the one-process forward has it at this point. One point-to-point channel per
                        # receiver (channel = base + 1,000,000 x (receiver + 1)), so each cotangent comes back on its own
                        # key. THE BREAKAGE THIS PREVENTS: rank 0 refused step 1, "native submit: MissingAllocation(874)".
                        if rs == r:
                            keep.append(op); stats["shared_copy_local"] += 1
                            for q in range(nr):
                                if q != r:
                                    keep.append({"kind": "peer_send", "stream": op["stream"], "channel": channel[(bn, k)] + 1000000 * (q + 1),
                                                 "peer": q, "source": op["source"], "bytes": op["bytes"]}); stats["peer_send_shared"] += 1
                        else:
                            keep.append({"kind": "peer_recv", "stream": op["stream"], "channel": channel[(bn, k)] + 1000000 * (r + 1),
                                         "peer": rs, "destination": op["destination"], "bytes": op["bytes"]}); stats["peer_recv_shared"] += 1
                        continue
                    if rs is not None and rd is None and op_rank(op) not in (None, rs):
                        sys.exit(f"refused: {bn}/{k}: rank {op_rank(op)} copies rank {rs}'s allocation {s} into {d}, owned by no rank and not shared")
                if op["kind"] == "device_to_host":
                    so = owner.get(op["source"]["allocation"]); io = op_rank(op)
                    if so is not None and io is not None and io >= 0 and so != io:
                        if op.get("host_consumer") not in (None, "unresolved"):
                            sys.exit(f"refused: {bn}/{k}: rank {io} reads rank {so}'s allocation to the host for {op.get('host_consumer')}")
                        # rank io read rank so's words to the host only for a check (host consumer unresolved: the
                        # observer's TEMPO word capture); that memory is not in rank io's process, so the read is dropped
                        stats["d2h_other_rank_dropped"] += 1
                        continue
                if (bn, k) in ev_keep:
                    if r in ev_keep[(bn, k)]:
                        keep.append(op); stats["kept_event_creator"] += 1
                    else:
                        stats["event_other_rank"] += 1
                    continue
                orank = op_rank(op)
                if orank is not None and orank >= 0 and orank != r and op["kind"].endswith("upload") and                         set(op_ids(op)) and set(op_ids(op)) <= shared:
                    # host bytes into a block every rank holds: replayed in every process, so each copy of the block holds
                    # what the one-process forward holds (a per-card table one rank uploads and another rank reads)
                    keep.append(op); stats["shared_upload_replicated"] += 1
                    continue
                if orank is not None and orank >= 0 and orank != r and op["kind"] == "kernel" and                         set(op_ids(op)) <= shared and op.get("function") not in POINTER_TABLE_KERNELS and                         op.get("function") not in own_fns[r] and                         not any(g.get("kind") == "tempo_table" for g in op.get("arguments", [])):
                    # A kernel that names only blocks every rank holds (a table builder: mynnsfc_tables fills the MYNN
                    # surface-layer lookup tables in its module's globals and the shared block 782) runs in every
                    # process: its module and those blocks exist in each one, and the other ranks read what it wrote.
                    # THE BREAKAGE THIS PREVENTS: rank 1's mynnsfc_columns (step 1 command 4102) read tables only rank 0's
                    # process had built, and 15,260 words of rank 1's block 875 differed from the one-process forward.
                    keep.append(op); stats["shared_kernel_replicated"] += 1; stats["shared_kernel_replicated:" + op["function"]] += 1
                    continue
                if id(op) in init_once and orank is not None and orank >= 0 and orank != r:
                    keep.append(op); stats["init_kernel_replicated"] += 1
                    continue
                if mirrored and orank is not None and orank != r and (
                        (op["kind"] in ("allocate", "free") and op.get("id") in mirrored) or
                        (op["kind"].endswith("upload") and set(op_ids(op)) & mirrored and set(op_ids(op)) <= mine)):
                    keep.append(op); stats["mirror_" + op["kind"]] += 1
                    continue
                if op["kind"] in ("allocate", "free") and op["id"] in shared:
                    orank = None  # a shared block exists in every rank process (made in place, by whichever rank's pool)
                if orank is None or orank == r:
                    if op.get("pointers"):
                        kp = [q for q in op["pointers"] if q["target"]["allocation"] in mine]
                        if len(kp) != len(op["pointers"]):
                            dsts = {g["destination"]["allocation"] for g in op.get("ranges", [])}
                            dropped["held_pointer:" + ("owned" if dsts & set(owner) else "shared")] += len(op["pointers"]) - len(kp)
                            op = dict(op); op["pointers"] = kp
                    keep.append(op); stats["kept" if orank is not None else "kept_all_ranks"] += 1
                else:
                    stats["other_rank"] += 1
            binds = [o for o in ops if o["kind"] == "tempo_pointer_table_bind"] if nr > 1 else []
            if binds:
                own = {o["binding"] for o in keep if o["kind"] == "tempo_pointer_table_bind"}
                if not own:
                    if any(tempo_miss[(bn, o["binding"])] for o in binds):
                        sys.exit(f"refused: block {bn}: other ranks put TEMPO tables into the process-wide 64-entry "
                                 f"dictionary and rank {r} has no table here to carry the count")
                else:
                    cnt, n, last = {}, 0, None
                    for o in binds:
                        if o["binding"] in own:
                            cnt[o["binding"]] = [n, 0]; n = 0; last = o["binding"]
                        elif tempo_miss[(bn, o["binding"])]:
                            n += 1
                    cnt[last][1] = n
                    keep = [dict(o, foreign_before=cnt[o["binding"]][0], foreign_after=cnt[o["binding"]][1])
                            if o["kind"] == "tempo_pointer_table_bind" else o for o in keep]
                    stats["tempo_tables"] += len(own)
                    stats["tempo_tables_of_other_ranks_counted"] += sum(x + y for x, y in cnt.values())
            if not keep:
                sys.exit(f"refused: block {bn} is empty on rank {r}")
            blocks[bn] = keep
        if tempo_miss:
            entries, foreign, clears = set(), 0, 0
            for bn in sorted(blocks):
                for o in blocks[bn]:
                    if o["kind"] != "tempo_pointer_table_bind":
                        continue
                    for leg in ("foreign_before", None, "foreign_after"):
                        if leg is None:
                            key = tempo_key(o)
                            miss = key not in entries
                            if miss:
                                if len(entries) + foreign >= 64:
                                    entries.clear(); foreign = 0; clears += 1
                                entries.add(key)
                            if miss != tempo_miss[(bn, o["binding"])]:
                                sys.exit(f"refused: rank {r} {bn} {o['binding']}: the rank process would "
                                         f"{'upload' if miss else 'reuse'} this TEMPO table where the one-process forward "
                                         f"did the opposite (the foreign-table counts do not reproduce its dictionary)")
                            continue
                        for _ in range(o[leg]):
                            if len(entries) + foreign >= 64:
                                entries.clear(); foreign = 0; clears += 1
                            foreign += 1
            stats["tempo_dictionary_clears"] = clears
        out["blocks"] = blocks
        checked = {i for i in owner if owner[i] == r}
        out["carried"] = [i for i in plan["carried"] if i in mine]
        if plan.get("carried_views") is not None:
            out["carried_views"] = [[v for v in fr if v["target"]["allocation"] in checked] for fr in plan["carried_views"]]
        if plan.get("boundaries"):
            out["boundaries"] = [{"live": [v for v in b["live"] if v["target"]["allocation"] in mine], "host_live": b["host_live"]} for b in plan["boundaries"]]
        out["recorded_bases"] = [x for x in plan.get("recorded_bases", []) if x[0] in mine]
        out["excluded_residue"] = [x for x in plan.get("excluded_residue", []) if x["allocation"] in checked]
        out["address_bearing"] = [x for x in plan.get("address_bearing", []) if x in mine]
        out["host_carried"] = [h for h in plan.get("host_carried", []) if writers.get(h, set()) <= {r}]
        # Host owners every rank's commands use (the RRFS copy-plan caches 900100-900116, the host frame 900001) stay
        # carried in each process, but each process's copy reflects only its own rank's calls (a cache hit in one process
        # is a miss in the one-process forward), so the runner logs a differing hash for them (peer_host_unchecked) instead of
        # stopping; device words are checked. THE BREAKAGE THIS PREVENTS: rank 0 stopped at step 1, "host state hash mismatch at step 1, allocation
        # 900100", on a copy-plan cache rank 1's calls also wrote in the one-process forward.
        host_unchecked = sorted(h for h in out["host_carried"] if len(host_users.get(h, ())) > 1)
        # The reference checks owned allocations only; a shared one is still carried (stored and restored by the window)
        # with the hash its own process computes at its first pass recorded as its reference (peer_unchecked).
        out["peer_unchecked"] = sorted(i for i in out["carried"] if i not in checked)
        out["carried"] = [i for i in out["carried"] if i in checked]
        # F232 poison claims name an allocate op of a block; another rank's allocate is not in this plan (THE BREAKAGE THIS
        # PREVENTS: the runner refused rank 0's plan, "poison claim step_01/857 names no overwritten-before-read allocate op").
        if plan.get("poison_claims") is not None:
            out["poison_claims"] = [c for c in plan["poison_claims"] if c["id"] in mine]
        if a.global_arena:
            out["peer_arena_sizes"] = [[i, dev[i]] for i in sorted(dev)]
        out["peer_split"] = {"source_plan": src, "source_sha256": hashlib.sha256(open(src, "rb").read()).hexdigest(),
                             "trace": a.trace, "rank": r, "ranks": nr, "rank_thread": ranks[r], "owned": len(checked),
                             "shared": sorted(shared), "mirrored": {str(i): sorted(mirror_why[i]) for i in sorted(mirrored)}, "init_kernels_in_every_rank": sorted(init_once.values()), "commands": dict(stats), "dropped_pointer_words": dict(dropped), "cross_rank_event_ops_dropped": dict(ev_cross), "rank_by_run": dict(run_stats)}
        d = f"{a.out}-r{r}"
        os.makedirs(d, exist_ok=True)
        for e in os.listdir(a.plan_dir):
            if e in (a.plan_name, os.path.basename(ref or "")) or os.path.lexists(os.path.join(d, e)):
                continue
            os.symlink(os.path.abspath(os.path.join(a.plan_dir, e)), os.path.join(d, e))
        if ref:
            keep_ids = set(out["carried"]) | set(out["host_carried"])  # host_unchecked rows stay: the runner logs those
            out["peer_host_unchecked"] = host_unchecked
            with open(os.path.join(d, os.path.basename(ref)), "w") as f:
                for row in refrows:
                    if row["allocation"] in keep_ids:
                        f.write(json.dumps(row) + "\n")
            out["reference"] = os.path.basename(ref)
        with open(os.path.join(d, a.plan_name), "w") as f:
            json.dump(out, f)
        print(json.dumps({"event": "split_rank", "rank": r, "dir": d, "dropped_pointer_words": dict(dropped), "allocations": len(out["allocations"]), "owned": len(checked),
                          "carried_checked": len(out["carried"]), "host_carried": len(out["host_carried"]), "commands": dict(stats)}))


if __name__ == "__main__":
    main()
