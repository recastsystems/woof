# SPDX-License-Identifier: Apache-2.0
"""ad-record: per-mesh step<=1 upload retention candidates from a recorded pair (the hrrrv5-v25 list's recipe).

The recorder retains an upload's bytes only when they are small or their sha256 is on the candidates list, and the
sealer admits a retained step-1 upload as immutable input (or a source literal / zero clear) by that sha. The v25 list
was made on x1.2562, so on another mesh every column-sized constant upload has a sha the list cannot know: the first
x1.40962 pair (pk120-1c, 2026-11-07) refused with 331 'upload_needs_live_rule' words (163,848- and 9.8 MB column
fields from device_pools_v842 prepared/field, TEMPO _session_tables, RUC tables, glue plans).

A candidate is a HostUploadBytes record at step <= 1 whose (sha256, bytes) occurs equally often in both recordings of
the pair; a pointer table (device addresses) differs between the two recordings and so never qualifies. Retention
candidates only, not certification: the sealer still classifies every retained upload.

RESIDUAL mode (record-mem031, 2026-11-07): "a pointer table differs between the two recordings" is not true when both
recordings get the same device address layout (two fresh processes, one per card, allocate deterministically). The
mem031 1-step pair (pk31-1) gave 88 'equal' uploads, one of them the RRFS copy-plan table (2,304 B, sha 273f8010):
listed, the sealer admitted it as immutable input with its recorded device pointers instead of the live
rrfs_copy_plan_upload rule, and the 18-step plan pk31 replayed into another state from step 1 (RUC land and ice
fatal flags, every dycore allocation different at boundary 1). With RESIDUAL (the stage2-residual.json of the same pair
sealed with BASE_LIST), only the uploads the sealer named 'upload_needs_live_rule' are added: the recorded
sequence numbers are looked up in A's trace, and each must occur equally often in B.
Usage: gen_immutable.py RECORDER_A RECORDER_B BASE_LIST OUT_LIST LABEL [RESIDUAL] [--equal-only]"""
import collections, glob, gzip, json, sys

def by_sequence(recorder, wanted):
    """The upload bytes record of each recorded MemcpyHtoD sequence the sealer names: the HostUploadBytes record with
    that call's host source and size nearest to it in the trace."""
    path = glob.glob(f"{recorder}/driver-*.jsonl.gz")
    calls, recs = {}, []
    with gzip.open(path[0], "rt") as f:
        for line in f:
            if '"HostUploadBytes"' in line:
                d = json.loads(line)
                recs.append((int(d["sequence"]), int(d["source"]), int(d["bytes"]), d.get("sha256")))
            elif '"type":"MemcpyHtoD' in line:
                d = json.loads(line)
                if int(d["sequence"]) in wanted:
                    calls[int(d["sequence"])] = (int(d["b"]), int(d["count"]))
    out = {}
    for q in wanted:
        if q not in calls:
            sys.exit(f"residual sequence {q}: no MemcpyHtoD record in {path[0]}")
        src, n = calls[q]
        hits = sorted((abs(r[0] - q), r) for r in recs if r[1] == src and r[2] == n)
        if not hits:
            sys.exit(f"residual sequence {q}: no HostUploadBytes record for host {src:#x}, {n} bytes")
        out[q] = (hits[0][1][3], n)
    return out

def uploads(recorder, max_step=1):
    c = collections.Counter()
    path = glob.glob(f"{recorder}/driver-*.jsonl.gz")
    if len(path) != 1:
        sys.exit(f"{recorder}: {len(path)} driver traces")
    with gzip.open(path[0], "rt") as f:
        for line in f:
            if '"HostUploadBytes"' not in line:
                continue
            d = json.loads(line)
            if int(d.get("step", 99)) <= max_step and d.get("sha256"):
                c[(d["sha256"], int(d["bytes"]))] += 1
    return c

equal_only = "--equal-only" in sys.argv
sys.argv = [x for x in sys.argv if x != "--equal-only"]
skipped = 0
a, b = uploads(sys.argv[1]), uploads(sys.argv[2])
base = {l.split()[0] for l in open(sys.argv[3]) if l.strip() and not l.startswith("#")}
both = sorted(k for k in set(a) & set(b) if a[k] == b[k] and k[0] not in base)
if len(sys.argv) > 6:
    need = json.load(open(sys.argv[6]))["residual"].get("upload_needs_live_rule", [])
    seq = by_sequence(sys.argv[1], {int(row[2]) for row in need})
    named = set(seq.values())
    # A named upload after step 1 (a first-use buffer, e.g. the resident output diagnostics' initial words, uploaded
    # at the first output step) is compared over the whole recording: equal sha and count in both.
    a_all, b_all = uploads(sys.argv[1], 1 << 30), uploads(sys.argv[2], 1 << 30)
    equal_any = {k for k in set(a_all) & set(b_all) if a_all[k] == b_all[k]}
    missing = [k for k in named if k not in set(both) and k not in equal_any and k[0] not in base]
    if (missing or any(k[0] is None for k in named)) and not equal_only:
        sys.exit(f"residual uploads not equal in both recordings (or without sha): {missing}")
    # --equal-only (woof.hex.ad.record): list the named uploads equal in both recordings and leave the others to the
    # seal, which still refuses them (a pointer table that differs between recordings is never immutable input).
    both = sorted(k for k in named if k[0] is not None and k[0] not in base and k not in set(missing))
    skipped = len(missing)
only = sum(v for k, v in (a - b).items())
with open(sys.argv[4], "w") as out:
    out.write(open(sys.argv[3]).read())
    # The tracer reads this file with fgets(line, 256): every line must stay under 255 characters, or the rest of a
    # long comment is parsed as an entry and the recording aborts (rc 134, pk120-1d 2026-11-07).
    for text in (f"# recorder {sys.argv[5]}: {len(both)} step<=1 uploads equal in both recordings of the pair;",
                 f"# {only} uploads of A differ from B (pointer tables etc.) and are not listed.",
                 f"# A: {sys.argv[1]}", f"# B: {sys.argv[2]}"):
        out.write(text[:250] + "\n")
    for sha, n in both:
        out.write(f"{sha} {n}\n")
print(json.dumps({"added": len(both), "named_not_equal_skipped": skipped, "a_step_le1": sum(a.values()), "b_step_le1": sum(b.values()), "a_only": only}))
