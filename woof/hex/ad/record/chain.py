# SPDX-License-Identifier: Apache-2.0
"""Record a hex trajectory and seal it into a replayable plan package (the tape of the differentiable engine).

    record   the production hex forward, unchanged, run twice under the CUPTI recorder (recording A and B: two
             processes, the same inputs) after one untraced warm forward that fills the resource, kernel and CuPy
             caches. Radiation and land are recorded live; every boundary's payload is kept.
    seal     A is analysed against B: catalog, launch correlation, linearisation, library audits, partial plan, the
             device-to-host classification, stage 2 (every launch argument and upload explained by a rule, a source
             literal, an immutable input or a proved CIndexer pair) and stage 3 (frames and per-step reference).
    retain   the per-run retention list. The recorder keeps an upload's bytes only when they are small or their
             sha256 is listed, and the sealer admits a retained step<=1 upload as immutable input by that sha. A list
             made on one date cannot know another date's column constants, so the seal names them
             ('upload_needs_live_rule'); the uploads it names that are equal in both recordings are added to this
             run's own list and the pair is recorded again with it, once. THE BREAKAGE THIS PREVENTS: every recording
             of a new start date refused at seal with {'cindexer_scratch_proof': 1, 'upload_needs_live_rule': 7}
             while a queue retried the same job on the same card (learn-the-model F2, 2026-11-09).
    pairs    a residual of unproved CIndexer scratch is proved from this recording (prove_cindexer_pairs.py, at most
             two rounds) into this package's own pairs file; any other residual is a refusal.
    package  finalize_package.py + p1fix_package.py, the allocation map, the retention list and pairs used, and
             record-receipt.json (every stage's wall time, the numerics label, the libraries and their sha256).
    replay   the strict replay of the new plan (rc 0, every boundary hash equal to its own reference).

Every refusal is one 'DEFECT: woof.hex.ad.record: ...' line and exit 3; there is no fallback to an older package and no
unchecked replay. Numerics: fast is the default, strict the named proof run; the label is in the receipt.
"""
from __future__ import annotations

import concurrent.futures as futures
import gzip
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import build as native
from . import coverage
from . import env
from . import lam
from . import route as routes

#: residual classes a sealed package may keep: recorded for check only / allocation first use (declared, not state)
ALLOWED_RESIDUAL = {"radii_recorded_for_check_only", "allocation_first_use_not_whole_write"}
#: physics suites the sealer has a launch catalog for (catalog_v10.py --config)
CATALOG_SUITES = {"hrrrv5": "hrrrv5", "convection_permitting": "convection_permitting"}


class Run:
    """One recording work directory and its log."""

    def __init__(self, work: Path):
        self.work = work
        work.mkdir(parents=True, exist_ok=True)
        self.logf = open(work / "record.log", "a", buffering=1)
        self.wall: dict[str, float] = {}

    def log(self, message: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} [hexad-record] {message}"
        print(line, flush=True)
        self.logf.write(line + "\n")

    def stage(self, name: str, t0: float) -> None:
        self.wall[name] = round(self.wall.get(name, 0.0) + time.monotonic() - t0, 1)

    def fail(self, message: str) -> "NoReturn":  # noqa: F821
        self.logf.write(f"DEFECT: {message}\n")
        env.defect(message)


def _py(script: str) -> list[str]:
    return [sys.executable, "-B", str(env.TOOLS / script)]


def _run(cmd: list[str], log: Path, *, cwd: Path, environ: dict | None = None, stdin=None, stdout=None) -> int:
    with open(log, "w") as err:
        done = subprocess.run(cmd, cwd=cwd, env=environ, stdin=stdin, stdout=stdout or err, stderr=err)
    return done.returncode


def _tail(path: Path, n: int = 1, width: int = 400) -> str:
    try:
        lines = [l for l in path.read_text(errors="replace").splitlines() if l.strip()]
    except OSError:
        return f"(no {path.name})"
    return " | ".join(lines[-n:])[-width:]


# ------------------------------------------------------------------------------------------------------------------
# record
# ------------------------------------------------------------------------------------------------------------------

#: The physics mode a recording is of. The derivative rows the window runs are of this arithmetic and no other.
RECORD_PHYSICS_MODE = "exact"


def forwarded_physics_mode(words) -> str | None:
    """The physics mode forecast words end on (``--physics-mode X`` or ``--physics-mode=X``, the last one wins as in
    the forecast door's parser), or None when none names one; a word with no value gives ``""``."""
    mode = None
    words = [str(w) for w in (words or [])]
    for i, word in enumerate(words):
        if word == "--physics-mode":
            mode = words[i + 1] if i + 1 < len(words) else ""
        elif word.startswith("--physics-mode="):
            mode = word.split("=", 1)[1]
    return mode


def refuse_other_physics_mode(extra) -> None:
    """The words after ``--`` go to the forecast after the recorder's own ``--physics-mode exact``, so a mode word
    among them would win. THE BREAKAGE THIS PREVENTS: ``record ... -- --physics-mode correct`` recorded the
    correct-mode forecast, wrote ``"physics_mode": "exact"`` in its receipt, and the window ran the exact-mode
    derivative rows on it: a gradient of another model that the tangent-against-adjoint check cannot see."""
    mode = forwarded_physics_mode(extra)
    if mode is not None and mode != RECORD_PHYSICS_MODE:
        env.defect(f"record: --physics-mode {mode or '(no value)'} after --: a recording is of the "
                   f"{RECORD_PHYSICS_MODE}-mode forecast, because the derivative rows the window runs are of that "
                   "arithmetic and this build has none of the corrected schemes. Nothing was recorded. Leave the "
                   "word out")


def forecast_args(o) -> list[str]:
    args = ["forecast", "--mesh", o.mesh, "--physics-suite", o.suite, "--init", str(o.init), "--grid", str(o.grid),
            "--hours", repr(o.hours), "--dt", str(o.dt), "--no-products", "--numerics", o.numerics,
            "--physics-mode", RECORD_PHYSICS_MODE]
    if o.oro:
        args += ["--orographic-statistics", str(o.oro)]
    if getattr(o, "lam", None):   # a limited-area mesh: its static, boundaries and start (lam.py)
        args += lam.forecast_words(o.lam)
    return args + list(o.forecast_extra) + rank_words()


# rank recordings (WOOF_HEX_RECORD_RANKS=N, N >= 2; unset or 1: the one-card recording as before): the forecast is recorded as N
# rank threads of one process on the one card (the region cut in N parts, halos as device copies), under the rank
# observer (tools/mdev_observer.py: the rank threads take turns under a baton and meet at one global step boundary;
# each rank thread has its own CuPy pool, so every recorded allocation belongs to one rank). tools/split_plan.py then
# cuts the sealed plan into one plan per rank (PeerSend / PeerRecv for the halo copies). an 8-rank chain (13 Nov) recorded
# this way through an earlier recorder; this brings it into the release recorder. THE BREAKAGE THIS PREVENTS: a mesh
# above 200,000 cells (12 km CONUS, 352,235) has no window on cards, because the only multi-card window needs a plan
# per rank and the release recorder could make none.
def record_ranks() -> int:
    v = os.environ.get("WOOF_HEX_RECORD_RANKS", "").strip()
    if not v:
        return 1
    n = int(v)
    if n < 1:
        raise ValueError(f"WOOF_HEX_RECORD_RANKS={v}: a count of rank threads, 1 or more")
    return n


def rank_words() -> list[str]:
    n = record_ranks()
    return [] if n < 2 else ["--samehost-devices", ",".join(["0"] * n), "--samehost-ranks", "threads"]


def rank_env(work: Path) -> dict:
    n = record_ranks()
    if n < 2:
        return {}
    logs = work / "rank-logs"
    logs.mkdir(parents=True, exist_ok=True)
    return {"MR_RANKS": str(n), "MR_RANK_POOLS": "1", "MR_TOOLS": str(env.TOOLS), "MR_LOGS": str(logs),
            "MR_BATON_TRACE": "1", "WOOF_HEX_MPAS_SAMEHOST_TIMEOUT": "3600", "WOOF_HEX_RECEIPT_EVERY_SECONDS": "0"}


def split_ranks(o, run, pair: dict) -> dict:
    """OUT -> OUT-r0 .. OUT-r<N-1>: the sealed N-rank-thread plan cut per rank (tools/split_plan.py: a command goes
    to the rank of the thread that issued it, read from recording A's driver trace; ownership from A's partial plan,
    which still carries every command's recorded_sequence; a cross-rank device copy becomes PeerSend/PeerRecv).
    Each rank plan is a package the multi-card window (``ad-binder ranks --pkg OUT``) replays on its own card.
    THE BREAKAGE THIS PREVENTS: a mesh above 200,000 cells (12 km CONUS) has no window on cards without rank plans,
    and split_plan.py run by hand after the recorder has removed its work folder has no trace to read."""
    n = record_ranks()
    trace = pair["a"].get("trace")
    partial = o.work / "seal-a" / "a" / "partial-plan.json.gz"
    if not trace or not Path(trace).is_file():
        run.fail(f"--ranks {n}: recording A left no driver trace to split by ({trace})")
    cmd = _py("split_plan.py") + [str(o.out), "--ranks", str(n), "--trace", str(trace), "--out", str(o.out)]
    if partial.is_file():
        cmd += ["--ownership-from", str(partial)]
    t0 = time.monotonic()
    log = o.work / "split.log"
    with open(log, "wb") as stream:
        rc = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, cwd=o.work).returncode
    run.stage("split", t0)
    plans = [Path(f"{o.out}-r{k}") for k in range(n)]
    whole = [p for p in plans if (p / "plan.json").is_file()]
    if rc != 0 or len(whole) != n:
        run.fail(f"--ranks {n}: split_plan.py rc {rc}, {len(whole)} of {n} rank plans written: {_tail(log, 3)}")
    run.log(f"SPLIT {o.out} into {n} rank plans {o.out}-r0..r{n - 1} ({round(time.monotonic() - t0, 1)} s)")
    return {"ranks": n, "prefix": str(o.out), "plans": [str(p) for p in plans], "log": str(log),
            "wall_s": round(time.monotonic() - t0, 1)}


def retain_kind(o) -> str:
    """Which small-upload bound the recorder keeps bytes under. A rank-thread recording uploads the same-host halo
    descriptor tables (9 int64 a row, 648 bytes and more), which the sealer rebuilds from live allocations only when
    their bytes are kept. THE BREAKAGE THIS PREVENTS: at the global bound (512) a 4-rank zoom-globe recording kept
    none of the 2,500 descriptor uploads of 648 to 2,592 bytes and its pair could not seal (18 Nov 2026)."""
    return "limited-area" if record_ranks() > 1 else o.mesh_kind


def _base_env(o, card: str) -> dict:
    e = env.runtime_env(card)
    e.pop("WOOF_HEX_CUDA_GRAPHS", None)
    # The route's kernel forms (route.py): its words win over the caller's, so the receipt's route is what ran.
    e.update(routes.words(getattr(o, "route", None) or routes.default()))
    # The prebuilt kernel packs of a wheel install (woof.hex.cuda_backend.aot) answer CuPy's cached compile with a
    # shipped image and never call the compile underneath, so the observer sees no CUDA source for those modules.
    # THE BREAKAGE THIS PREVENTS: a fresh-install x1.2562 pair observed 8 sources instead of ~100 and stage 2 left
    # 7,602 CuPy kernel arguments unresolved (cupy_where/less/absolute/copy, cub_any in TEMPO, MYNN, GWDO). With the
    # packs off every module comes through CuPy's own cache (filled by the warm forward), same NVRTC bytes.
    e["WOOF_HEX_AOT"] = "off"
    e.update({
        "PYTHONHASHSEED": "0",
        "CUDA_MPS_PIPE_DIRECTORY": str(o.work / "no-mps"),  # never through an MPS daemon: the recorder needs the context
        # a recording keeps every block it allocated: a memory-pressure pool release in one recording and not the
        # other leaves the plan's table pointers in freed blocks (a step-10 hash failure on a shared card).
        "WOOF_HEX_POOL_RELEASE": "0", "GPUWM_HEX_LMEM_RELEASE": "0",
        # hoisted edge/cell kernels pick levels-per-thread by timing at first call: same bits, different launch words,
        # a timing race between A and B (44 'frame_rule_needed' words on x1.40962). A recording pins it.
        "WOOF_HEX_HOIST_LPT": os.environ.get("WOOF_HEX_HOIST_LPT", "15"),
        "WOOF_KERNEL_PLAN": "on", "WOOF_KERNEL_PLAN_CATALOG": "off",
        "WOOF_KERNEL_PLAN_WORKERS": str(o.compile_workers), "WOOF_HEX_HOSTPREP_PROCESSES": "4",
        "WOOF_HEX_ASSEMBLE_THREADS": "4", "CUPY_CACHE_SAVE_CUDA_SOURCE": "1",
    })
    if o.xdg:
        e["XDG_CACHE_HOME"] = str(o.xdg)
    if getattr(o, "lam", None):
        e.update(lam.forecast_env(o.lam))
    # Both recordings of a pair take the same memory-sized decisions (the LW column chunk, the MYNN-EDMF arena and the
    # rest of hexcore's free-memory readers see this many MiB free; record_observer.py). THE BREAKAGE THIS PREVENTS:
    # on a shared card two recordings of one plan read different free memory and split the LW solver and the MYNN-EDMF
    # launch differently, so their launch streams no longer aligned and the seal could not pair them (a 3 km storm
    # recording pair, 2026-11-07). Only batching changes; columns are independent.
    if getattr(o, "fixed_free_mib", None):
        e["P1_FIXED_FREE_MIB"] = str(int(o.fixed_free_mib))
    return e


def warm(o, run: Run) -> None:
    """One untraced forward of the same configuration: resources fetched/built, kernel store and CuPy cache filled,
    so no traced recording compiles or downloads (a compile inside the tracer runs on the forecast thread and its
    module loads land in the trace)."""
    w = o.work / "warm"
    if (w / "done").is_file():
        run.log("warm forward: done earlier, kept")
        return
    shutil.rmtree(w, ignore_errors=True)
    (w / "tmp").mkdir(parents=True)
    e = _base_env(o, o.card)
    e.update({"TMPDIR": str(w / "tmp"), "CUPY_CACHE_DIR": str(w / "cuda"), "CUDA_CACHE_PATH": str(w / "cuda"),
              "WOOF_HEX_KERNEL_CACHE_ROOT": str(w / "kcache")})
    t0 = time.monotonic()
    # the warm forward writes into warm/fc (its history is not used)
    cmd = _cpus(o.cpus_a) + [sys.executable, "-B", "-m", "woof.hex"] + _with_out(forecast_args(o), w / "fc")
    deadline = t0 + 60 * o.wait_for_card
    while True:
        shutil.rmtree(w / "fc", ignore_errors=True)
        rc = _run(cmd, w / "forecast.log", cwd=w, environ=e)
        if rc == 0 or time.monotonic() > deadline or "Forecast has not started" not in _tail(w / "forecast.log", 3, 4000):
            break
        run.log(f"warm forward: card {o.card} admission refused (shared card), retrying in 30 s")
        time.sleep(30)
    run.stage("warm", t0)
    if rc != 0:
        run.fail(f"warm forward failed (rc {rc}): {_tail(w / 'forecast.log', 2)}")
    (w / "done").write_text(json.dumps({"wall_s": run.wall["warm"]}))
    run.log(f"warm forward rc 0 ({run.wall['warm']} s)")


def _with_out(args: list[str], out: Path) -> list[str]:
    return args + ["--out", str(out)]


def _cpus(spec: str | None) -> list[str]:
    return ["taskset", "-c", spec] if spec else []


#: WOOF_HEX_RECORD_STORE: what a recording keeps at each step boundary after step 1 (the recorder's
#: BINDER_BOUNDARY_PAYLOADS). lean: every allocation hashed (the same reference hashes), the TEMPO residue digests
#: written by the recorder, bytes kept only where a held write landed. full: every changed allocation kept whole too.
#: THE BREAKAGE lean PREVENTS: full kept 1.1 GB a step per recording at 2,890 12 km columns and 1.7 GB at 11,688 3 km
#: columns, A and B both, for one reader (p1fix's masked TEMPO hashes) that needs none of it; a 73-step zoom-globe
#: recording wrote 349 GB and no window past minutes fit a box's disk. Both give the same plan. Proof (12 km example
#: start, 16 steps, 2026-11-18): in the full recording p1fix computed all 80 residue hashes from the kept bytes and
#: from the recorder's lines, 80 equal; the lean recording kept 32 bytes a step against 1.19 GB, sealed, replayed
#: strict (7,232 boundary hashes, 80 masked equal, 0 mismatch, as the full one), and the example's adjoint-test on the
#: two plans printed the same objective dot and the same 20 Taylor ratios.
BOUNDARY_STORES = {"lean": "digest", "full": "1"}


def boundary_store() -> str:
    store = os.environ.get("WOOF_HEX_RECORD_STORE", "lean")
    if store not in BOUNDARY_STORES:
        env.defect(f"WOOF_HEX_RECORD_STORE={store}: one of {sorted(BOUNDARY_STORES)}")
    return store


def record_one(o, run: Run, tag: str, card: str, cpus: str | None, libs: dict, retention: Path) -> dict:
    w = o.work
    rec, obs, cuda = w / f"rec-{tag}-recorder", w / f"rec-{tag}-observer", w / f"cuda-{tag}"
    kroot, tmp = w / f"kcache-{tag}", w / f"tmp-{tag}"
    for p in (rec, obs, kroot, tmp, w / f"fc-{tag}"):
        shutil.rmtree(p, ignore_errors=True)
    shutil.rmtree(cuda, ignore_errors=True)
    shutil.copytree(w / "warm" / "cuda", cuda, symlinks=True)
    for p in (rec / "payloads", rec / "images", obs, kroot, tmp):
        p.mkdir(parents=True, exist_ok=True)
    e = _base_env(o, card)
    stores = [str(p) for p in sorted(w.glob("kcache-*/kernel-packs")) if p.parent != kroot]
    stores += [str(p) for p in sorted((w / "warm" / "kcache").glob("kernel-packs"))]
    cub, thrust = env.cupy_host_libs()
    preload = [libs["cublas_observer"], libs["library_observer"], libs["recorder"]]
    if e.get("LD_PRELOAD"):
        preload.append(e["LD_PRELOAD"])
    e.update({
        "TMPDIR": str(tmp), "CUPY_CACHE_DIR": str(cuda), "CUDA_CACHE_PATH": str(cuda),
        # each recording has its own kernel root and reads the warm/earlier stores read-only: a second recording in one
        # root replays the first one's prewarm manifest on the prewarm pool thread, where the observer never sees the
        # module load (48 launches recorded with no image).
        "WOOF_HEX_KERNEL_CACHE_ROOT": str(kroot),
        "BINDER_CAPTURE_DIR": str(rec),
        "BINDER_DIGEST_IMAGE": str(native.digest_image(card)),
        "BINDER_CUB_HOST_LIB": str(cub), "BINDER_THRUST_HOST_LIB": str(thrust),
        "BINDER_LIBRARY_OBSERVER": str(obs / "library-calls.jsonl"),
        "BINDER_CUBLAS_OBSERVER": str(obs / "cublas-geam-calls.jsonl"),
        "LD_PRELOAD": ":".join(preload),
        "BINDER_HOLD_SLOTS": o.hold_slots, "BINDER_D2H_STACKS": "1", "BINDER_D2H_PAYLOADS": "1",
        "BINDER_RETAIN_SMALL_UPLOADS": str(lam.RETAIN_SMALL_UPLOADS[retain_kind(o)]),
        # the store word reaches the recorder (a merge took the rank recording's side of this line and dropped it: a lean
        # 3 h 30 km plan wrote about 3 GB a step per recording while its receipt said lean).
        "BINDER_BOUNDARY_PAYLOADS": BOUNDARY_STORES[boundary_store()],
        "BINDER_BOUNDARY_WORKERS": str(o.boundary_workers),
        "BINDER_IMMUTABLE_UPLOADS_FILE": str(retention),
    })
    if stores:
        e["WOOF_HEX_SYSTEM_KERNEL_CACHE"] = ":".join(stores)
    # retained immutable uploads earlier recordings of this run wrote are linked in, so the recorder does not gzip
    # them again on the forecast thread (content-named; the recorder verifies them).
    subprocess.run(_py("seed_payloads.py") + [str(retention), str(w / "payload-store"), str(w), str(rec / "payloads")],
                   cwd=w, capture_output=True)
    e.update(rank_env(w))
    cmd = _cpus(cpus) + _py("mdev_observer.py" if record_ranks() > 1 else "record_observer.py") + ["--observer-dir", str(env.TOOLS), "--out", str(obs), "--"] \
        + _with_out(forecast_args(o), w / f"fc-{tag}")
    t0 = time.monotonic()
    deadline = t0 + 60 * o.wait_for_card
    while True:
        rc = _run(cmd, w / f"record-{tag}.log", cwd=w, environ=e)
        # a shared card: hex's memory admission refused before the forecast started (nothing ran, nothing was
        # recorded); with --wait-for-card the recording waits for the card instead of failing the pair.
        if rc == 0 or time.monotonic() > deadline or "Forecast has not started" not in _tail(w / f"record-{tag}.log", 3, 4000):
            break
        run.log(f"recording {tag}: card {card} admission refused (shared card), retrying in 30 s")
        time.sleep(30)
        for p in (rec, obs, kroot, w / f"fc-{tag}"):  # the refused attempt's trace and observer files are not this run's
            shutil.rmtree(p, ignore_errors=True)
        for p in (rec / "payloads", rec / "images", obs, kroot):
            p.mkdir(parents=True, exist_ok=True)
        subprocess.run(_py("seed_payloads.py") + [str(retention), str(w / "payload-store"), str(w), str(rec / "payloads")],
                       cwd=w, capture_output=True)
    wall = round(time.monotonic() - t0, 1)
    traces = sorted(rec.glob("driver-*.jsonl.gz"))
    return {"tag": tag, "rc": rc, "wall_s": wall, "card": card, "trace": str(traces[0]) if len(traces) == 1 else None,
            "recorder": str(rec), "observer": str(obs), "cuda": str(cuda)}


def record_pair(o, run: Run, libs: dict, retention: Path) -> dict:
    t0 = time.monotonic()
    if o.serial:
        a = record_one(o, run, "a", o.card, o.cpus_a, libs, retention)
        b = record_one(o, run, "b", o.card, o.cpus_a, libs, retention) if a["rc"] == 0 else {"tag": "b", "rc": -1}
    else:
        with futures.ThreadPoolExecutor(2) as pool:
            fa = pool.submit(record_one, o, run, "a", o.card, o.cpus_a, libs, retention)
            time.sleep(20)  # B starts once A holds its context; the two never share a kernel root
            fb = pool.submit(record_one, o, run, "b", o.card_b, o.cpus_b, libs, retention)
            a, b = fa.result(), fb.result()
    run.stage("record", t0)
    for r in (a, b):
        if r["rc"] != 0 or not r.get("trace"):
            log = o.work / f"record-{r['tag']}.log"   # Python 3.11: no same-quote nesting inside an f-string
            run.fail(f"recording {r['tag']} failed (rc {r['rc']}): {_tail(log, 2)}")
    run.log(f"recorded A ({a['wall_s']} s) and B ({b['wall_s']} s), retention {retention.name}")
    return {"a": a, "b": b}


# ------------------------------------------------------------------------------------------------------------------
# seal
# ------------------------------------------------------------------------------------------------------------------

def analyze(o, run: Run, r: dict, runner: Path) -> Path:
    """catalog, correlation, linearisation and library audits in parallel, then the partial plan and its proposals."""
    tag, trace, O = r["tag"], Path(r["trace"]), Path(r["observer"])
    root = o.work / f"seal-{tag}" / "a"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    e = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(root), OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    jobs = [
        (_py("catalog_v10.py") + ["--driver", str(trace), "--observer", str(O), "--out", str(root / "catalog.json"),
                                  "--config", CATALOG_SUITES[o.suite]], "catalog.log", None, None),
        (_py("correlate_v10.py") + ["--driver", str(trace), "--observer", str(O / "launches.jsonl.gz"),
                                    "--out", str(root / "correlation.json.gz")], "correlate.log", None, None),
        (_py("audit_library_calls.py") + ["--library", str(O / "library-calls.jsonl"), "--driver", str(trace),
                                          "--slots", str(O / "host-slots.jsonl.gz"),
                                          "--out", str(root / "library-calls.json")], "library.log", None, None),
        (_py("audit_cublas_geam.py") + ["--calls", str(O / "cublas-geam-calls.jsonl"), "--driver", str(trace),
                                        "--slots", str(O / "host-slots.jsonl.gz"),
                                        "--out", str(root / "cublas-geam.json")], "cublas.log", None, None),
    ]

    def linearize() -> int:
        with gzip.open(trace, "rb") as src, open(root / "provisional-linearization.json", "wb") as dst, \
                open(root / "linearize.log", "w") as err:
            proc = subprocess.Popen([str(runner), "linearize"], stdin=subprocess.PIPE, stdout=dst, stderr=err)
            shutil.copyfileobj(src, proc.stdin, 1 << 22)
            proc.stdin.close()
            return proc.wait()

    with futures.ThreadPoolExecutor(5) as pool:
        done = [pool.submit(_run, cmd, root / log, cwd=root, environ=e) for cmd, log, _, _ in jobs]
        done.append(pool.submit(linearize))
        rcs = [f.result() for f in done]
    names = ["catalog", "correlate", "library", "cublas", "linearize"]
    for name, rc in zip(names, rcs):
        if rc != 0:
            run.fail(f"analyze {tag}: {name} failed (rc {rc}): {_tail(root / f'{name}.log')}")
    hexsrc = env.hexcore_src()
    cmd = _py("make_partial_plan.py") + [
        "--driver", str(trace), "--observer", str(O / "launches.jsonl.gz"),
        "--correlation", str(root / "correlation.json.gz"),
        "--linearization", str(root / "provisional-linearization.json"), "--census", str(root / "catalog.json"),
        "--source-cache", r["cuda"], "--observed-sources", str(O), "--observed-driver-images",
        str(Path(r["recorder"]) / "images"),
        "--same-run-global-source", str(hexsrc / "mpasphys" / "rrtmg_lw_v842.py"),
        "--verified-source-proof", str(env.DATA / "verified" / "recovered-lw-source-proof.json"),
        "--verified-source-code", str(env.DATA / "verified" / "recovered-lw-source.cu.gz"),
        "--asset-prefix", "recording", "--max-steps", str(o.steps), "--out", str(root / "partial-plan.json.gz")]
    if _run(cmd, root / "partial-plan.log", cwd=root, environ=e) != 0:
        run.fail(f"analyze {tag}: partial plan failed: {_tail(root / 'partial-plan.log', 2)}")
    if _run(_py("audit_library_bindings.py") + ["--catalog", str(root / "catalog.json"), "--library",
                                                str(root / "library-calls.json"), "--driver", str(trace), "--out",
                                                str(root / "library-bindings.json")],
            root / "bindings.log", cwd=root, environ=e) != 0:
        run.fail(f"analyze {tag}: library bindings failed: {_tail(root / 'bindings.log')}")
    cub, thrust = env.cupy_host_libs()
    if _run(_py("propose_library_replacements.py") + [
            "--plan", str(root / "partial-plan.json.gz"), "--audit", str(root / "library-calls.json"),
            "--bindings", str(root / "library-bindings.json"), "--out", str(root / "library-proposals.json"),
            "--cub-path", str(cub), "--cub-sha", env.sha256_file(cub),
            "--thrust-path", str(thrust), "--thrust-sha", env.sha256_file(thrust)],
            root / "proposals.log", cwd=root, environ=e) != 0:
        run.fail(f"analyze {tag}: library proposals failed: {_tail(root / 'proposals.log')}")
    return root


def layout_source(o, run: Run, a: dict) -> Path:
    """CUDA source of the raw-loaded isobaric-diagnostics module (no observed source); stage 2 types its launches."""
    F = o.work / "seal-in"
    F.mkdir(exist_ok=True)
    hits = [p for p in sorted((Path(a["recorder"]) / "images").glob("*.gz"))
            if b".text.slp_finish\x00" in gzip.decompress(p.read_bytes())]
    if not hits:
        # a window with no isobaric diagnostics (a short limited-area window before its first history time) has
        # nothing to type; the rented-card chain's rule. THE BREAKAGE THIS PREVENTS: such a window refused here.
        run.log("isobaric image: no recorded image holds slp_finish (no isobaric diagnostics in this window)")
        return None
    if len(hits) != 1:
        run.fail(f"isobaric image: {len(hits)} recorded images hold slp_finish (exactly one expected)")
    if _run(_py("make_layout_source.py") + [str(hits[0]), str(F)], F / "layout.log", cwd=o.work,
            environ=env.runtime_env(o.card)) != 0:
        run.fail(f"layout source failed: {_tail(F / 'layout.log')}")
    return F / "mpasdiag-isobaric.cu.gz"


def seal_once(o, run: Run, pair: dict, runner: Path, retention: Path, pairs: Path, from_stage2: bool) -> dict:
    a, b = pair["a"], pair["b"]
    S = o.work / "seal-a"
    root = S / "a"
    trace, O, R = a["trace"], Path(a["observer"]), Path(a["recorder"])
    B, driver_b = Path(b["observer"]), b["trace"]
    e = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(root), OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    if not from_stage2:
        t0 = time.monotonic()
        with futures.ThreadPoolExecutor(2) as pool:
            ra, rb = pool.submit(analyze, o, run, a, runner), pool.submit(analyze, o, run, b, runner)
            ra.result(), rb.result()
        run.stage("analyze", t0)
        layout = layout_source(o, run, a)
        if _run(_py("make_frame_states.py") + [str(env.TOOLS), str(B / "frames.jsonl.gz"),
                                              str(o.work / "frame-states-b.jsonl")],
                o.work / "frame-states.log", cwd=o.work) != 0:
            run.fail(f"frame states from recording B: {_tail(o.work / 'frame-states.log')}")
        t0 = time.monotonic()
        if _run(_py("d2h_pystack_sites.py") + [str(root / "partial-plan.json.gz"), trace,
                                               str(O / "host-slots.jsonl.gz"), str(S / "d2h-pystack-sites.json")],
                S / "d2h-pystack-sites.log", cwd=S, environ=e) != 0:
            run.fail(f"d2h sites: {_tail(S / 'd2h-pystack-sites.log')}")
        if _run(_py("classify_d2h_checked.py") + [str(root / "partial-plan.json.gz"), str(S / "d2h-pystack-sites.json"),
                                                  str(R / "payloads"), str(S / "d2h-checked.json"),
                                                  str(env.DATA / "verified" / "image-constants.json"),
                                                  str(env.hexcore_src())],
                S / "d2h-checked.log", cwd=S, environ=e) != 0:
            run.fail(f"d2h classification: {_tail(S / 'd2h-checked.log')}")
        run.stage("d2h", t0)
    layout = o.work / "seal-in" / "mpasdiag-isobaric.cu.gz"
    t0 = time.monotonic()
    lib = env.cublas()
    cmd = _py("seal_phys_stage2.py") + [
        "--plan-a", str(root / "partial-plan.json.gz"), "--driver-a", trace, "--slots-a", str(O / "host-slots.jsonl.gz"),
        "--plan-b", str(o.work / "seal-b" / "a" / "partial-plan.json.gz"), "--driver-b", driver_b,
        "--slots-b", str(B / "host-slots.jsonl.gz"),
        "--catalog-a", str(root / "catalog.json"), "--library-a", str(root / "library-proposals.json"),
        "--cindexer-pairs", str(pairs), "--cublas-a", str(root / "cublas-geam.json"),
        "--cublas-library", str(lib), "--cublas-library-sha256", env.sha256_file(lib),
        "--copy-plans-a", str(O / "copy-plan-inputs.jsonl.gz"), "--copy-plans-b", str(B / "copy-plan-inputs.jsonl.gz"),
        "--ruc-pointers-a", str(O / "ruc-pointers.jsonl.gz"), "--tempo-pointers-a", str(O / "tempo-pointers.jsonl.gz"),
        "--asarray-a", str(O / "host-asarray.jsonl.gz"), "--payloads-a", str(R / "payloads"),
        "--immutable-candidates", str(retention), "--d2h-checked", str(S / "d2h-checked.json"),
        "--images-a", str(R / "images"), "--hexcore-src", str(env.hexcore_src()),
        *(["--layout-source", str(layout)] if layout.is_file() else []),
        "--out", str(S / "stage2.json.gz"), "--residual", str(S / "stage2-residual.json")]
    rc2 = _run(cmd, S / "stage2.log", cwd=S, environ=e)
    try:
        residual = json.loads((S / "stage2-residual.json").read_text())
    except (OSError, ValueError):
        run.fail(f"stage 2 failed (rc {rc2}) without a residual: {_tail(S / 'stage2.log', 2)}")
    counts = residual.get("residual_counts") or {}
    rc3 = None
    if set(counts) <= ALLOWED_RESIDUAL:
        rc3 = _run(_py("seal_phys_stage3.py") + [
            "--plan", str(S / "stage2.json.gz"), "--driver", trace, "--catalog", str(root / "catalog.json"),
            "--frames", str(O / "frames.jsonl.gz"), "--frame-states", str(o.work / "frame-states-b.jsonl"),
            "--frame-states-from-observation", "--out", str(S / "stage3.json.gz"),
            "--reference", str(S / "stage3-reference.jsonl")], S / "stage3.log", cwd=S, environ=e)
    run.stage("seal", t0)
    return {"rc2": rc2, "rc3": rc3, "counts": counts, "residual": str(S / "stage2-residual.json")}


def retain(o, run: Run, pair: dict, base: Path, residual: Path, round_: int) -> Path:
    """This run's retention list: BASE plus the step<=1 uploads the seal named that are equal in both recordings."""
    out = o.work / f"retention-{round_}.txt"
    if _run(_py("gen_immutable.py") + [str(Path(pair["a"]["recorder"])), str(Path(pair["b"]["recorder"])), str(base),
                                       str(out), f"hexad-record round {round_}", str(residual), "--equal-only"],
            o.work / f"retention-{round_}.log", cwd=o.work) != 0:
        run.fail(f"retention round {round_}: {_tail(o.work / f'retention-{round_}.log')}")
    run.log(f"retention round {round_}: {_tail(o.work / f'retention-{round_}.log')}")
    return out


def prove_pairs(o, run: Run, pair: dict, pairs: Path, round_: int) -> None:
    S = o.work / "seal-a"
    new = o.work / f"pairs-new{round_}.json"
    rc = _run(_py("prove_cindexer_pairs.py") + [str(S / "stage2-residual.json"), str(S / "a" / "partial-plan.json.gz"),
                                                pair["a"]["observer"], str(new)],
              o.work / f"pairs-round{round_}.log", cwd=env.TOOLS if os.access(env.TOOLS, os.W_OK) else o.work)
    if rc != 0:
        run.fail(f"CIndexer pairs round {round_}: {_tail(o.work / f'pairs-round{round_}.log')}")
    proved, old = json.loads(new.read_text()), json.loads(pairs.read_text())
    bad = [(p["function"], p["rank"]) for p in proved["pairs"] if not p["set_before_get"]]
    if bad:
        run.fail(f"CIndexer pairs round {round_}: not set-before-get: {bad}")
    have = {(p["function"], p["rank"], p["image_sha256"]) for p in old["pairs"]}
    add = [p for p in proved["pairs"] if (p["function"], p["rank"], p["image_sha256"]) not in have]
    old["pairs"] += add
    old.setdefault("added", []).append({"from": f"hexad-record prove_cindexer_pairs.py round {round_}",
                                        "pairs": len(add)})
    pairs.write_text(json.dumps(old, indent=1))
    run.log(f"CIndexer pairs round {round_}: added {[(p['function'], p['rank']) for p in add]}")


def package(o, run: Run, pair: dict, retention: Path, pairs: Path) -> None:
    a, b = pair["a"], pair["b"]
    S, out = o.work / "seal-a", o.out
    shutil.rmtree(out, ignore_errors=True)
    t0 = time.monotonic()
    log = o.work / "package.log"
    if _run(_py("finalize_package.py") + [str(S / "stage2-residual.json"), str(S / "stage3.json.gz"),
                                          str(S / "stage3-reference.jsonl"), a["recorder"], a["observer"], str(out)],
            log, cwd=o.work) != 0:
        run.fail(f"finalize refused: {_tail(log)}")
    shutil.copy(S / "a" / "catalog.json", out / "catalog-a.json")
    e = dict(os.environ, P1FIX_CATALOG_B=str(o.work / "seal-b" / "a" / "catalog.json"),
             P1FIX_RECORDER_B=b["recorder"], P1FIX_DRIVER_B=b["trace"], PYTHONDONTWRITEBYTECODE="1")
    if _run(_py("p1fix_package.py") + [str(out), str(out / "catalog-a.json"), a["trace"], a["observer"], b["observer"]],
            o.work / "p1fix.log", cwd=o.work, environ=e) != 0:
        run.fail(f"p1fix_package failed: {_tail(o.work / 'p1fix.log')}")
    shutil.copy(pairs, out / "cindexer-pairs.json")
    shutil.copy(retention, out / "retention.txt")
    if _run(_py("make_alloc_map.py") + [str(out), a["observer"]], o.work / "allocmap.log", cwd=o.work) != 0:
        run.fail(f"allocation map: {_tail(o.work / 'allocmap.log')}")
    run.stage("package", t0)


def check_payloads(package_dir: Path) -> dict:
    """Unpack and hash every payload file the plan names; a payload file is named for the SHA-256 of its bytes.
    Returns {"checked": n, "bad": [[file name, what its bytes hash to or why it cannot be read], ...]}.
    THE BREAKAGE THIS PREVENTS (x1.40962 from a start file, 2026-11-11): one of 558 kept files, named for 163,848
    zero bytes, held other bytes (written by the card-to-host capture, fixed in native/d2h_capture.h). Nothing
    looked inside the files before the package was made, so the first sign was the runner's bare "SHA-256
    mismatch" at replay, with no file name. All cores; 558 files (1.9 GB packed) take a few seconds."""
    import hashlib
    import re
    import zlib
    package_dir = Path(package_dir)
    named = set(re.findall(r"[0-9a-f]{64}", (package_dir / "plan.json").read_text()))
    pay = package_dir / "recording" / "payloads"
    files = sorted(f for f in os.listdir(pay) if f.endswith(".gz") and f[:-3] in named)

    def one(name: str):
        digest = hashlib.sha256()
        try:
            with gzip.open(pay / name, "rb") as src:
                while chunk := src.read(1 << 22):
                    digest.update(chunk)
        except (OSError, EOFError, zlib.error) as error:
            return [name, f"unreadable: {error}"]
        return None if digest.hexdigest() == name[:-3] else [name, digest.hexdigest()]

    with futures.ThreadPoolExecutor(max_workers=max(1, min(32, len(os.sched_getaffinity(0))))) as pool:
        bad = [row for row in pool.map(one, files) if row]
    return {"checked": len(files), "bad": bad}


def payload_gate(run: Run, package_dir: Path) -> dict:
    """Refuse, by file name, a package whose plan names a payload file that does not hold the bytes of its name."""
    t0 = time.monotonic()
    result = check_payloads(package_dir)
    run.stage("payload_check", t0)
    if result["bad"]:
        names = "; ".join(f"recording/payloads/{name} holds bytes that hash to {got}" if not got.startswith("unreadable")
                          else f"recording/payloads/{name} is {got}" for name, got in result["bad"][:8])
        run.fail(f"payload check: {len(result['bad'])} of {result['checked']} plan-named payload files do not hold "
                 f"the bytes they are named for: {names}")
    run.log(f"payload check: {result['checked']} plan-named payload files hash to their names")
    return result


def internalize(o, run: Run) -> dict:
    """Make the package self-contained. finalize_package.py links recording/{payloads,images} to recording A, each
    recording/sources file to A's observer and the held store to stage 2. Payloads plan.json does not name are dropped
    (most of a recording is not bound: ~33 GB per x1.2562 boundary), then every link that points into the work
    directory is replaced by its target (moved, or copied with --keep-work). After this the work directory can go.
    THE BREAKAGE THIS PREVENTS: a package whose links point into a deleted work directory binds none of its payloads
    and its replay refuses at load (an earlier pruner deleted recording A itself, 623 payloads lost)."""
    import re
    t0 = time.monotonic()
    out, work = o.out.resolve(), o.work.resolve()
    named = set(re.findall(r"[0-9a-f]{64}", (out / "plan.json").read_text()))
    pay = (out / "recording" / "payloads").resolve()
    files = sorted(os.listdir(pay))
    bound = [f for f in files if f.endswith(".gz") and f[:-3] in named]
    dropped = 0
    if not o.keep_work:
        for f in files:
            if f not in bound:
                (pay / f).unlink()
                dropped += 1
    moved = copied = 0
    placed: dict[str, Path] = {}  # a target already moved into the package: later links to it stay links, relative
    for dirpath, dirnames, filenames in os.walk(out, followlinks=False):
        for name in list(dirnames) + filenames:
            link = Path(dirpath) / name
            if not link.is_symlink():
                continue
            target = Path(os.path.realpath(link))
            inside = next((k for k in placed if str(target) == k or str(target).startswith(k + os.sep)), None)
            if inside is not None:  # the target, or a directory holding it, is already in the package
                new = placed[inside] / os.path.relpath(target, inside) if str(target) != inside else placed[inside]
                link.unlink()
                link.symlink_to(os.path.relpath(new, link.parent))
                continue
            if not str(target).startswith(str(work) + os.sep):
                continue
            link.unlink()
            placed[str(target)] = link
            if o.keep_work:
                (shutil.copytree(target, link, symlinks=False) if target.is_dir() else shutil.copy2(target, link))
                copied += 1
            else:
                shutil.move(str(target), str(link))
                moved += 1
    left = [str(p) for p in out.rglob("*") if p.is_symlink() and str(p.resolve()).startswith(str(work))]
    lost = [f for f in bound if not os.access(out / "recording" / "payloads" / f, os.R_OK)]
    if left or lost:
        run.fail(f"package not self-contained: {len(left)} links into the work directory, {len(lost)} of {len(bound)} "
                 "plan-named payloads unreadable")
    run.stage("internalize", t0)
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    info = {"payloads_bound": len(bound), "payloads_dropped": dropped, "links_moved": moved, "links_copied": copied,
            "package_bytes": size}
    run.log(f"package self-contained: {info}")
    return info


# ------------------------------------------------------------------------------------------------------------------
# replay
# ------------------------------------------------------------------------------------------------------------------

def replay(package_dir: Path, steps: int, runner: Path, card: str, log_dir: Path, name: str = "replay",
           continue_: bool = False) -> dict:
    e = env.runtime_env(card)
    e["BINDER_BOUNDARY_PAYLOADS"] = "1"
    e["CUDA_MPS_PIPE_DIRECTORY"] = str(log_dir / "no-mps")
    if continue_:
        e.update(AD_BINDER_DIAGNOSTIC="1", AD_BINDER_CONTINUE="1")
    out, err = log_dir / f"{name}.stdout", log_dir / f"{name}.stderr"
    t0 = time.monotonic()
    with open(out, "w") as so, open(err, "w") as se:
        rc = subprocess.run([str(runner), "replay", "--config", "plan.json", "--steps", str(steps)],
                            cwd=package_dir, env=e, stdout=so, stderr=se).returncode
    text = out.read_text(errors="replace")
    return {"rc": rc, "wall_s": round(time.monotonic() - t0, 1), "boundary_hashes": text.count('"state_sha256"'),
            "masked_equal": text.count('"masked_equal":true'), "masked_differ": text.count('"masked_equal":false'),
            "mismatch": text.count("state_mismatch"), "stderr_tail": _tail(err, 1, 300)}


# ------------------------------------------------------------------------------------------------------------------
# the whole chain
# ------------------------------------------------------------------------------------------------------------------

def _card_free_mib(card: str) -> int | None:
    """Free MiB of CARD now (nvidia-smi), pinned for both recordings of a limited-area pair; None if unreadable."""
    try:
        out = subprocess.run(["nvidia-smi", "-i", str(card), "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout.split()
        return int(out[0]) if out else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def base_retention(o) -> Path:
    if o.retention:
        return Path(o.retention)
    shipped = env.DATA / "immutable" / f"{o.suite}-{o.mesh}.txt"
    if shipped.is_file():
        return shipped
    empty = o.work / "retention-0.txt"
    empty.write_text("# step<=1 host upload candidates (sha256 bytes); none shipped for this suite and mesh\n")
    return empty


def sampler(o, stop: threading.Event) -> None:
    """Every 10 s: this run's processes at >= 90 % of one core, and their host and device memory (the receipt's
    single-core and memory lines). Reads /proc and nvidia-smi only."""
    me, mark = os.getpid(), str(o.work).encode()
    with open(o.work / "sampler.log", "a") as log:
        while not stop.wait(10):
            pids = []
            for p in Path("/proc").iterdir():
                if p.name.isdigit() and int(p.name) != me:
                    try:
                        if mark in (p / "cmdline").read_bytes():
                            pids.append(p.name)
                    except OSError:
                        pass
            if not pids:
                continue
            top = subprocess.run(["top", "-b", "-H", "-n", "1", "-w", "512", "-p", ",".join(pids[:20])],
                                 capture_output=True, text=True).stdout.splitlines()[7:]
            hot = [l.split() for l in top if len(l.split()) > 11 and float(l.split()[8].replace(",", ".")) >= 90]
            rss = subprocess.run(["ps", "-o", "rss=", "-p", ",".join(pids)], capture_output=True, text=True).stdout
            log.write(f"{time.strftime('%H:%M:%S')} rss_kb {sum(int(x) for x in rss.split())} hot "
                      f"{[(h[0], h[8], h[11]) for h in hot]}\n")
            log.flush()


def record(o) -> dict:
    o.work = Path(o.work or f"{o.out}.work").resolve()
    o.out = Path(o.out).resolve()
    run = Run(o.work)
    # One milestone for every stage: 2 records and seals radiation and land live, 1 holds the named slots. Every tool
    # of the chain (recorder, classify_r2_r3 and the stages that import it) inherits it from this process.
    # THE BREAKAGE THIS PREVENTS: the recorder ran at milestone 2 and the sealer at its default 1, so stage 2 removed
    # 950 radiation launches as held although they were recorded live, the seal passed, and the strict replay left
    # the trajectory in step 1 (deferred-check words and transport species words differed).
    os.environ["BINDER_PHYS_MILESTONE"] = "1" if o.hold_slots else "2"
    (o.work / "no-mps").mkdir(exist_ok=True)
    if o.suite not in CATALOG_SUITES:
        run.fail(f"--suite {o.suite}: the sealer has a launch catalog for {sorted(CATALOG_SUITES)} only")
    steps = o.hours * 3600.0 / o.dt
    if abs(steps - round(steps)) > 1e-9 or round(steps) < 1:
        run.fail(f"--hours {o.hours} is not a whole number of {o.dt} s steps")
    o.steps = int(round(steps))
    for name in ("init", "grid") + (("oro",) if o.oro else ()):
        if not Path(getattr(o, name)).is_file():
            run.fail(f"--{name} {getattr(o, name)} is missing")
    # The mesh kind is data: a limited-area grid records with its case (lam.py), a global one as before. A limited-area
    # grid without a case is refused here, before any card: the forecast would have no boundaries.
    if getattr(o, "lam", None) is None and lam.is_limited_area(o.grid):
        run.fail(f"--grid {o.grid} is a limited-area mesh (bdyMaskCell): pass --lam CASE (the regional door's work "
                 "directory, or a LAM_* case file) so the recording gets its static file, boundaries, start and contract")
    o.mesh_kind = "limited-area" if getattr(o, "lam", None) else "global"
    if getattr(o, "lam", None):
        o.lam = lam.localize_rows(o.lam, o.work / "case")
    else:
        o.lam = None
    # A rank-thread recording runs under the rank observer, whose every free-memory reading is MR_FIXED_FREE_MIB
    # (16,384 MiB unless the recorder names one), so it gets the card's free memory as a limited-area one does.
    # THE BREAKAGE THIS PREVENTS: the 103,950-column zoom globe as 4 rank threads (40,950,330,742 bytes asked) was
    # refused at recording A's own card admission, "but has 17179869184 free", on an empty 96 GB card (18 Nov 2026).
    if (o.mesh_kind == "limited-area" or record_ranks() > 1) and not getattr(o, "fixed_free_mib", None):
        o.fixed_free_mib = _card_free_mib(o.card)
    runner = env.which_runner(o.runner)
    if runner is None:
        run.fail("no replay runner: pass --runner, set HEXAD_RUNNER, or install the hexad runner (ad-binder on PATH)")
    T0 = time.monotonic()
    o.route = getattr(o, "route", None) or routes.default()
    routes.words(o.route)   # an unknown route refuses here, before any card
    run.log(f"{o.suite} on {o.mesh}, dt {o.dt} s, {o.hours} h = {o.steps} steps, numerics {o.numerics}, card "
            f"{o.card}{'' if o.serial else '/' + str(o.card_b)}{' (serial pair)' if o.serial else ''}, out {o.out}, "
            f"route {o.route}")
    t0 = time.monotonic()
    libs = native.build(quiet=True)
    run.stage("build", t0)
    run.log(f"recorder libraries {Path(libs['recorder']).parent.name} ({libs['receipt']['cupti']})")
    stop = threading.Event()
    threading.Thread(target=sampler, args=(o, stop), daemon=True).start()
    try:
        warm(o, run)
        retention = base_retention(o)
        pairs = o.work / "cindexer-pairs.json"
        if not pairs.is_file():
            shutil.copy(o.pairs or env.DATA / "cindexer-pairs-base.json", pairs)
        pair = record_pair(o, run, libs, retention)
        result = seal_once(o, run, pair, runner, retention, pairs, False)
        run.log(f"seal residual {result['counts']} (stage 2 rc {result['rc2']}, stage 3 rc {result['rc3']})")
        if "upload_needs_live_rule" in result["counts"]:
            grown = retain(o, run, pair, retention, Path(result["residual"]), 1)
            added = json.loads(_tail(o.work / "retention-1.log")).get("added", 0)
            if added:  # the recorder must retain the newly listed bytes: the pair is recorded once more with them
                retention = grown
                pair = record_pair(o, run, libs, retention)
                result = seal_once(o, run, pair, runner, retention, pairs, False)
                run.log(f"seal residual {result['counts']} after the retention round ({added} uploads listed)")
        for round_ in (1, 2):
            if "cindexer_scratch_proof" not in result["counts"]:
                break
            prove_pairs(o, run, pair, pairs, round_)
            result = seal_once(o, run, pair, runner, retention, pairs, True)
            run.log(f"seal residual {result['counts']} after CIndexer round {round_}")
        if not set(result["counts"]) <= ALLOWED_RESIDUAL:
            run.fail(f"pair cannot seal: stage-2 residual {result['counts']} ({result['residual']} names the words)")
        if result["rc3"] != 0:
            run.fail(f"stage 3 failed (rc {result['rc3']}): {_tail(o.work / 'seal-a' / 'stage3.log', 2)}")
        package(o, run, pair, retention, pairs)
        # Every kernel kind of the plan has a derivative in this release, or nothing is sealed (coverage.py).
        covered = coverage.check(o.out / "plan.json", runner)
        (o.out / "derivative-coverage.json").write_text(json.dumps(covered, indent=1) + "\n")
        if coverage.refuses(covered):
            (o.out / "plan.json").rename(o.out / "plan.refused.json")
            run.fail(coverage.refusal(covered))
        run.log(f"derivative coverage: {covered['kinds']} kernel kinds, every one covered {covered['by']}"
                + (f"; held by name: {', '.join(covered['held_by_name'])}" if covered["held_by_name"] else ""))
        if covered["runner_lacks"]:
            run.log(f"WARNING: the runner bound here ({runner}) has no window rule for {', '.join(covered['runner_lacks'])}"
                    " (they came in 0.6.2): run this plan's windows with a runner built from 0.6.2 or later "
                    "(--runner or WOOF_HEX_AD_RUNNER), or every window stops at the first such launch")
        deck = deck_of(o.work / "record-a.log")
        run.log("deck of the recorded forecast: " + (json.dumps({k: v for k, v in deck.items() if k not in
                                                                   ("source", "door_line")}) if deck
                                                      else "the forecast printed no end record; the receipt says null"))
        checked = payload_gate(run, o.out)
        packed = internalize(o, run)
        packed["payloads_hashed"] = checked["checked"]
        proof = None
        if not o.no_replay:
            proof = replay(o.out, o.steps, runner, o.card, o.work, "replay-strict")
            run.stage("replay", time.monotonic() - proof["wall_s"])
            run.log(f"replay strict: rc {proof['rc']}, {proof['boundary_hashes']} boundary hashes, "
                    f"{proof['masked_equal']} residue masked-equal, {proof['mismatch']} mismatch, {proof['wall_s']} s")
    finally:
        stop.set()
    receipt = {
        "schema": "hexad.record-receipt/v1", "suite": o.suite, "mesh": o.mesh, "dt_s": o.dt, "hours": o.hours,
        "steps": o.steps, "numerics": o.numerics, "physics_mode": RECORD_PHYSICS_MODE, "hold_slots": o.hold_slots,
        "mesh_kind": o.mesh_kind, "limited_area_case": o.lam, "fixed_free_mib": getattr(o, "fixed_free_mib", None),
        "milestone": os.environ["BINDER_PHYS_MILESTONE"], "route": routes.record(o.route),
        "boundary_store": boundary_store(),
        "derivative_coverage": {k: covered[k] for k in ("kinds", "by", "held_by_name", "table")},
        "init": str(o.init), "init_sha256": env.sha256_file(Path(o.init)), "forecast_args": forecast_args(o),
        "deck": deck,
        "pair": {k: {x: v[x] for x in ("rc", "wall_s", "card")} for k, v in pair.items()},
        "residual": result["counts"], "retention": {"file": "retention.txt", "sha256": env.sha256_file(retention)},
        "libraries": libs["receipt"], "runner": {"path": str(runner), "sha256": env.sha256_file(runner)},
        "hexcore": _hexcore_version(), "wall_s": {**run.wall, "total": round(time.monotonic() - T0, 1)},
        "replay_strict": proof, "package": packed,
    }
    (o.out / "record-receipt.json").write_text(json.dumps(receipt, indent=1))
    if proof is not None and (proof["rc"] != 0 or proof["mismatch"] or proof["boundary_hashes"] == 0):
        run.fail(f"strict replay of the new plan failed: {proof}")
    run.log(f"SEALED {o.out} ({receipt['wall_s']['total']} s)")
    if record_ranks() > 1:   # --ranks N: the sealed plan cut into one plan per rank while the recording is still here
        receipt["rank_split"] = split_ranks(o, run, pair)
        (o.out / "record-receipt.json").write_text(json.dumps(receipt, indent=1))
    if not o.keep_work:  # the recordings are regenerable from the receipt; the package and the logs are kept
        for p in o.work.iterdir():
            if p.is_dir() and p.name != "no-mps":
                shutil.rmtree(p, ignore_errors=True)
    return receipt


#: the deck words a receipt names (the forecast's own resolved values, not the words the recorder passed)
DECK_WORDS = ("config_radtlw_interval", "config_radtsw_interval", "config_convection_scheme", "config_gfl_sub3d",
              "config_number_of_sub_steps", "config_dt")


def deck_of(log: Path) -> dict | None:
    """The deck the recorded forecast ran, in the forecast's own words: the DECK_WORDS of its end record (the
    'mpas-native-forecast-v1' line a forecast prints when it finishes) and the door's line naming NOAA's per-mesh row,
    read from the recording's log. None when the log holds no end record.
    THE BREAKAGE THIS PREVENTS: a package did not say which radiation interval or cumulus setting its forecast ran.
    A 3 km recording made on NOAA's deck through the plain door ran radiation every 15 minutes (the per-mesh row)
    while the deck that matches the reference model bit for bit on that box runs it every 60, and nothing in the
    package told the two apart: a window check or a gradient read on it was a statement about a forecast nobody
    had compared with the reference (2026-11-16)."""
    record = door_line = None
    try:
        lines = log.read_text(errors="replace").splitlines()
    except OSError:
        return None
    for line in lines:
        text = line.strip()
        if text.startswith("woof hex: NOAA hrrrv5 deck for this mesh"):
            door_line = text
        elif text.startswith("{") and '"mpas-native-forecast-v1"' in text:
            try:
                record = json.loads(text)
            except ValueError:
                pass
    if record is None:
        return None
    found: dict[str, list] = {}

    def walk(value) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in DECK_WORDS and not isinstance(item, (dict, list)):
                    if item not in found.setdefault(key, []):
                        found[key].append(item)
                else:
                    walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(record)
    deck = {key: (values[0] if len(values) == 1 else values) for key, values in sorted(found.items())}
    deck["source"] = "the recorded forecast's own end record (recording A)"
    if door_line:
        deck["door_line"] = door_line
    return deck


def _hexcore_version() -> str:
    try:
        from importlib.metadata import version
        return version("recast-woof")
    except Exception:  # noqa: BLE001 - a source checkout has no distribution metadata
        return "source"
