"""Measured parallelism per stage, and the single-core gate.  Standard library only; runs by path.

The rule (2026-10-16): no path uses a single CPU.  A stage of a forecast, a DA
cycle or a box's setup that runs on one CPU thread while cards or cores wait is a defect, not a cost.

Breakage this prevents, measured 2026-10-16: the hex init vertical helper (rw_mpas_geometry --protocol
hex-vertical-v1) held one core for 8+ minutes while an RTX PRO 6000 sat at 0 %.  It had been parallelized
that morning (engine 08be92aabdd, identical words), but hex pinned an older bridge and the box ran the old
binary; no receipt said "1 core", so nobody saw it until a person looked.  Nine stages were single-core that day.  A stage now carries its measured cores in its receipt, and
the gates below fail when one runs at about one core.

What is measured, and how (no profiler, no device syncs, no copies):
  cores = CPU seconds / wall seconds.  CPU seconds are the process tree's: the process, its live
  descendants (from /proc/<pid>/task/*/children) and every descendant already reaped (cutime/cstime), so
  Rust tools, worker pools and threads all count.  Off Linux it is os.times() (self + reaped children).
  Card activity comes from one `nvidia-smi -lms` reader process, never from the CUDA context.

The rule (`judge`): a stage or window FAILS when it lasts more than MIN_WALL_S (5 s), averages between
MIN_CORES (0.6) and MAX_CORES (1.3) cores, and no card was busy for at least half of it.  Below 0.6 cores
the tree is waiting (network, disk, a queue, a card in another process); that is listed as WAITING in every
receipt, never hidden, but adding threads does not fix a wait, so it does not fail this gate.  A stage the
samples cover completely is judged by its contiguous windows: a parallel burst followed by an I/O wait
can average one core without ever running on one core. Without full samples the stage average still gates.
A stage the
law allows (a wait by design, a host thread feeding a busy card) is exempted only by an allowlist entry
that names its reason, its owner and when it retires; an entry without a reason is refused.

    python parallelism.py watch --out P.json [--ledger timeline.jsonl] [--allow A.json] -- CMD ...
    python parallelism.py check --ledger timeline.jsonl [--samples P.json] [--allow A.json]
    python parallelism.py selftest      # the gate must fail a deliberately serial stage, and pass a parallel one
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

MIN_WALL_S = 5.0
MAX_CORES = 1.3
MIN_CORES = 0.6
CARD_BUSY_PERCENT = 5
CARD_BUSY_FRACTION = 0.5
ALLOWLIST = Path(__file__).with_name("single_core_allowlist.json")
BASIS = ("process tree CPU (self + live descendants + reaped children) / wall; "
         "cards from one nvidia-smi -lms reader; no device syncs")

try:
    _TICK = float(os.sysconf("SC_CLK_TCK"))
except (AttributeError, ValueError, OSError):
    _TICK = 100.0
_LINUX = sys.platform.startswith("linux") and os.path.isdir("/proc/self/task")


# ---------------------------------------------------------------------------------------------- CPU clock
def _stat(pid):
    """(ppid, own+reaped-children ticks, comm, own ticks) of one pid, or None when it is gone."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as handle:
            raw = handle.read().decode("utf-8", "replace")
    except OSError:
        return None
    close = raw.rfind(")")
    fields = raw[close + 2:].split()
    # fields[0] is state (stat field 3); utime..cstime are stat fields 14..17
    return (int(fields[1]), sum(int(x) for x in fields[11:15]), raw[raw.find("(") + 1:close],
            int(fields[11]) + int(fields[12]))


def _children(pid):
    out = []
    try:
        tasks = os.listdir(f"/proc/{pid}/task")
    except OSError:
        return out
    for tid in tasks:
        try:
            with open(f"/proc/{pid}/task/{tid}/children") as handle:
                out.extend(int(x) for x in handle.read().split())
        except OSError:
            continue
    return out


def tree_pids(root=None):
    root = os.getpid() if root is None else int(root)
    seen, todo = [], [root]
    while todo:
        pid = todo.pop()
        if pid in seen:
            continue
        seen.append(pid)
        todo.extend(_children(pid))
    return seen


def tree_ticks(root=None, own=None):
    """{pid: (ticks, comm)} for the tree under root (Linux); ``own`` (a dict) also receives each pid's own
    utime+stime, without its reaped children."""
    out = {}
    for pid in tree_pids(root):
        row = _stat(pid)
        if row is not None:
            out[pid] = (row[1], row[2])
            if own is not None:
                own[str(pid)] = row[3]
    return out


_NVIDIA = re.compile(r"^/dev/nvidia\d+$")


def holds_card(pids, known=None):
    """True when a pid in pids has a /dev/nvidiaN open: the card's activity can be this tree's.

    A busy card excuses a one-core host thread only when that thread's tree is on the card: on a shared host
    another job's kernels must not hide a CPU-only stage (2026-10-16, a shared host: a hex init on one core read as
    card-bound because another lane's forecast held the 5090). Positive pids are remembered in `known`."""
    known = set() if known is None else known
    for pid in pids:
        if pid in known:
            return True
        try:
            fds = os.listdir(f"/proc/{pid}/fd")
        except OSError:
            continue
        for fd in fds:
            try:
                if _NVIDIA.match(os.readlink(f"/proc/{pid}/fd/{fd}")):
                    known.add(pid)
                    return True
            except OSError:
                continue
    return False


def tree_cpu_seconds(root=None):
    """CPU seconds used so far by root's process tree (default: this process)."""
    if _LINUX:
        return round(sum(t for t, _ in tree_ticks(root).values()) / _TICK, 3)
    if root not in (None, os.getpid()):
        return None
    t = os.times()
    return round(t.user + t.system + t.children_user + t.children_system, 3)


# ---------------------------------------------------------------------------------------------- cards
class CardSampler:
    """Card utilisation from one long-lived `nvidia-smi -lms` process (NVML, outside any CUDA context)."""

    def __init__(self, interval_ms=1000):
        self.samples = []          # (t_mono, max utilisation over the visible cards)
        self.proc = None
        self.interval = interval_ms / 1000.0
        exe = __import__('woof.gpu_smi', fromlist=['executable']).executable()
        from woof.local_gpu import no_local_gpu
        if exe is None or no_local_gpu():
            return
        command = [*__import__('woof.gpu_smi', fromlist=['command']).command(), "--query-gpu=index,utilization.gpu", "--format=csv,noheader,nounits",
                   f"-lms={int(interval_ms)}"]
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if visible and all(x.strip().isdigit() for x in visible.split(",")):
            command.insert(1, "--id=" + visible)
        try:
            self.proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except OSError:
            self.proc = None
            return
        threading.Thread(target=self._read, name="card-sampler", daemon=True).start()

    def _read(self):
        tick, best = None, 0
        for line in self.proc.stdout:
            parts = [x.strip() for x in line.split(",")]
            if len(parts) != 2 or not parts[1].isdigit():
                continue
            now = time.monotonic()
            if tick is not None and now - tick > self.interval / 2:
                self.samples.append((tick, best))
                tick, best = None, 0
            if tick is None:
                tick = now
            best = max(best, int(parts[1]))
        if tick is not None:
            self.samples.append((tick, best))

    @property
    def active(self):
        return self.proc is not None

    def busy_fraction(self, t0, t1):
        """Fraction of samples in [t0, t1] with a visible card at >= CARD_BUSY_PERCENT; None if unknown."""
        inside = [u for t, u in list(self.samples) if t0 <= t <= t1]
        if not inside:
            return None
        return round(sum(u >= CARD_BUSY_PERCENT for u in inside) / len(inside), 3)

    def close(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            with contextlib.suppress(Exception):
                self.proc.wait(timeout=5)


# ---------------------------------------------------------------------------------------------- the rule
def load_allowlist(path=None):
    """Entries {match, reason, owner, retire_when}; `match` is a regex on a stage name or process signature."""
    path = ALLOWLIST if path is None else Path(path)
    if not Path(path).is_file():
        return []
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = rows.get("allow", []) if isinstance(rows, dict) else rows
    for row in rows:
        missing = [k for k in ("match", "reason", "owner", "retire_when") if not str(row.get(k, "")).strip()]
        if missing:
            raise ValueError(f"{path}: allowlist entry {row.get('match')!r} has no {', '.join(missing)}: "
                             "a single-core exemption must name why, who owns it and when it retires")
        re.compile(row["match"])
    return rows


def allowed(name, allow):
    for row in allow:
        if re.search(row["match"], name or ""):
            return row
    return None


def judge(wall_s, cores, card_busy, *, min_wall=MIN_WALL_S, max_cores=MAX_CORES, min_cores=MIN_CORES):
    """'single-core' | 'waiting' | 'card-bound' | 'parallel' | 'short' | 'unmeasured'."""
    if cores is None or wall_s is None:
        return "unmeasured"
    if wall_s <= min_wall:
        return "short"
    if cores > max_cores:
        return "parallel"
    if card_busy is not None and card_busy >= CARD_BUSY_FRACTION:
        return "card-bound"
    if cores < min_cores:
        return "waiting"
    return "single-core"


def stage_record(wall_s, cpu_s, card_busy=None, **limits):
    cores = round(cpu_s / wall_s, 2) if wall_s and cpu_s is not None and wall_s > 0 else None
    return {"wall_s": round(wall_s, 3), "cpu_s": cpu_s, "cores": cores,
            "card_busy": card_busy, "verdict": judge(wall_s, cores, card_busy, **limits)}


class StageMeter:
    """Per-stage parallelism for an in-process driver: mark() at a stage start, since(mark) at its end."""

    def __init__(self, cards=True):
        self.cards = CardSampler() if cards else None
        self.ncpu = os.cpu_count() or 1
        self.own, self._known, self._stop = [], set(), threading.Event()
        if self.cards is not None and self.cards.active and _LINUX:
            threading.Thread(target=self._watch_own, name="card-owner", daemon=True).start()

    def _watch_own(self):
        while not self._stop.wait(1.0):
            self.own.append((time.monotonic(), holds_card(tree_pids(), self._known)))

    def mark(self):
        return time.monotonic(), tree_cpu_seconds()

    def since(self, mark, **limits):
        t0, c0 = mark
        t1, c1 = time.monotonic(), tree_cpu_seconds()
        busy = self.cards.busy_fraction(t0, t1) if self.cards is not None else None
        if busy is not None and not any(on for t, on in list(self.own) if t0 <= t <= t1 + 1.0):
            busy = 0.0          # a busy card that this process tree never held excuses nothing
        cpu = None if c0 is None or c1 is None else round(c1 - c0, 3)
        return stage_record(t1 - t0, cpu, busy, **limits)

    def close(self):
        self._stop.set()
        if self.cards is not None:
            self.cards.close()


def failures(records, allow=()):
    """[(name, record, allow-entry-or-None)] for every single-core record; allowed ones carry their entry."""
    out = []
    for name, rec in records:
        if rec.get("verdict") == "single-core":
            out.append((name, rec, allowed(name, allow)))
    return out


# ---------------------------------------------------------------------------------------------- C1 ledger stages
def ledger_stages(path, cards=None, **limits):
    """Per-stage parallelism from a C1 timeline: begin/end pairs (same pid, stage) that carry attrs.cpu_s."""
    open_, rows = {}, []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        key = (e.get("pid"), e.get("stage"))
        cpu = (e.get("attrs") or {}).get("cpu_s")
        if e.get("phase") == "begin":
            open_.setdefault(key, []).append((e.get("t_mono"), cpu))
        elif e.get("phase") == "end" and open_.get(key):
            t0, c0 = open_[key].pop()
            t1 = e.get("t_mono")
            if not isinstance(t0, (int, float)) or not isinstance(t1, (int, float)):
                continue
            busy = cards(t0, t1) if cards else None
            cpu_s = round(cpu - c0, 3) if isinstance(cpu, (int, float)) and isinstance(c0, (int, float)) else None
            rows.append((e["stage"], {**stage_record(t1 - t0, cpu_s, busy, **limits), "t0": t0, "t1": t1,
                                      "pid": e.get("pid")}))
    return rows


# ---------------------------------------------------------------------------------------------- watch a command
def _signature(pid, comm):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            argv = [a.decode("utf-8", "replace") for a in handle.read().split(b"\0") if a]
    except OSError:
        argv = []
    if not argv:
        return comm
    exe = os.path.basename(argv[0])
    if exe.startswith("python") and len(argv) > 1:
        if "-m" in argv[1:3]:
            i = argv.index("-m")
            return "python -m " + (argv[i + 1] if i + 1 < len(argv) else "")
        return "python " + os.path.basename(next((a for a in argv[1:] if not a.startswith("-")), ""))
    tail = next((a for a in argv[1:4] if a.startswith("--protocol") or not a.startswith(("/", "-"))), "")
    if tail == "--protocol" and "--protocol" in argv:
        tail = "--protocol " + argv[argv.index("--protocol") + 1]
    return (exe + " " + tail).strip()


def watch(command, *, interval=1.0, **limits):
    """Run command; sample its tree each interval. Returns (rc, samples dict)."""
    if not _LINUX:
        raise SystemExit("parallelism watch needs Linux /proc")
    cards = CardSampler(int(interval * 1000))
    started = time.monotonic()
    proc = subprocess.Popen(command)
    ticks, names, on_card = [], {}, set()
    try:
        while True:
            now = time.monotonic()
            own = {}
            table = tree_ticks(proc.pid, own)
            for pid, (t, comm) in table.items():
                if pid not in names:
                    names[pid] = _signature(pid, comm)
            total = sum(t for t, _ in table.values())
            # a pid that left between ticks was reaped by its parent, whose cutime now holds it
            ticks.append({"t": round(now - started, 3), "mono": now, "total": total,
                          "per": {str(p): t for p, (t, _) in table.items()}, "own": own,
                          "on_card": holds_card(table, on_card) if cards.active else False})
            if proc.poll() is not None:
                break
            time.sleep(max(0.0, interval - (time.monotonic() - now)))
    finally:
        rc = proc.wait()
        time.sleep(interval * 1.2)
        cards.close()
    return rc, analyse(ticks, names, cards, interval=interval, ncpu=os.cpu_count() or 1, **limits)


def analyse(ticks, names, cards, *, interval, ncpu, **limits):
    """Per-tick tree cores, contiguous single-core windows, per-process totals."""
    min_wall = limits.get("min_wall", MIN_WALL_S)
    series = []
    for a, b in zip(ticks, ticks[1:]):
        dt = b["mono"] - a["mono"]
        if dt <= 0:
            continue
        cores = max(0.0, (b["total"] - a["total"]) / _TICK / dt)
        busy = cards.busy_fraction(a["mono"], b["mono"]) if cards is not None and cards.active else None
        if busy is not None and not (a.get("on_card") or b.get("on_card")):
            busy = 0.0          # the card is busy, but not with this tree: it excuses nothing
        top = sorted(((b["per"].get(p, 0) - a["per"].get(p, b["per"].get(p, 0)), p) for p in b["per"]),
                     reverse=True)[:1]
        # The busiest single process's own CPU (no reaped children): one thread pinned near a core, or the
        # tree's CPU spread over many processes that each mostly wait. Ticks without "own" (older samples)
        # carry None and are judged on the tree's cores alone.
        own_a, own_b = a.get("own"), b.get("own")
        busiest = (max((own_b[p] - own_a.get(p, own_b[p]) for p in own_b), default=0) / _TICK / dt
                   if own_a is not None and own_b is not None else None)
        series.append({"t0": a["t"], "t1": b["t"], "mono0": a["mono"], "mono1": b["mono"],
                       "cores": round(cores, 2), "card_busy": busy,
                       "busiest": round(busiest, 2) if busiest is not None else None,
                       "top": names.get(int(top[0][1])) if top and top[0][0] > 0 else None})
    windows, run = [], []

    def close_run():
        if not run:
            return
        wall = run[-1]["t1"] - run[0]["t0"]
        cpu = sum(s["cores"] * (s["t1"] - s["t0"]) for s in run)
        busy = [s["card_busy"] for s in run if s["card_busy"] is not None]
        tops = {}
        for s in run:
            if s["top"]:
                tops[s["top"]] = tops.get(s["top"], 0) + 1
        rec = stage_record(wall, round(cpu, 3), round(sum(busy) / len(busy), 3) if busy else None, **limits)
        peaks = [s["busiest"] for s in run if s.get("busiest") is not None]
        if rec["verdict"] == "single-core" and len(peaks) == len(run) and \
                sum(peaks) / len(peaks) < limits.get("min_cores", MIN_CORES):
            # About one core in total, but no process ran even MIN_CORES of one: many processes each waiting
            # (24 rw_zarr day extracts at ~0.05 core each on network reads, the hydro spin-up forcing fetch,
            # 2.9.4 R11 2026-10-22). No thread is the bottleneck, so more threads would not help: a wait.
            rec = {**rec, "verdict": "waiting", "busiest_process_cores": round(sum(peaks) / len(peaks), 2)}
        if rec["verdict"] in ("single-core", "waiting"):
            busiest = sorted(tops, key=lambda k: -tops[k])[:3]
            windows.append({**rec, "t0": run[0]["t0"], "t1": run[-1]["t1"], "mono0": run[0]["mono0"],
                            "mono1": run[-1]["mono1"], "processes": busiest,
                            "process": ", ".join(f"{k} ({tops[k] * interval:.0f} s)" for k in busiest) or None})
        run.clear()

    for s in series:
        per = judge(s["t1"] - s["t0"] + min_wall + 1, s["cores"], s["card_busy"], **limits)
        if per in ("single-core", "waiting") and (not run or _same_kind(run[0], s, limits)):
            run.append(s)
        else:
            close_run()
            if per in ("single-core", "waiting"):
                run.append(s)
    close_run()
    per_proc = {}
    first, last = (ticks[0], ticks[-1]) if ticks else ({}, {})
    for pid, name in names.items():
        seen = [k for k in ticks if str(pid) in k["per"]]
        if len(seen) < 2:
            continue
        wall = seen[-1]["mono"] - seen[0]["mono"]
        cpu = (seen[-1]["per"][str(pid)] - seen[0]["per"][str(pid)]) / _TICK
        if wall > 0 and cpu > 0:
            per_proc[f"{pid} {name}"] = {"wall_s": round(wall, 1), "cpu_s": round(cpu, 1),
                                         "cores": round(cpu / wall, 2)}
    wall = (last.get("mono", 0) - first.get("mono", 0)) if ticks else 0
    cpu = (last.get("total", 0) - first.get("total", 0)) / _TICK if ticks else 0
    return {"basis": BASIS, "interval_s": interval, "ncpu": ncpu, "limits": _limits(limits),
            "wall_s": round(wall, 1), "cpu_s": round(cpu, 1), "cores": round(cpu / wall, 2) if wall else None,
            "share_of_box": round(cpu / wall / ncpu, 3) if wall else None,
            "windows": windows, "processes": per_proc, "series": series}


def _same_kind(a, b, limits):
    lo = limits.get("min_cores", MIN_CORES)
    return (a["cores"] < lo) == (b["cores"] < lo)


def _limits(limits):
    return {"min_wall_s": limits.get("min_wall", MIN_WALL_S), "max_cores": limits.get("max_cores", MAX_CORES),
            "min_cores": limits.get("min_cores", MIN_CORES), "card_busy_percent": CARD_BUSY_PERCENT,
            "card_busy_fraction": CARD_BUSY_FRACTION}


def verdict(result, allow, ledger_rows=()):
    """Name every single-core window by its C1 stage (if a ledger was given) or its busiest process."""
    bad, exempt = [], []
    for w in result.get("windows", []):
        if w["verdict"] != "single-core":
            continue
        stages = sorted({name for name, r in ledger_rows
                         if r.get("t0", 1e18) <= w["mono1"] and r.get("t1", -1) >= w["mono0"]})
        names = stages + (w.get("processes") or [w.get("process") or "?"])
        entry = next((allowed(n, allow) for n in names if allowed(n, allow)), None)
        what = stages + [w.get("process") or "?"]
        row = {"what": " / ".join(what), "t0": w["t0"], "wall_s": w["wall_s"], "cores": w["cores"],
               "card_busy": w["card_busy"]}
        (exempt if entry else bad).append({**row, **({"allowed": entry["reason"]} if entry else {})})
    for name, rec, entry in failures(ledger_rows, allow):
        # A fetch can include a parallel pre-warm followed by a long network
        # wait. Its whole-stage average can be one core without any serial
        # window. Complete samples already judge the contiguous windows above;
        # retain the aggregate as evidence, but do not invent another window.
        if _stage_sampled(result.get("series", ()), rec):
            continue
        row = {"what": name, "wall_s": rec["wall_s"], "cores": rec["cores"], "card_busy": rec["card_busy"],
               "basis": "C1 stage"}
        (exempt if entry else bad).append({**row, **({"allowed": entry["reason"]} if entry else {})})
    return bad, exempt


def _stage_sampled(series, record):
    """Full coverage of this stage's CPU, rather than an unrelated subtree."""
    cursor, end = record.get("t0"), record.get("t1")
    cpu = record.get("cpu_s")
    if cursor is None or end is None or cpu is None:
        return False
    observed, edges = 0.0, 0.0
    for sample in series:
        if sample["mono1"] <= cursor:
            continue
        if sample["mono0"] > cursor:
            return False
        overlap = min(end, sample["mono1"]) - cursor
        observed += sample["cores"] * overlap
        # A boundary sample cannot locate its CPU within that interval.
        if sample["mono0"] < cursor or sample["mono1"] > end:
            edges += sample["cores"] * (sample["mono1"] - sample["mono0"])
        cursor = sample["mono1"]
        if cursor >= end:
            # /proc ticks and rounded sample cores also incur small clock
            # differences. A serial child overlapped by another parallel
            # subtree has different CPU evidence and must still fail.
            return abs(cpu - observed) <= edges + 0.1 + observed * 0.01
    return False


def _print(result, bad, exempt, out=sys.stdout):
    print(f"single-core gate: {'RED' if bad else 'GREEN'}  wall {result.get('wall_s')} s, "
          f"{result.get('cores')} cores of {result.get('ncpu')} on average", file=out)
    for row in bad:
        print(f"  SINGLE-CORE {row['what']}: {row['wall_s']:.1f} s at {row['cores']} cores, card busy "
              f"{row['card_busy']}", file=out)
    for row in exempt:
        print(f"  allowed     {row['what']}: {row['wall_s']:.1f} s at {row['cores']} cores ({row['allowed']})",
              file=out)
    for w in result.get("windows", []):
        if w["verdict"] == "waiting":
            print(f"  waiting     {w.get('process')}: {w['wall_s']:.1f} s at {w['cores']} cores (I/O, network "
                  "or a queue; listed, not failed)", file=out)


def _limits_args(args):
    return {"min_wall": args.min_wall, "max_cores": args.max_cores, "min_cores": args.min_cores}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("watch", "check", "selftest"):
        s = sub.add_parser(name)
        s.add_argument("--allow", type=Path, default=None)
        s.add_argument("--min-wall", type=float, default=MIN_WALL_S)
        s.add_argument("--max-cores", type=float, default=MAX_CORES)
        s.add_argument("--min-cores", type=float, default=MIN_CORES)
        s.add_argument("--interval", type=float, default=1.0)
        if name != "selftest":
            s.add_argument("--ledger", type=Path)
            s.add_argument("--out", type=Path)
        if name == "watch":
            s.add_argument("--record-only", action="store_true", help="write the receipt, never fail")
            s.add_argument("command", nargs=argparse.REMAINDER)
        if name == "check":
            s.add_argument("--samples", type=Path)
    args = p.parse_args(argv)
    allow = load_allowlist(args.allow)
    limits = _limits_args(args)
    if args.cmd == "selftest":
        return selftest(interval=args.interval, **limits)
    if args.cmd == "watch":
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        if not command:
            p.error("watch needs a command after --")
        rc, result = watch(command, interval=args.interval, **limits)
        cards = _cards_from(result)
        rows = ledger_stages(args.ledger, cards, **limits) if args.ledger and args.ledger.is_file() else []
    else:
        result = json.loads(args.samples.read_text()) if args.samples else {"windows": []}
        rc, cards = 0, _cards_from(result)
        rows = ledger_stages(args.ledger, cards, **limits) if args.ledger else []
    bad, exempt = verdict(result, allow, rows)
    result.update(command=locals().get("command"), stages=[{"stage": n, **r} for n, r in rows],
                  single_core=bad, allowed=exempt, gate="RED" if bad else "GREEN")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    _print(result, bad, exempt)
    if rc:
        return rc
    return 3 if bad and not getattr(args, "record_only", False) else 0


def _cards_from(result):
    series = [s for s in result.get("series", []) if s.get("card_busy") is not None]
    if not series:
        return None

    def busy(t0, t1):
        inside = [s["card_busy"] for s in series if s["mono0"] <= t1 and s["mono1"] >= t0]
        return round(sum(inside) / len(inside), 3) if inside else None
    return busy


SERIAL = "import time\nend = time.monotonic() + {s}\nx = 0\nwhile time.monotonic() < end:\n    x += 1\n"


def selftest(*, interval=0.5, seconds=None, **limits):
    """The gate must FAIL one thread spinning past min_wall and PASS the same work on several processes."""
    seconds = seconds or limits.get("min_wall", MIN_WALL_S) + 2.0
    serial = [sys.executable, "-c", SERIAL.format(s=seconds)]
    n = max(2, min(4, (os.cpu_count() or 2)))
    parallel = [sys.executable, "-c",
                "import subprocess, sys\n"
                f"ps = [subprocess.Popen([sys.executable, '-c', {SERIAL.format(s=seconds)!r}]) for _ in range({n})]\n"
                "sys.exit(max(p.wait() for p in ps))\n"]
    from woof.local_gpu import NO_LOCAL_GPU_ENV
    old = os.environ.get(NO_LOCAL_GPU_ENV)
    os.environ[NO_LOCAL_GPU_ENV] = "1"     # the self-test judges CPU only; a busy shared card must not excuse it
    try:
        _, one = watch(serial, interval=interval, **limits)
        _, many = watch(parallel, interval=interval, **limits)
    finally:
        if old is None:
            os.environ.pop(NO_LOCAL_GPU_ENV, None)
        else:
            os.environ[NO_LOCAL_GPU_ENV] = old
    bad_one, _ = verdict(one, [])
    bad_many, _ = verdict(many, [])
    ok = bool(bad_one) and not bad_many
    print(f"single-core gate self-test: {'PASS' if ok else 'FAIL'}: serial stage {one['cores']} cores -> "
          f"{'caught' if bad_one else 'MISSED'}; {n}-process stage {many['cores']} cores -> "
          f"{'wrongly failed' if bad_many else 'passed'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
