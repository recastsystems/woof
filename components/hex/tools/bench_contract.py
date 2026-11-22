#!/usr/bin/env python3
"""Time the real driver; add C1 deck intervals to the unmodified baseline."""
import argparse
from datetime import datetime, timezone
import importlib
import json
import os
from woof.hex import card_queue
from pathlib import Path
import socket
import sys
import time


def complete_cache(cache, contract, lane):
    """Only a complete strict contract can authorize a later warm cache."""
    if not cache:
        return
    root = Path(cache).resolve()
    owned = Path(lane).resolve()
    if root == owned or not root.is_relative_to(owned):
        raise ValueError("benchmark cache must be below the current lane")
    result = json.loads(Path(contract).read_text())
    summary = result["summary"]
    if not (summary["decks"] == summary["decks_passed"] == 8
            and summary["kernels_covered"] == summary["kernels_declared"] == 22
            and summary["all_dual_run_identical"] and summary["all_controls_have_teeth"]):
        raise ValueError("an incomplete or failed contract cannot authorize WARM")
    (root / "complete.json").write_text(json.dumps({"contract": str(contract), "summary": summary}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-root",required=True)
    p.add_argument("--label",required=True)
    p.add_argument("--wall-out",required=True)
    args, driver_args = p.parse_known_args()
    sys.path.insert(0,str(Path(args.source_root)/"src"))
    driver = importlib.import_module("woof.hex.drivers.run_cuda_regional_contract")
    original = driver.run_deck
    os.environ["WOOF_RUN_ID"] = args.label
    def emit(stage,phase,attrs):
        visible = (os.environ.get("WOOF_CARD") or card_queue.get(card_queue.CARDS).split(",")[0]
                   or os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0])
        record = {"v":1,"t_utc":datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00","Z"),
            "t_mono":time.monotonic(),"host":socket.gethostname(),"rental_id":os.environ.get("WOOF_RENTAL_ID"),
            "run_id":args.label,"product":"hex","stage":stage,"phase":phase,
            "card":int(visible) if visible.isdecimal() else None,"pid":os.getpid(),
            "attrs":attrs}
        try:
            fd = os.open(os.environ["WOOF_TIME_LEDGER"],os.O_APPEND|os.O_CREAT|os.O_WRONLY,0o600)
            try:
                os.write(fd,(json.dumps(record)+"\n").encode())
            finally:
                os.close(fd)
        except OSError as error:
            print("benchmark timeline append failed:",error,file=sys.stderr)
    # Only the baseline lacks the emitter. The wrapper changes no inputs,
    # references, launches or comparisons.
    if not hasattr(driver,"_emit"):
        def run(deck,bundle,cache):
            attrs = {"deck":deck.name,"kernels":list(deck.kernels)}
            emit("prep.deck","begin",attrs)
            try:
                return original(deck,bundle,cache)
            finally:
                emit("prep.deck","end",attrs)
        driver.run_deck = run
    started = time.perf_counter()
    emit("worker.claim","mark",{"basis":"contract-only proxy"})
    sys.argv = ["run_cuda_regional_contract",*driver_args]
    rc = driver.main()
    emit("finish","mark",{"basis":"contract-only proxy"})
    Path(args.wall_out).write_text(json.dumps({"label":args.label,"wall_seconds":time.perf_counter()-started,
        "exit_code":rc,"source_root":args.source_root,"affinity":sorted(os.sched_getaffinity(0)),
        "ledger":os.environ.get("WOOF_TIME_LEDGER")},indent=2))
    if rc == 0 and os.environ.get("BENCH_CACHE"):
        contract = driver_args[driver_args.index("--out") + 1]
        # The mutex shell may reuse its inherited L variable for a lock path.
        # Derive ownership from the controlled source checkout instead.
        complete_cache(os.environ["BENCH_CACHE"], contract,
                       Path(args.source_root).resolve().parents[1])
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
