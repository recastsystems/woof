"""Phase 0 of the SASE gray-zone campaign: baseline, bands, actuation probe.

Registered by docs/internal/specs/2026-08-02-sase-grayzone-campaign.md
(approved 2026-08-03).  This runner drives the committed
instrument -- ``woof.verify.cases.cbl_dry.partition_run`` -- and appends
one JSON receipt per run to a JSONL ledger; the ``score`` subcommand
reduces that ledger to the per-rung table the campaign freezes its bands
from.  It implements NO candidate: the only thing it can do to the model
is force the existing S3-6f partition cap to a constant (the spec's
pre-registered actuation probe, ``--force-cap 0`` at the probe rungs and
``--force-cap 1`` as the mirror control), by monkeypatching
``woof.core.sase.partition_cap`` in-process -- no model code changes.

Usage:
    python tools/grayzone_phase0.py run --dx 400 --seeds 20260801,20260802
        [--force-cap {0,1}] [--tag NAME] [--ledger PATH] [--npz DIR]
    python tools/grayzone_phase0.py score [--ledger PATH]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# The tree this script sits in is the tree it measures: receipts must
# never be minted from a different checkout than the one committed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_LEDGER = Path("docs/internal/receipts/grayzone/"
                      "phase0_runs.jsonl")


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True,
            text=True, check=True).stdout.strip()
    except Exception:                                # pragma: no cover
        return "unknown"


def cmd_run(args) -> int:
    from woof.config import SASE_PBL_SCHEME
    from woof.verify.cases import cbl_dry

    # --pbl selects the closure under test; the DEFAULT is the SASE
    # baseline, so every receipt minted before the flag existed still
    # means exactly what its command line said.  Scoring is untouched
    # either way -- the firewall: scoring never becomes fitting.
    pbl = SASE_PBL_SCHEME if args.pbl is None else int(args.pbl)
    if args.force_cap is not None and pbl != SASE_PBL_SCHEME:
        # The actuation probe monkeypatches woof.core.sase.partition_cap,
        # which no other closure reads: forcing it under --pbl 11 would
        # mint a receipt claiming a probe that never actuated.
        print(f"--force-cap actuates the SASE partition cap only; "
              f"--pbl {pbl} never reads it", file=sys.stderr)
        return 2
    if args.force_cap is not None:
        # The spec's actuation probe: the composition site is
        # f_used = min(f_solved, f_cap, f_w) with f_cap the host-FP64
        # scalar this one function returns (woof/core/sase.py:1095).
        # Forcing its return forces the bound; 0.0 forces f_used = 0
        # (full RANS), 1.0 removes the cap (the mirror control, which
        # must reproduce HEAD within 2*sigma_seed).
        import woof.core.sase as sase_core
        forced = float(args.force_cap)
        sase_core.partition_cap = lambda delta_h, zi: forced
    ledger_path = Path(args.ledger)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    for repeat in range(args.repeat):
        for seed in seeds:
            t0 = time.time()
            receipt = cbl_dry.partition_run(
                float(args.dx), seed, bl_pbl_physics=pbl,
                outdir=args.npz, tag=args.tag)
            receipt.update({
                "force_cap": (None if args.force_cap is None
                              else float(args.force_cap)),
                "repeat": repeat,
                "wall_s": round(time.time() - t0, 1),
                "git": _git_sha(),
                "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
            with ledger_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(receipt, default=float) + "\n")
            print(f"dx={args.dx} seed={seed} tag={args.tag!r} "
                  f"repeat={repeat} scored="
                  f"{receipt['scored_mixed_layer_fraction']:.6f} "
                  f"x={receipt['x']:.4f} wall={receipt['wall_s']}s",
                  flush=True)
    return 0


def cmd_score(args) -> int:
    import numpy as np

    from woof.verify import gray_zone
    from woof.verify.cases import cbl_dry

    rows = [json.loads(line) for line in
            Path(args.ledger).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r.get("tag", ""), r["dx"]), []).append(r)
    out = {}
    for (tag, dx), members in sorted(groups.items()):
        # determinism repeats score the pair, not the ensemble
        base = [m for m in members if m.get("repeat", 0) == 0]
        scores = np.array([m["scored_mixed_layer_fraction"] for m in base])
        x_mean = float(np.mean([m["x"] for m in base]))
        env = tuple(float(v) for v in
                    gray_zone.subgrid_tke_envelope(x_mean))
        sigma = float(scores.std(ddof=1)) if len(scores) > 1 else 0.0
        entry = {
            "n": len(base),
            "mean": float(scores.mean()),
            "sigma_seed": sigma,
            "x_mean": x_mean,
            "target": float(gray_zone.subgrid_tke_fraction(x_mean)),
            "envelope": env,
            "band": cbl_dry.sweep_band(env, sigma, n=len(base))
            if len(base) > 1 else env,
            "column_fraction_mean": float(np.mean(
                [m["column_fraction"] for m in base])),
            "entrainment_mean": float(np.mean(
                [m["entrainment_fraction"] for m in base])),
            "f_used_mean": float(np.mean(
                [np.mean([l["f"] for l in m["ledger"]])
                 for m in base if m.get("ledger")]))
            if any(m.get("ledger") for m in base) else None,
        }
        pairs = {}
        for m in members:
            pairs.setdefault((m["seed"],), []).append(m["digest"])
        det = {str(k[0]): (len(set(v)) == 1 and len(v) > 1)
               for k, v in pairs.items() if len(v) > 1}
        if det:
            entry["determinism_pairs_bitwise"] = det
        out[f"{tag or 'baseline'}@{int(dx)}m"] = entry
    print(json.dumps(out, indent=2))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("--dx", type=float, required=True)
    p_run.add_argument("--seeds", required=True,
                       help="comma-separated integer seeds")
    p_run.add_argument("--force-cap", type=float, default=None,
                       choices=[0.0, 1.0])
    p_run.add_argument("--pbl", type=int, default=None,
                       choices=[900, 11],
                       help="bl_pbl_physics closure under test (default "
                            "900 = SASE, the Phase-0 baseline; 11 = "
                            "Shin-Hong).  Plumbs to partition_run's "
                            "closure selection only; the scoring path is "
                            "identical for every closure.")
    p_run.add_argument("--tag", default="")
    p_run.add_argument("--repeat", type=int, default=1,
                       help="2 = determinism pair (same seeds twice)")
    p_run.add_argument("--ledger", default=str(DEFAULT_LEDGER))
    p_run.add_argument("--npz", default=None)
    p_run.set_defaults(func=cmd_run)
    p_score = sub.add_parser("score")
    p_score.add_argument("--ledger", default=str(DEFAULT_LEDGER))
    p_score.set_defaults(func=cmd_score)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
