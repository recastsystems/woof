#!/usr/bin/env python3
"""What the radiation chunk width costs at the run's device high-water mark.

WHY THIS EXISTS
---------------
The per-allocation ledger says the run's device high-water mark is reached
inside phase-1 physics, and that most of what is live there is RRTMG chunk
workspace whose size comes from three constants, not from the mesh:

    woof/core/rrtmg_sw.py     SW_BATCH_COLUMN_CHUNK    = 2048
    woof/core/rrtmg_lw.py     LW_BATCH_COLUMN_CHUNK    = 4096
    woof/core/rrtmg_mcica.py  MCICA_DEVICE_COLUMN_CHUNK = 16384

They cost the same on a 40,962-cell mesh as on a 163,842-cell one.  WOOF-ArWen
already treats the width as a budget knob -- ``RRTMGLegacyRadiation`` takes a
``column_chunk``, and ``legacy_radiation_vram_bytes(..., column_chunk=...)``
prices it -- but WOOF-ArWen's own runtime passes it on the RTE+RRTMGP branch and NOT
on the legacy branch the MPAS port runs, so the port gets the class default.

WHAT THIS IS
------------
An INSTRUMENT, not a shipped route.  The WOOF-ArWen checkout is frozen and
hash-verified by the port; this tool sets the three module attributes at
runtime, which the byte-level source guard cannot see.  It exists to put a
measured number and a measured bit-verdict against the upstream change, so the
one-line WOOF-ArWen fix can be argued from evidence.  Nothing here is a front door
and nothing here should become one.

The comparator is the forecast tool's own ``payload_sha256``.  Chunk width
groups work, so a different width is a DIFFERENT ARITHMETIC, not a different
memory layout: expect the digest to move, and report it when it does.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT / "tools"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from measure_physics_tier_park import LiveBytesHook, install_labels  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--sw-chunk", type=int, default=None)
    parser.add_argument("--lw-chunk", type=int, default=None)
    parser.add_argument("--mcica-chunk", type=int, default=None)
    args, rest = parser.parse_known_args(argv)

    import mpas_mesh_binding as binding
    import run_cuda_v841_forecast as forecast

    proof = forecast.proof
    grid = rest[rest.index("--grid") + 1]
    static = rest[rest.index("--static") + 1]
    bind_receipt = binding.bind_mesh(
        proof, args.mesh, grid=Path(grid), static=Path(static), forecast=forecast
    )

    import cupy as cp
    from cupy.cuda import memory_hook as memory_hook_module

    hook = LiveBytesHook()
    install_labels(hook)

    applied: dict[str, Any] = {}
    original_construct = proof._construct_device_stack

    def wrapped_construct(*a: Any, **k: Any) -> Any:
        # woof.core is imported and pinned by now, so the attributes exist and
        # every radiation call reads them fresh.
        import woof.core.rrtmg_lw as rrtmg_lw
        import woof.core.rrtmg_mcica as rrtmg_mcica
        import woof.core.rrtmg_sw as rrtmg_sw

        for module, name, requested in (
            (rrtmg_sw, "SW_BATCH_COLUMN_CHUNK", args.sw_chunk),
            (rrtmg_lw, "LW_BATCH_COLUMN_CHUNK", args.lw_chunk),
            (rrtmg_mcica, "MCICA_DEVICE_COLUMN_CHUNK", args.mcica_chunk),
        ):
            was = int(getattr(module, name))
            applied[name] = {"shipped": was, "used": was}
            if requested is not None:
                if requested < 1:
                    raise ValueError(f"{name} must be >= 1")
                setattr(module, name, int(requested))
                applied[name]["used"] = int(requested)
        return original_construct(*a, **k)

    proof._construct_device_stack = wrapped_construct

    class _Hook(memory_hook_module.MemoryHook):
        name = "LiveBytesHook"

        def malloc_postprocess(self, **kwargs: Any) -> None:
            hook.malloc_postprocess(**kwargs)

        def free_postprocess(self, **kwargs: Any) -> None:
            hook.free_postprocess(**kwargs)

    started = time.perf_counter()
    try:
        with _Hook():
            rc = forecast.main(rest)
    finally:
        proof._construct_device_stack = original_construct
    wall = time.perf_counter() - started

    pool = cp.get_default_memory_pool()
    report = {
        "mesh": args.mesh,
        "n_cells": binding.MESH_BINDINGS[args.mesh].n_cells,
        "bind_rebound": bind_receipt.get("rebound"),
        "chunk_widths": applied,
        "rc": int(rc),
        "wall_seconds": wall,
        "peak_live_bytes": hook.peak,
        "peak_label": hook.peak_label,
        "peak_by_label_bytes": dict(hook.by_label),
        "pool_total_bytes_at_exit": int(pool.total_bytes()),
        "nonclaim": (
            "chunk width groups work; a moved payload digest here is a "
            "different arithmetic, not a different memory layout"
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(
        "[chunk-measure] widths %s; peak %.1f MiB in %s; pool_total %.1f MiB; wall %.1f s"
        % (
            {k: v["used"] for k, v in applied.items()},
            hook.peak / (1024 * 1024),
            hook.peak_label,
            pool.total_bytes() / (1024 * 1024),
            wall,
        )
    )
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
