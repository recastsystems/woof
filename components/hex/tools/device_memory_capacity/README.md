# Device-memory capacity protocol

These tools freeze the measurement protocol for the 12 GiB capacity
work. They do not change model arithmetic and they do not turn an extrapolation
into a hardware result.

## Receipts

Use one directory per isolated run. Keep the command, source commit, device
identity, process-memory trace, in-process phase snapshots, model receipt,
state hash, and timing output together. A run is not comparable when its source
commit, CuPy/CUDA build, device, mesh, physics configuration, allocator policy,
or launch order differs without being recorded.

The primary whole-process number is the command process tree's own
`nvidia-smi` rows. `cudaMemGetInfo` appears only as a labelled device-wide
cross-check because another process can move it.

## Instrument controls

Run these before believing a before/after ledger:

```bash
python tools/device_memory_capacity/runtime_ledger.py identity \
  --output evidence/capacity/device.json

python tools/device_memory_capacity/runtime_ledger.py selftest-allocation \
  --allocation-mib 512 \
  --output evidence/capacity/allocation-view-control.json

python tools/device_memory_capacity/kernel_reservation_probe.py matrix \
  --local-floats 2048 \
  --output-dir evidence/capacity/kernel-controls

python tools/device_memory_capacity/sibling_contamination_probe.py control \
  --allocation-mib 512 \
  --output evidence/capacity/sibling-control.json
```

The allocation/view control must show one allocation and no allocation for its
view. The kernel matrix must show zero local bytes for the negative control, a
non-zero `local_size_bytes` and process reservation for the positive control,
and no material second reservation after the context is already sized. The
sibling control must show device-wide free memory moving while the parent's
per-process row stays stable.

## Whole-process wrapper

Wrap each baseline and candidate command:

```bash
python tools/device_memory_capacity/process_memory_probe.py \
  --output evidence/capacity/candidate-x1-run1/process.json \
  --stdout evidence/capacity/candidate-x1-run1/stdout.log \
  --stderr evidence/capacity/candidate-x1-run1/stderr.log \
  --interval-ms 50 \
  -- \
  python src/hexcore/drivers/run_cuda_v841_full_physics_x4.py <runner arguments>
```

The sampled process peak is an anchor. Add calls to
`runtime_ledger.cuda_snapshot()` at admitted phase boundaries to capture CuPy
pool live/total/free blocks and an explicit array inventory. Synchronize the
intended stream immediately before each snapshot; do not insert synchronizes
inside kernels or change launch order merely for instrumentation.

For every resident or transient array, record:

- canonical owner and any aliases;
- dtype, shape, strides, bytes, allocation pointer, and data pointer;
- lifetime/phase and read/write access;
- allocation and release call site.

Use `ArrayContract` at every borrow boundary, `LeaseToken` to invalidate stale
workspace views after rebinding, `assert_lifetime_contracts()` for every arena
or alias, and `assert_parking_safe()` before moving a canonical owner to host.
Concurrent read-only aliases are accepted. A concurrent writable overlap, wrong
shape/dtype/layout, stale generation, or live shared allocation refuses by name.

## Static accounting

The copy-elision patch removes known allocation events. Compute their geometric
sizes without mislabelling them as a measured peak:

```bash
python tools/device_memory_capacity/copy_elision_accounting.py \
  --cells 163842 --edges 491520 --vertical-levels 55 \
  --prior-fixed-mib 6296.5 --prior-bytes-per-cell 93474 \
  --output evidence/capacity/copy-elision-static-x4.json
```

`--prior-fixed-mib` and `--prior-bytes-per-cell` are **required and have no
defaults**, and the tool refuses by name without them.

**Every arm this tool accepts is HISTORICAL, and there is no affine row of
record any more.** Since 2026-08-27 the admission surface is not
`fixed + slope × cells`: it is a card core, plus the Grell-Freitas and YSU
workspaces sized to `min(cells, tile(card))` (which stop growing once the
mesh is bigger than the card) plus a per-cell term, under a margin priced
from the card (`evidence/memory-shape-20260827/`). So this tool cannot ask the
current question. Send a new question about what fits a card to the surface
instead:

```python
from woof.hex.device_admission import KNOWN_CARDS, model_for_card
model_for_card(KNOWN_CARDS["10gib-68sm"]).required_bytes(40_962)
```

The pair in the command above is the **2026-08-24 arm**, measured at WOOF-ArWen
seam pin `0d04db712` after the Grell-Freitas local-memory frame cut
(`evidence/gf-pin-move-measured-20260824/`); pass it only to reproduce an
08-24-era projection. The tool's own refusal text lists every arm it knows
(the retired 2026-08-26 row, the 08-25 converged row, this one, and the 08-20
pre-cut ledger) each with its date, its pin and its evidence directory. A
test refuses any arm labelled "of record" there and reads the retired row's
coefficients out of `device_admission`, so the stale-model-prevention tool
cannot itself recommend a stale model.

To reproduce the copy-elision accounting as it was landed, pass the arm
it was computed against instead: `--prior-fixed-mib 9797.8
--prior-bytes-per-cell 86630`, the 2026-08-20 ledger at pin `629ddb6f0`. They
are not interchangeable: the frame cut moved the fixed term down 3,501.3 MiB
and the slope up 6,844 B/cell, and because those errors partly cancel, mixing
the models yields a plausible verdict rather than an obvious failure. The pair
you pass is recorded in the report's `prior_gap` block.

The sum in that file must not be subtracted from `nvidia-smi` peak. Allocation
events repeat, overlap, and interact with pool retention.

## Fit and gates

After at least two mesh widths and dual runs at each required gate, create a
`gpuwm-hex.capacity-samples.v1` JSON file. Each label must be unique. A complete
entry has this shape (nullable measurements may be omitted until they exist):

```json
{
  "schema": "gpuwm-hex.capacity-samples.v1",
  "samples": [
    {
      "variant": "candidate",
      "label": "x4-candidate-a",
      "cells": 163842,
      "process_peak_bytes": 123456789,
      "success": true,
      "state_sha256": "<64 lowercase hex characters>",
      "seconds_per_step": 1.0,
      "forecast_seconds_per_wall_second": 120.0,
      "pool_live_peak_bytes": 1,
      "pool_total_peak_bytes": 1,
      "physical_device_total_bytes": 12884901888,
      "effective_device_limit_bytes": null,
      "whole_device_limit_enforced": false,
      "limit_includes_non_pool": false,
      "limit_includes_local_backing_store": false,
      "isolated_card": true,
      "source_commit": "<40 hex characters>",
      "command": ["python", "src/hexcore/drivers/run_cuda_v841_full_physics_x4.py"]
    }
  ]
}
```

For a physical <=12 GiB card, record `physical_device_total_bytes`. For an
enforced-limit experiment on a larger card, record the effective byte limit
and set all three limit booleans only after the separate controls prove them.
Then run:

```bash
python tools/device_memory_capacity/capacity_model.py \
  --input evidence/capacity/samples.json \
  --output evidence/capacity/report.json \
  --target-cells 163842 \
  --budgets-gib 12 16 24 32 \
  --headroom-mib 512
```

The report labels affine results `PROJECTION ONLY`. The 12 GiB gate can become
`PASS` only from at least two byte-identical target-cell runs on a physical
12 GiB device, or a validated whole-device limit that includes pool, non-pool,
and CUDA local-memory backing-store allocations. The qualifying process peak
must also leave the declared headroom, which this tool still defaults to
512 MiB. `CUPY_GPU_MEMORY_LIMIT` is a pool limit and is not sufficient for
that gate.

**That flat headroom is RETIRED as an admission margin**, and this tool's
`--headroom-mib` default is the retired constant kept for reproducing a
historical projection: it is a protocol knob here, never the gate. It named
no breakage and it failed by 96 MiB on `v6.75.112676` and 28 MiB on
`v20.80.151649`. What a live decision holds back is the model's own margin,
priced from the card and naming what each part absorbs
(`ShapedFootprintModel.margin_terms()`): the card's RRTMG shortwave workspace,
the largest block that does not scale with the mesh, measured to move the
pool high-water by 1,707.2 MiB when it stops being servable from the free list,
at **1,745.6 MiB on a 170 SM part and 872.8 MiB on a 68 or 70 SM part, plus
11.2 MiB of instrument convention**. The forecast door still takes
`--headroom-mib`, but it now defaults to `None`, meaning that margin; the
retired constant stays computable at
`device_admission.RETIRED_FLAT_HEADROOM_BYTES`.
