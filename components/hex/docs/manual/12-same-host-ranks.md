# Same-host resident partitions

`forecast` accepts `--samehost-devices 0,1`, `--samehost-layouts mesh.part2.K3.npz`
and `--samehost-part mesh.graph.info.part.2`. Each listed ordinal owns one
resident partition. Repeating an ordinal, such as `0,0`, checks concurrent
ranks on one card. The ordinary single-card command is unchanged.

The partition builder supports closed and regional meshes. Regional layouts
retain missing-cell sentinels and each rank's native garbage element. The LBC
pool gathers the rank's real global rows before the existing CUDA derivation.
Every rank also carries the canonical garbage sounding donor and its K=3
neighborhood, so the derived garbage columns use the one-card donor.

The executor uses the existing dycore round sites, one compute stream per rank,
device byte packets, peer copies when available and pinned host staging
otherwise. `--samehost-transport staged` selects the staged route explicitly.
Each round enqueues producer and consumer CUDA fences. Compute streams wait
for all consumers before packet or pinned-buffer reuse. A completion worker
retains field owners until its blocking CUDA event finishes; exchanges do not
poll events on the host. A failed rank stops the group. A shared clock advances
only after every rank completes the step.

Memory admission sums local resident-rank predictions on each card. This is a
derived use of the existing footprint row, pending measured multi-card peaks.
The card margin is charged once after the sum. Preflight verifies layout
ownership, file hashes, device assignments and admission, then runs the ordinary
host preflight. It does not construct rank stacks or establish identity.

`--samehost-reference boundary-fingerprints.jsonl` compares reconstructed owned
regions with a single-card reference at each recorded boundary. With a
reference the default interval is one step. Without a reference the default is
zero: initial and final boundaries only, because hashing the whole committed
state after every step costs several seconds per step on a large mesh.
`--samehost-fingerprint-every N` sets the interval explicitly in either case. History uses the existing diagnostic preparation and writer;
the receipt records every published field hash, backend boundary hashes and
final dynamics-forcing hashes. The three named, unconsumed staggered-u batch
carriers are retained as per-rank artifacts instead of silently discarded.

Qualification is incomplete. Full-physics history identity, the regional
contract decks and distinct-card peer/staged scaling must pass before this
path can be treated as a forecast-speed result. Ordinary GIL Python can run
the threads, but host launch work may serialize. Measure a compatible
free-threaded interpreter separately on a multi-card box.

## MPAS-physics route: one process per card

On the MPAS-physics route (the hex default), `--samehost-devices` runs one
process per card by default on Linux: rank 0 stays in the forecast process and
ranks 1..N-1 are child processes that hex starts itself (no MPI, no mpi4py, no
second Python). Each rank has its own interpreter and sees only its own card,
so its host launch work runs on its own core on any CPython, with or without
the GIL. Halo packets cross between processes through shared host memory
registered for DMA in both, ordered on the cards by flags in that memory; no
host thread waits for a card. The rounds, packets, byte kernels and gather
order are the thread executor's, so the output words do not depend on the
executor.

`--samehost-ranks threads` (or `WOOF_HEX_SAMEHOST_RANKS=threads`) keeps every
rank a thread of the forecast process. On a Python with the GIL that executor
prints a warning, and its receipt records `rank_host_parallelism:
"serialised (GIL)"`. `samehost-rank-launch.json` records the executor, each
rank process's PID, interpreter and CPU seconds.

Measured on 8x RTX 5090 (no peer access), r700 zoom globe (286,530 cells),
convection_permitting, fast numerics, plain CPython 3.12: see the hex 0.4.10
changelog. The WOOF-physics route still runs rank threads.

## Rank restart

The forecast door forwards `--checkpoint-hours 3` and `--restart-from CHECKPOINT` to the single-card, same-host rank and one-card full-step LTS executors. A same-host checkpoint is written at a drained history boundary and includes every rank's local state, halos, saved diagnostics, physics restart state and pending dynamics tendencies. Resume with the original `--start-time` and final `--hours`, the same grid, init, static, partition, backend and configuration. Its identity rejects a different constructor or physics selection before constructing ranks.

`--jedi-background` and `--da-analysis` reach the executor too. The same-host rank path still refuses them: it does not assemble rank-local physics fields and edge winds into the global JEDI contract or scatter a recoupled analysis to rank halos. Multi-card full-step LTS also refuses cycling because class-to-rank checkpoint transfer does not persist backend accumulators or pending tendencies. One-card LTS supports cycling. These are remaining implementation gaps.
