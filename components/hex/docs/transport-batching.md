# Bounded scalar transport batching

The regional aerosol speed gate is **CLEARED** on the assigned 167254-cell,
200-step RTX 5090 case: aerosol/mp28 loop 1.021x and runtime 0.990x, with the
full before/after output byte witness and fast/strict device oracles passing.


`cuda_transport_v841.py` previously launched each scalar separately, even on
meshes where several species fit in the limiter workspace. It now batches
contiguous species under a 2 GiB scratch ceiling and a conservative free-memory
limit. The estimate includes the live edge, cell and interface arrays plus
12.5% allocator headroom. At 167254 cells, 501756 edges and 55 levels, the
budget permits three species per pass. The previous 655362-cell memory-census
case remains one species per pass. Explicit pass-width controls and declared
halo exchange schedules take precedence.

Each RK call also classifies old, stage and tendency words with a fused
integer OR reduction across cells at each level, followed by a reduction
across levels on the GPU. Only one word per species returns to the
host. Tracers with three bitwise positive-zero inputs share one computed
transport result. Copying that result, rather than assuming the answer is
positive zero, preserves signed zeros and non-finite flow arithmetic. A
nonzero tendency or changed stage immediately reactivates its tracer. Halo
exchange and CUDA graph capture retain their scheduled path without this
classification. The classifier measurement fell from 22.302 to 2.536 ms per
three-stage step; the host scheduling read remains.

No transport CUDA arithmetic body changed: the complete 76791-byte source
string has SHA-256
`ad305a38f1a634b85317b6a9c34fa3463dc2d88437232ef45cd0c2661d48758f`,
identical to `0dbb67aae837711adbb5bfab7f2f438eca2e1f57`. Source-file pins were
re-derived with `tools/repin_source_tables.py --write`; archived measurement
hashes and scopes were retained. This does not constitute a new GPU or
regional forecast certification.

Controls remain available on every physics route:

| Control | Default | Purpose |
| --- | --- | --- |
| `WOOF_HEX_TRANSPORT_WORKSPACE_BYTES` | 2147483648 | Maximum automatic scratch budget; at least the historical one-species pass |
| `WOOF_HEX_LIMITER_SPECIES_PER_PASS` | automatic | Explicit width; 1 restores serial species, 0 batches all |
| `WOOF_HEX_TRANSPORT_ZERO_REUSE` | 1 | Set 0 to disable repeated-zero reuse for comparison |

`tests/test_transport_species_batch.py` checks both scheduling and real
transport arithmetic. Host tests compile the unchanged CUDA bodies as C++
with thread-index shims, then compare every output word for serial, batched,
all-species and automatic launches. The cases cover RK2/RK3, density splitting,
regional masks, both launch geometries, sources, reactivation, signed zeros,
subnormals and non-finite-flow refusal. These host tests do not substitute for
NVRTC/device execution. The GPU counterparts run with
`WOOF_HEX_TRANSPORT_BATCH_GPU_TEST=1` inside an allocated card, separately with
`HEX_TRANSPORT_TEST_NUMERICS=fast` and `strict`.

The release gate retains both comparisons: the measured native
`convection_permitting` reference and the coordinator-selected equivalent-work
`arwen_mpas_column` with `config_microp_scheme=mp_thompson_aerosols` (WRF mp28).
The latter route exists and resolves to `thompson_aero_column`; its public row
uses YSU/Noah-MP, so this is the requested tracer-work referee, not a claim of
identical scientific configuration or forecast fields. Its fresh 200-step
measurement is recorded in the linked receipt. Component attribution remains
separate from plain forecast timing; launch count alone is not evidence.
