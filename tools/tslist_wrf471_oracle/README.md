# Native WRF station time series

CUDA samples each committed domain step and Rust writes the native eight-file,
non-solar, non-WRF-Chem layout. Stock `&domains` controls are `ts_buf_size`,
`max_ts_locs`, `max_ts_level`, and `tslist_unstagger_winds`. The adjacent fixed-name
`tslist` activates output. A missing file or zero maximum locations disables it.
The native pin is WRF v4.7.1, `f52c197ed39d12e087d02c50f412d90d418f6186`.
Public source provenance and the public-domain notice are in
[../tslist_bridge/SOURCE.md](../tslist_bridge/SOURCE.md).

Copy a strict native build into private scratch. `instrument.py` verifies the
public hash of `share/wrf_timeseries.F` and inserts capture I/O after the native
calculation without changing numerical statements. For the physical-edge phase
witness, apply `instrument_boundary.py` to the private `dyn_em/solve_em.F`.
Rebuild `em_real` with at most eight jobs; qualification used two. Run `real.exe`
and `wrf.exe` with two MPI ranks and strict floating-point flags. The supplied
January 2000 fixture needs matching met_em files and native runtime tables,
including `CAMtr_volume_mixing_ratio`. Each capture run uses exactly one station,
so the owning rank creates one diagnostic stream. Retain source, configuration,
forcing and executable hashes. East and north phase cases use stations (73,30)
and (35,60); the interior case uses (35,30).

Build and replay the actual production code:

```sh
cargo test --offline --manifest-path tools/tslist_bridge/Cargo.toml -j2
cargo build --release --offline --manifest-path tools/tslist_bridge/Cargo.toml -j2
export WOOF_TSLIST_LIBRARY="$PWD/tools/tslist_bridge/target/release/libgpuwm_tslist.so"
export PYTHONPATH="$PWD"
GPUWM_WRF_EXACT=1 python tools/tslist_wrf471_oracle/replay.py CASE/tslist-real-input.bin strict.json
GPUWM_WRF_EXACT=0 python tools/tslist_wrf471_oracle/replay.py CASE/tslist-real-input.bin fast.json
GPUWM_WRF_EXACT=1 python tools/tslist_wrf471_oracle/staged_replay.py CASE/tslist-real-input.bin EAST/tslist-real-input.bin NORTH/tslist-real-input.bin staged.json
GPUWM_WRF_EXACT=1 python tools/tslist_wrf471_oracle/async_write.py CASE FRESH_OUTPUT async.json
```

The staged replay compares 360 initialized float32 words: 120 per real native
interior/east/north capture. It strips the captured outer tangential halo from
the runtime-shaped inputs. The production boundary kernel reconstructs only the
needed packet from the separately captured pre-forcing source; both 44-word
phase witnesses equal native's stored ghosts exactly. The production neighbor
and finish kernels then reproduce all initialized native results. The native
allocation tail contains 210 uncomputed words per case; the reader counts and
excludes those words only. Strict proof requires zero initialized differences.
Default production uses fast numerics. The asynchronous writer check compares
all eight native headers and first records, fills both bounded slots with
buffer size three, then verifies rollback and continuation. Later samples in
that component check reuse captured inputs and are not a forecast.

The real front door is:

```sh
GPUWM_WRF_EXACT=0 python -m woof go --wrfinput CASE --outdir OUTPUT --products none
```

Shared-card qualification adds `--allow-shared-gpu --no-supervise`. The normal
run-stamped directory owns the eight station files. There is no per-step
weather readback or CUDA synchronization on the stepping host thread. Bounded
batch reuse can apply backpressure; checkpoint and close deliberately drain.
Rust validates every saved file prefix before truncating any uncommitted tail.
A verified ensemble roster may also return a pending member to its initial
state: the Rust writer checks all expected headers before discarding that
attempt's station rows. Fresh runs still refuse existing station files.

Both fixed-nest feedback modes sample each domain's committed clock before
parent feedback. Streamed runs gather owning columns and separate canonical
neighbor packets before tile reuse, then join those packets using CUDA events.
Physical tangential packets are captured before final specified forcing;
normal faces are read afterwards. Restart reconstructs this temporary workspace
from the durable station plan. The plan retains native's once-only placement
across state replacement; the existing global moving-nest admission rules still
apply. No new TS-specific relocation refusal remains.

The lane evidence includes strict replay, the complete real three-hour forecast,
four-hour Rust maps beside native WRF and archived observations, unchanged
weather histories, both feedback modes with parent and child edge stations,
resident and streamed checkpoint continuation, two distinct production ensemble
members and an actual sealed-roster rollback, stage timings and native formatted
WRITE checks. WRF-Chem O3 and WRF-Solar station layouts are separate compile-time
contracts outside this standard eight-file build. The optional standalone tile
CUDA-graph mode has stable boundary-job addresses, but is not claimed as a
separately qualified production forecast route.
