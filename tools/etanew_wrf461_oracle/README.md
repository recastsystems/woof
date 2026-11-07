# WRF v4.6.1 Eta Ferrier column oracle (mp_physics = 95)

Reference outputs of WRF's own `phys/module_mp_etanew.F` (`ETAMP_NEW`,
package `etampnew`: moist `qv, qc, qr, qs`, scalar `qt`, state
`f_ice_phy, f_rain_phy, f_rimef_phy`, restart state `mp_restart_state,
tbpvs_state, tbpvs0_state`), compiled byte-unmodified from WRF tag v4.6.1
(commit `d66e442fccc04111067e29274c9f9eaccc3cef28`) with a stub framework
and a one-point column driver.  The port it grades is
`woof/core/kernels/etanew.cu` (`woof.core.etanew`).

## Files

| file | role |
|---|---|
| `make_columns.py` | Deterministic column generator (closed-form soundings, no RNG): 80 columns x 50 levels in ten regimes. Writes `columns_inputs.csv` (binary32 hex) and `columns.in`. |
| `run_etanew.F90` | Driver. Per column: `ETANEWinit` (GSMDT = 0, restart = false) at the column's time step, the column's own fractions installed, then `ETAMP_NEW` for 10 steps with theta, qv, qt, the fractions, RAINNC and the restart vectors carried. Writes `levels.csv`, `surface.csv` every step and `init.csv` (restart vector and both saturation tables) per distinct time step. |
| `stub_wrf.F90` | `wrf_error_fatal`, `wrf_error_fatal3`, `wrf_debug`, `wrf_dm_on_monitor` (true), `wrf_dm_bcast_bytes` (no-op). |
| `build.sh` | Checks the pinned commit and that the module and `run/ETAMPNEW_DATA.expanded_rain` are unedited; builds `-O0` (reference), the WRF scalar referee's flags (`-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize`), WRF's own compiled `module_mp_etanew.o` from a full build (optional third argument), and a `--coverage` twin; writes `SHA256SUMS`; compares the builds. |
| `compare_etanew.py` | Bitwise comparator: per field, rows differing, max ULP, max abs and relative difference, worst column and regime. `--subset` grades only the rows the reference holds. |
| `run_port.py` | Runs WOOF's port on `columns_inputs.csv` and writes the same files. `--plant OLD=NEW` edits one literal of the compiled kernel source (the planted-bit control). |
| `make_fixtures.py` | Cuts `tests/data/oracles/etanew/` from a build: steps 1, 2, 10 of every level, every surface row, the restart vector, the table digests. |

## Rerun (CPU box)

```bash
source /work/wrf/env.sh
bash tools/etanew_wrf461_oracle/build.sh $WRF_SOURCE_ROOT $L/oracle-build \
    /work/wrf/build/scalar/WRF-4.6.1
python3 tools/etanew_wrf461_oracle/make_fixtures.py $L/oracle-build \
    tests/data/oracles/etanew
```

Then on a GPU, strict arithmetic:

```bash
GPUWM_WRF_EXACT=1 python tools/etanew_wrf461_oracle/run_port.py \
    tests/data/oracles/etanew/columns_inputs.csv $L/port-strict
python tools/etanew_wrf461_oracle/compare_etanew.py \
    tests/data/oracles/etanew $L/port-strict --subset
```

Toolchain of record: gfortran 13.3.0, glibc 2.39 (Ubuntu 24.04).  The
`-O0` object imports scalar `expf`, `logf`, `powf`, `lroundf` and no
libmvec symbol.

## What the reference showed

* `-O0`, the WRF referee flags and WRF's own object write bitwise-identical
  `levels.csv`, `surface.csv` and `init.csv`: no flag-dependent arithmetic
  on this column set.
* gcov: every line of `EGCP01COLUMN`, `CONDENSE`, `DEPOSIT` and
  `EGCP01DRV` runs (99.24 % of the module; the rest is the NMM-only
  branch, the unit-search and open-failure fatals, and the negative-DTPH
  arm of `MY_GROWTH_RATES`).  The edge columns were added to reach the
  last three: the `VEL_RF(:, Nrime)` end of the rimed fall-speed table
  (30 g/kg unrimed ice above 0 C pushes the second pass's rime factor past
  RFmax) and the 1 mm mean-drop ends of the rain-rate and rain-content
  lookups (40 g/kg rain).
* gfortran 13 `MAX(a, b)` is `a > b ? a : b` and `MIN(a, b)` is
  `a < b ? a : b` (argument order kept; measured with signed zeros and
  NaN at -O0 and -O2).

Facts a port must reproduce rather than "fix":

* ARW never initializes `qt`: real.exe leaves the scalar at zero and only
  the NMM build syncs it from qc + qr + qs (`:229-255`).  The column
  oracle feeds `qt` directly.
* `ETANEWinit(restart = .false.)` resets the three fractions to 0, 0, 1;
  the driver installs the column's own fractions after the init.
* At the model top (`L = 1` after the flip) `EGCP01DRV` forces the
  condensate floor and F_ICE = 1, F_RAIN = 0, F_RIMEF = 1 (`:436-440`).
