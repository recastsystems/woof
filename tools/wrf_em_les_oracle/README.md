# WRF v4.6.1 `em_les` oracle: reproduction

Everything needed to rebuild the oracle and regenerate its receipts. Nothing
here imports woof; the oracle side has to be able to disagree with the engine
it scores.

## Build

```sh
curl -sL -o v4.6.1.tar.gz \
  https://github.com/wrf-model/WRF/releases/download/v4.6.1/v4.6.1.tar.gz
# expect sha256 b8ec11b240a3cf1274b2bd609700191c6ec84628e4c991d3ab562ce9dc50b5f2
tar xzf v4.6.1.tar.gz && cd WRFV4.6.1

source /opt/intel/oneapi/setvars.sh
export NETCDF=<prefix with include/netcdf.inc and lib/libnetcdff.so>
export WRF_EM_CORE=1 WRF_NMM_CORE=0 WRF_DA_CORE=0

printf '34\n1\n' | ./configure          # 34 = GNU gfortran/gcc dmpar
sed -i 's/-w -O3 -c/-w -O3 -std=gnu17 -c/' configure.wrf   # GCC-15 K&R fix
./compile em_les -j 24                  # ~11 min, ~600 MB tree
```

On Ubuntu the system netCDF lives in `/usr/include` + `/usr/lib/<triplet>`,
which is not the `$NETCDF/lib` layout `configure` expects. Build a shim prefix
of symlinks rather than a second netCDF.

**Do not rebuild the pinned campaign build.** It is identified by its binary
hashes, not by a path: `wrf.exe` `bda5e5e6…`, `real.exe` `a300ada0…`.
`PINNED-BUILD-BASELINE-sha256.txt` (in the receipts directory) is the manifest
to diff against before and after any work on these nodes.

## Run

```sh
python3 gen_namelist.py .               # writes namelist.* + run_matrix.json
python3 gen_namelist.py . --check-dry <dir with the committed namelists>
./run_les.sh match_km3_100m ./namelist.match_km3_100m 24
```

`--check-dry` proves the five dry namelists are byte-identical to the
committed ones. `mp_physics` became a parameter in P1 and the dry family must
not have moved when it did; that is a check, not a claim.

`run_les.sh` takes an optional 5th argument, the iofields file, and refuses to
run if it does not match the name the namelist asks for: staging the wrong
one is how a run silently loses its SGS rows.

A grid smaller than ~10 cells per MPI patch is refused by WRF itself
("Reduce the MPI rank count, or redistribute the tasks"). The 39-cell stock
grid needs `-n 9` or fewer; 48 cells needs 4; the 96- and 192-cell matched
grids take 24.

`run_les.sh` stages the WRF `run/` directory with **`cp -L`**. That is not
cosmetic: `run/MPTABLE.TBL` is a relative symlink, and a plain `cp` leaves a
dangling link that makes Noah-MP stop inside `read_mp_veg_parameters` with an
error that reads like a physics failure and is not one.

`iofields_les.txt` adds `tke, xkmv, xkmh, xkhv, xkhh` to history stream 0.
Without it they are restart-only and every SGS quantity is silently
unavailable after the run finishes. WRF writes them **uppercase** even though
the iofields file names them lowercase.

## Score

```sh
python3 score_wrf_les.py <run_dir> <out_prefix> --window-min 30
python3 verify_independent.py <run_dir> <out_prefix>.json   # second opinion
python3 same_instrument.py arwen_kmN.npz <out_prefix>_profiles.npz <label>
python3 spec_compare.py arwen_kmN.npz wrf_wslab.npy <dx> <outdir>
python3 window_spread.py <run_dir> <out.json> --win-min 15 --spinup-min 60
```

`same_instrument.py` is the one that matters for the comparison: it reduces
**both** models' raw per-minute profiles with a single routine, so a
difference in the output is a difference in the models rather than in two
codebases' conventions. It validates on the way in by reproducing WOOF-ArWen's own
published fit from WOOF-ArWen's own arrays.

`INSTRUMENT-HISTORY.md` records which parts of the scorer were changed after
output had been seen, and why. No band is cut anywhere in this lane.

## Moist arm (P1)

```sh
python3 gen_namelist.py .                      # moist probes come out too
./run_les.sh moist_smoke_stocksnd_1h ./namelist.moist_smoke_stocksnd_1h 9 \
             ./input_sounding.wrf_em_les ./iofields_les_moist.txt
python3 score_moist_les.py <run_dir> <out_prefix> --window-min 30
python3 same_instrument_moist.py arwen_moist.npz <out_prefix>_moist_profiles.npz <label>
```

The moist delta from the dry family is three lines and nothing else:
`mp_physics 0 -> 1` (Kessler), `iofields_les_moist.txt` in place of
`iofields_les.txt`, and a moist sounding in place of `input_sounding.arwen_cbl`
(which carries qv ≡ 0). `use_theta_m` stops being moot the moment moisture
exists; it stays **1** for the stock family (what WRF ships) and **0** for the
matched family, because woof stores dry theta and says so
(`woof/core/moist.py:29`, `woof/core/microphysics.py:23`). Matching it is
not a preference.

`score_moist_les.py` is the moist instrument and `score_wrf_les.py` is
unchanged by P1: run **both** on a moist run. The moist one adds exactly what
the dry one cannot see: resolved and SGS `w'θ_v'` and `w'q_v'`, the qc profile
and cloud fraction, liquid-water path, precipitation, and the per-level
fraction of points on WRF's saturated-BN2 arm under WRF's own predicate.

`same_instrument_moist.py` documents, at the top of the file, **the npz
contract the WOOF-side P1 case module must satisfy** so both sides can be
reduced by one routine. Read that before writing the WOOF-ArWen side.

**One name, two heights: the z_i trap.** `zi_thetav_load_m` is the height of
the total-buoyancy-flux minimum. In a clear CBL that is the inversion; in a
cloud-topped CBL it is **cloud base**. Measured on two runs of one capped
family differing only in vapour: the capped-dry anchor reads 1526.3 m (the
inversion, base at 1500 m), the capped-moist draw reads 1274.4514 m, which is
`cloud_base_m` to every digit. A moist-vs-dry z_i difference is therefore not
a boundary-layer depth change, and a cloudy-vs-clear comparison of this metric
compares two different heights while looking like one quantity. Reduce it the
same way on both sides, say which meaning applies whenever a cloudy z_i is
quoted, and use `cloud_top_m` or the theta profile if you want the inversion
in a cloudy case.

### The matched moist sounding is not chosen here

`gen_namelist.py` emits the matched moist arms only under
`--matched-moist-sounding <asset>`. That is deliberate: which sounding defines
the moist case is a case definition with physics consequences, the shipped
`test/em_les/input_sounding` measurably does not condense under the em_les
forcing (see `docs/internal/receipts/les/moist-oracle-sounding-probe-*.md`),
and a recipe that quietly picked one would have decided it. Two shipped
candidates are staged as assets:

| asset | WRF source | sha256 |
|---|---|---|
| `input_sounding.wrf_em_les` | `test/em_les/input_sounding` | `6aed509b22519dcd933e3dc6ad57b60dbd0b7aa7a8b0edf983bb3c808a905b90` |
| `input_sounding.wrf_em_les_shalconv` | `test/em_les/input_sounding_shalconv` | `ff044b473cc56389119357c9a4cb6b89edac5021f0bd043d1d208c410b26c670` |

Neither is committed to this repository: both are WRF v4.6.1 assets,
identified by the hashes above and reproduced from the pinned tarball.
