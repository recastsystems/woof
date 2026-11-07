# DFI filter 8: bounded RHO filter

Opt-in `dfi_nfilter=8`: **native WRF 4.7.1 fails here; this fixes it, not
bitwise to WRF**. DFI remains off by default; filter 7 remains its default
when initialization is selected. The stock namelist and experiment TOML
doors accept filter 8 without a second hidden flag.

WRF 4.7.1 `share/dfi.F` is byte-identical to the pinned 4.6.1 DFI source:
SHA256 `c7b94989ccdf637ebe756b9dcdff26d5b9ba798852ec74c3415a853a32a4ae0f`.
The public-domain notice is in `../dfi_wrf461_oracle/NOTICE.txt`.
Native 4.7.1 source commit: `f52c197ed39d12e087d02c50f412d90d418f6186`.
Source: <https://github.com/wrf-model/WRF/blob/v4.7.1/share/dfi.F>.

`DFCOEF` (:1097) requests a row of length `2*NSTEPS+1` from RHOFIL but
declares only `NSTEPS+1` output words. RHOFIL (:1629) clears 101 output
words regardless of its argument, then (:1692–1693) copies the entire row.
Its matrix and work vectors also have a fixed bound of 100. These are bounds
defects, independent of GPU arithmetic. The strict unmodified native routine
with bounds checking stops while clearing FROW, at index 11 for extent 0:10.
The probe extracts complete original routines; it does not replace them
with a Fortran restatement.

The smallest repair retains the actual caller contract: construct the full
Quick-Start row in correctly sized scratch, clear `FROW(:)` and copy only
`0:MIN(NSTEP,NOSIZE)`. `optfil_driver` (:821–830) still reflects this prefix
around the filter midpoint. This is the WRF symmetric finite-span wrapper
around the RHO row, not an unmirrored causal filter with a changed valid time.
The native coefficient formulas, REAL products, DOUBLE sums and initialization
type 2 correction are unchanged. Rust retains the two nonzero preceding rows;
native terms outside these rows are zero. Weather-field accumulation and
normalization continue in the existing CUDA kernels, with Rust pass submission.
No host weather-field processing or Python model-step loop is added.

The [WRF technical note](https://www2.mmm.ucar.edu/wrf/users/docs/technote/v4_technote.pdf),
§5.3.1, describes its second-order Quick-Start filter implemented as a finite
weighted sum. The [recursive-filter paper](https://maths.ucd.ie/~plynch/Publications/Tellus-94.pdf),
Tellus 46A (1994), pp. 583–597, §2.2 and Appendix A derive the Quick-Start
coefficients; §3.2 and Appendix B derive the filter matrix row. That
recurrence is the physics retained here.
The [WRF user guide](https://www2.mmm.ucar.edu/wrf/users/docs/user_guide_v4/v4.1/users_guide_chap5.html)
documents the DFI controls. This bounds repair does not claim a new filter
response or universal forecast skill.

Build the whole crate, then run the oracle on a host with the strict native
source, GNU Fortran and Python:

```sh
cargo build --release --locked --offline --manifest-path tools/dfi-runtime/Cargo.toml
python tools/dfi_wrf471_oracle/check.py \
  --source /workspace/native-wrf471/share/dfi.F \
  --library tools/dfi-runtime/target/release/libarwen_dfi.so \
  --output /workspace/runs/port-dfi8/oracle
```

The output preserves the extracted original source, bounded source and patch,
strict compiler flags, original failure, command vectors, every native and
Rust coefficient file, and per-case word comparisons. Original native safe
RHOFIL rows (with a sufficient declared output extent) and original filters
0–7 are distinguished from the bounded-reference results for larger rows.
The native probe is a routine-level failure witness; whole forecasts and
captured weather-state GPU replays are separate evidence.
