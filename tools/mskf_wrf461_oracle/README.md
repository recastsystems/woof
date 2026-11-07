# MSKF native source and proof tools

`source-pin.json` pins WRF v4.7.1 `phys/module_cu_mskf.F`. It is byte
identical to the v4.6.1 module used by the retained column oracle. The
upstream public-domain statement is copied unchanged in `LICENSE.WRF`.
The module's embedded aerosol-routine permission notice is retained in
`woof/core/kernels/mskf_aerosol.cuh`. No monthly CESM climatology is
bundled; its data licence has not been established.

`build.sh` builds the retained synthetic column oracle. These fixtures are
regression checks and do not establish replay of a real forecast. The
defined-read patch and its native undefined-memory witness are documented
in that script. A replay of pristine real-run calls is a separate proof.

Instrument a private copy of the pinned native source before compiling
with `-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize`:

```sh
python tools/mskf_wrf461_oracle/instrument_native.py \
  PRIVATE_WRF/phys/module_cu_mskf.F --receipt instrumentation.json \
  --nz 49 --steps 1,241,481
```

Instrumentation adds stream writes around `MSKF_CPS`; it changes no
scientific assignments. Each selected call captures 16 interior columns
per native tile, including the initial values of every INOUT output.
Capture files are little-endian streams named by step and tile start.
The header has 12 int32 words: nz, columns, trigger, KTAU, STEPCU,
warm_rain, F_QR, F_QI, F_QS, aercu_opt, KF_EDRATES, adaptive.
Seventeen float32 scalars follow: DT, DX, CUDT, aercu_fct and the 13 WRF
thermodynamic constants. Each input record then contains grid i/j,
the nine `mskf.INPUT_3D` arrays, interface W, four `mskf.INPUT_2D`
values, 26 initial volume outputs, seven surface outputs and CU_ACT.
Output records contain the same 26 volumes, seven surface outputs,
CU_ACT and updated W0AVG. CU_ACT is stored as float32 0 or 1.
Inactive aerosol-package slots are explicit zero extensions. The writer
never indexes WRF's unallocated aerosol dummy arrays at `aercu_opt=0`.
The five entrainment/detrainment slots receive the same treatment when
`kf_edrates=0`. Replay receipts count native words separately from these
extensions. Native warm starts restore the allocated packages and rebuild
only the absent inactive slots as zero; GPUWM checkpoints retain all 35.

```sh
GPUWM_WRF_EXACT=1 GPUWM_NUMERICS=strict \
  python tools/mskf_wrf461_oracle/replay_native_capture.py \
  PRIVATE_RUN/mskf-*.bin --out replay.json --graph
```

The verifier compares every float32 word, including signed zero and
unchanged held-column state. Its receipt records capture hashes,
per-field mismatch counts and the first differing grid locations. The
current reader accepts trigger 1 with aercu_opt=0 and refuses other
headers explicitly. `--graph` also checks that a warmed call can execute
inside a CUDA graph with deferred health checks.

Normal forecasts use CUDA binary64 math while preserving native operation order.
Float32 thermodynamics retains its pinned functions because CUDA float32 libm
amplified small cloud-water differences past the existing regression bound.
FMA contraction changed threshold-sensitive aerosol number fields beyond the
regression tolerance, so it is disabled in both modes. Strict replay is opt-in.
The two modes have distinct compiled cache keys and checkpoint identity.

`kfeta_trigger=2` is admitted for MSKF because the pinned native file's
`MSKF_eta_PARA` declares `trigger`, `TPART_H0` and `TPART_V0` at lines
4030 and 4039-4040 and never reads them. Its executed trigger remains the
Kain branch at 4573-4593. The driver-only calculation at 3687-3831 is dead
with respect to all module outputs. The GPU therefore omits that calculation.
The pin can be audited with `rg -ni 'tpart|trigger' module_cu_mskf.F`.
