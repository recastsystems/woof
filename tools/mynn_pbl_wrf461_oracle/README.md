# WRF 4.6.1 MYNN PBL level-2 oracle

This harness compiles the pinned, unmodified WRF sources
`phys/module_bl_mynn_common.F` and `phys/module_bl_mynn.F`, then calls
`mym_level2` for four eight-level profiles. The 28 output rows exercise stable,
convective, exact-neutral, and moist/cloud-modified buoyancy gradients.

It also calls `get_pblh` and `scale_aware` for four ten-level columns spanning
convective land, stable land, marine, and low-TKE cold-pool conditions. The
resulting `pblh-scale.csv` contains the complete column inputs and repeated
column output identity for independent reproduction.

Finally, `mixlength.csv` calls the default `bl_mynn_mixlength=1` path for
stable, convective, high-shear, and active-EDMF columns. It records all column
inputs plus WRF's interface `el` and `qkw` outputs.

The same four harnesses accept an optional second argument `2` to select
`bl_mynn_mixlength=2`. `build.sh` also emits `mixlength2.csv`,
`initialize2.csv`, `turbulence2.csv`, and `driver2.csv`. The local-length
fixture adds weak buoyancy, zero scale factor, a deep boundary layer, and
high TKE columns, for 96 levels total. The other fixtures cover five
initialization columns, four turbulence columns, and five two-step driver
columns. Inputs and outputs are recorded together. Source, harness and CSV
hashes are in `tests/data/oracles/mynn/mixlength2-provenance.txt`.

`tests/test_mynn_mixlength2.py` compares these WRF outputs with the CPU and
CUDA ports. The CPU leaf, initialization, turbulence and warm-driver checks
require exact equality. The cold driver permits one ULP for `qke`; its
remaining fields are exact. The assembled option-2 CUDA driver has measured
cold and warm bounds in `tests/test_mynn_mixlength2.py`, reproduced on two
GPU architectures with NVRTC 13.4 and on RTX PRO 6000 with NVRTC 12.9,
and documented in `docs/dev/mynn-mixlength2.md`.

The cold driver passes vapor `sqv` into `mym_initialize` as WRF does, rather
than the total-water argument used by the later turbulence call. This fixes
the former resolved-cloud cold-start divergence; the option-1 CPU driver is
now exact on both steps and all five columns.

Run:

```bash
./build.sh /path/to/WRF-4.6.1 /absolute/build/directory
```

The generated `pbl-level2.csv`, source hashes, compiler identity, and validator
result form the first numerical oracle for the coupled MYNN PBL port. This
oracle alone does not admit `bl_pbl_physics=5`.

`run_driver.F90` exercises the assembled driver for two calls over five
columns. Its `snow_anvil` column sets `FLAG_QS=.TRUE.`, carries nonzero snow
with no cloud ice above the inversion, and records `sqs3d` alongside every
input and output. This directly pins the wrapper/driver snow plumbing against
the unmodified WRF v4.6.1 source rather than inferring it from the WOOF-ArWen port.
The source tree is tag `v4.6.1` at
`d66e442fccc04111067e29274c9f9eaccc3cef28`; GNU Fortran 13.3.0 produced
the 300-row fixture. The following SHA-256 values are the historical
option-1 generation receipt, before the optional selector argument was
added. Current harness hashes are in `mixlength2-provenance.txt` above:

- `module_bl_mynn_common.F`:
  `71cc8a9a77280fff950b89223cc65d6ad62b1844f48fb9cf8f85e17e7313ae17`
- `module_bl_mynn.F`:
  `7fde5fc9d9760c106a400416feb127e5b18f6ba6f03b8aaf0ac8f2c1af2766e2`
- `run_driver.F90`:
  `8668f0c74168c34730bffabe5656693f209ebad5dc3a1a9dd1f3f92e32c18a03`
- `driver.csv`:
  `f608d456eaa21878491985c3ed4d5bde70896b452f3eb9b71076d6466fdbf955`

## Wrapper column oracle

`build_columns.sh WRF_SOURCE_ROOT BUILD_DIR [oracle|stock]` grades the MYNN
entry WRF actually calls, `mynnedmf_wrapper_run`, rather than the driver. It
compiles the five unmodified WRF modules with no stub and writes the columns
from `wrapper_columns.py gen`. It runs `run_wrapper_columns.F90` for three
option sets and both mixing lengths, then packs one fixture per run.

- The `oracle` flavour (-O0 -ffp-contract=off) is the 0 ULP reference.
- The `stock` flavour uses WRF's own GNU optimisation flags, which records
  how far a stock-built WRF moves from it.

`wrapper_columns.py grade FIXTURE` runs WOOF's `mynn_pbl_step` on the GPU over
the same columns, step by step (`replay`) and free-running (`free`), and
prints per-output word counts and max ULP. Fixtures and their provenance are
described in `tests/data/oracles/mynn/README.md`.
