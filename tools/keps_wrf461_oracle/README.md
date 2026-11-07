# K-epsilon PBL oracle, WRF v4.6.1

Historical synthetic evidence only. The current port's authority is the
real-input WRF v4.7.1 replay in [../keps_wrf471_oracle](../keps_wrf471_oracle).
This older diagnostic patch is not applied to the production kernel.

The WRF numbers for `bl_pbl_physics = 17` (`phys/module_bl_keps.F`, Zonato
et al. 2022): prognostic TKE, dissipation rate and turbulent potential energy
with a surface-layer Prandtl number and a counter-gradient heat flux.

```
bash tools/keps_wrf461_oracle/build.sh <WRF_SOURCE_ROOT> <BUILD_DIR>
```

`module_bl_keps.F` must hash to
`7e07e1125023e589b29ca688b2e9de69fbb3ee01794851d72b6143e5845d3584` (WRF
v4.6.1); build.sh refuses anything else.  The module has no USE statement and
calls nothing outside itself, so there is no stub: the reference object's
undefined symbols are `expf`, `powf` and `tanhf` (`libm-report.txt`).

## Two references from one file

* `pristine`: the byte-unmodified module, kept as evidence.
* `declared`: the module with `declared-divergences.patch`, the fixture of
  record.  The patch fixes four defects and nothing else:
  1. GET_PBLH's maximum-TKE scan compares against `ttke`, an undeclared and
     never-assigned variable (:822-823).
  2. `radsum` is accumulated without being set (:1065).
  3. The first-level y-momentum flux reads the x wind (:363).
  4. z/L = br*psim*psim/psih is 0/0 in the surface layer's neutral regime
     (psim = psih = 0); the patch takes that regime's own z/L = 0 into the
     same clamps, which is the answer WRF's -O2 build reaches through
     maxss's NaN rule and an -O0 build does not (it carries the NaN into the
     whole column).

The driver also never copies the model-top face pressure into `p8w1D`, but
GET_PBLH's exit safeguards keep the radiative sum below level kte-2, so that
word is never read and nothing is patched for it.

## The reference flags are WRF's

Both references are built with the WRF scalar build's own flags (`-O2
-fno-fast-math -ffp-contract=off -fno-tree-vectorize`).  Measured on gfortran
13.3: an -O0 build calls `powf` for `x**2.`, `x**1.` and `x**(-1.)`, which is
not correctly rounded, where the compiled WRF folds those three exponents to
`x*x`, `x` and `1/x`; the two disagree in last bits wherever a square is
taken (`divergence-evidence.txt` lists the -O0 build against the reference,
word counts per field).  The -O0 objects are built only as evidence.

## Fixture sets

72 columns per set, eight consecutive calls of WRF's own 3-D `keps` over the
whole tile (the driver carries column arrays across the columns of one call,
so one call per column would not test that):

| set | grid | calls | state |
| --- | --- | --- | --- |
| Acold | 40 levels to ~17 km, dt 30 s | itimestep 1..8 | WRF's `xtime <= 2` reset |
| Adev | same | 3..10 | developed TKE/dissipation/TPE |
| Abep | same | 3..10 | Adev with flag_bep, an urban fraction and nonzero BEP source terms |
| Bcold | 20 levels to ~2.2 km, dt 10 s | 1..8 | reset; PBL tops in the last levels the search reaches |
| Bdev | same | 3..10 | developed |

Column families: convective day, stable night, neutral windy, cloud-topped
with cloud-top radiative cooling, BEP-blended urban surface fluxes, and
branch probes (uniform wind, the neutral regime, `br = -0`, tiny `ust`, zero
fluxes, strong heating, a calm first level, thin layers, roughness near half
the first layer, positive-psi stable).

`Abep` is never committed: build.sh asserts every output word equals Adev's
(`bep-identity.txt`), which is the proof that keps reads none of the BEP
source terms, urban fraction or flag it receives.

Between calls the program applies WRF's update `x = x + xBLTEN*dt` in float32
and carries the scheme's own state untouched; only the initial state and the
outputs are written, and `woof.verify.keps_oracle.replay_inputs` rebuilds
every call's inputs with the same float32 update.

## Files

`woof/data/keps/oracle/keps-<set>.bin.xz` (raw float32, xz) and
`keps-<set>.manifest` (field, levels, columns, word offset), plus
`oracle-sha256sums.txt`, `libm-report.txt`, `compiler.txt`,
`bep-identity.txt` and `divergence-evidence.txt`.  `compare_sets.py` prints
per-field word differences between two fixture files.
