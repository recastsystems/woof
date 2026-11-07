# BouLac PBL oracle, WRF v4.6.1 (bl_pbl_physics = 8)

Column oracle for the port of the Bougeault-Lacarrere TKE scheme,
`phys/module_bl_boulac.F` (sha256
`68285f1457fe62e37ad1d76680d3c69e4a48a6baca93348f6f8a15c5fb40d871`; the
v4.7.1 file is byte-identical).  The port is `woof/core/boulac.py` and
`woof/core/kernels/boulac.cu`; `tests/test_boulac_wrf461_parity.py` grades
it word for word on the fixtures this directory writes.

```
bash tools/boulac_wrf461_oracle/build.sh $WRF_SOURCE_ROOT BUILD_DIR [WRF_SCALAR_BUILD_ROOT]
```

## Files

| file | what it is |
|---|---|
| `make_cases.py OUTDIR` | the input grids (column regimes, BEP urban columns, branch probes) |
| `run_bl_boulac.F90` | the driver: WRF's 3-D `BOULAC` entry for N consecutive steps, TKE and DLG_BEP carried by the scheme, the state advanced by its own tendencies, every step's inputs and outputs recorded |
| `build.sh` | builds the variants below, runs every grid, compares the variants, collects `BUILD/fixtures` |

`woof/verify/boulac_oracle.py` reads the fixtures; the packaged copies are
`tests/data/oracles/boulac/` with their `SHA256SUMS`.

## Variants

* **wrfobj** (the fixtures of record when the scalar build root is given):
  the driver linked against `phys/module_bl_boulac.o` of the scalar WRF
  build, the object inside the 0 ULP referee `wrf.exe` (gfortran 13.3.0,
  `-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize`).
* **o2**: the pinned source compiled here with those flags.  Required
  byte-identical to wrfobj.
* **o0**: the pinned source at -O0.  Evidence.
* **snan**: o2 with every local real initialised to a signalling NaN.
  Required byte-identical: no uninitialised local reaches an output.

On S4 (2026-10-08) all three are byte-identical to wrfobj
(`variants-report.txt`).

## Measured semantics the port follows

* The object calls `powf` and nothing else from libm (`nm -u`): four call
  sites, `**(1./3.)`, `**(-2./3.)`, `**(2./3.)` and `**.5` (the U and V
  drag share one).  `x**2.` is folded to `x*x`.  `**.5` stays a pow call
  (no `-funsafe-math-optimizations`), so the port calls WOOF's own float32
  pow there, not sqrt.
* `t2_1D`/`w2_1D` (the `-2./3.` and `2./3.` powers) feed only the
  `igamma = 2, 3` arms; BOULAC hard-codes `igamma = 1`, so they reach no
  output and the port does not form them.
* `ceps_b = 1/1.4` is the float32 quotient 0x3F36DB6E.

## Grids (135 columns)

| grid | nz | dt | steps | columns | what |
|---|---|---|---|---|---|
| rural40 | 40 | 30 s | 10 | 48 | 12 regime families x 4 variants: deep and shallow convective, stable with a low-level jet, neutral windy, strong shear, high terrain (thin air, coarser dz), cloud-topped, isentropic (pblh = 0 with hfx > 0, bbb == 0 and th(izz) == th(iz)), exactly repeated theta pairs, calm lowest level with ust > 0 (WRF's -inf drag), the exch/TKE floors, deep developed TKE |
| rural61 | 61 | 10 s | 6 | 48 | the same families on a stretched 20 m-first-layer grid |
| bep40 | 40 | 20 s | 10 | 24 | flag_bep: frc_urb2d 0, 0.25, 0.6, 1 x 6 regimes, nonzero implicit/explicit sources for U, V, theta, QV and TKE, building length scales (the harmonic-mean arm of length_bougeault), reduced sf/vl, DLG_BEP written back and carried |
| calmprobe40 | 40 | 30 s | 1 | 3 | ust = 0 on a calm lowest level (WRF's 0/0, the declared divergence), ust = 0 with wind, ust = 1e-3 calm |
| idiff40 | 40 | 30 s | 3 | 12 | idiff = 1: TKE, lengths and exchange coefficients only |

## The declared divergence

`calm-ust0`: WRF's rural drag `-ust*ust/dz(1)/|V(1)|` is 0/0 and WRF writes
NaN through the whole U and V columns (RUBLTEN, RVBLTEN, WU_TUR, WV_TUR).
The port writes the drag's limit, -0., and is finite there; every other
word of that column is WRF's.
