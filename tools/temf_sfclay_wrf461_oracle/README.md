# TEMF surface-layer oracle (WRF v4.6.1, sf_sfclay_physics=10)

`build.sh WRF_SOURCE_ROOT BUILD_DIR [WRF_BUILD_ROOT]` drives the byte-pinned
`phys/module_sf_temfsfclay.F` (sha256 `70f2ed6b...`) through its own 3-D
entry `TEMFSFCLAY`, after `TEMFSFCLAYINIT`, over 78 columns and 8 steps, and
writes every input word before each call and every output word after it.

Catalogue (13 regimes x 6 variants, `run_temf_sfclay.F90`): convective land,
stable night land, neutral windy, calm (u = v = 0: the S < 1e-15 limiter
with N2 < 0, N2 > 0 and N2 from moisture only), sea, rough, very stable
(Ri clamp), MAVAIL = 0, TE_TEMF below TEmin with deep convective HD_TEMF,
near calm, cold / high latitude / equator, hot desert with the first level
below 10 m, and six columns whose zm/znt lands on a binary32 word where the
C library's log10f is not correctly rounded.

Between steps the driver plays the parts the surface layer does not own
(the TEMF PBL's new TE_TEMF and HD_TEMF, moving winds, theta, moisture and
skin), as recorded float32 formulas; HFX and WM_TEMF carry.

Arms:

| arm | object | log10f |
|---|---|---|
| A | the WRF scalar build's own `module_sf_temfsfclay.o` (`-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize`) | C library |
| B | the pinned source at `-O0` | C library |
| C | arm A's object | correctly rounded (`log10f_correctly_rounded.c`) |

A and B must agree word for word (the script fails otherwise). A is the
fixture `temf-*-wrf.csv`, C is `temf-*-wrf-crlog10.csv`.

Receipts: `log10-probe.txt` (how often the C library's log10f misses the
correctly rounded word), `log10-exhaustive.txt` (WOOF's own
`log10_flt32.cuh`, compiled as C, against MPFR on all 2,139,095,039 positive
finite binary32 words), `compiler.txt`, `oracle-sha256sums.txt`.

Graded by `tests/test_temf_sfclay_wrf461_oracle.py` (CPU) and
`tests/test_temf_sfclay_wrf461_parity.py` (GPU).

Production and paired replay use the same surface column kernel. The paired
PBL launcher delegates its default surface call to woof.core.temf_sfclay.
Its stock-oracle surface arm remains confined to verify_stock_wrf=True.

The paired WRF oracle carries actual TEMF PBL output over 216 columns on
35, 45 and 61 levels for 30 steps. tests/test_temf_wrf461_parity.py compares
all 52 fields by full-field SHA-256 at every step; the independent 78-column
oracle reports per-field ULP against stock WRF and the CR-LOG10 arm.
MOL is not an output of TEMFSFCLAY. ZNT is an unchanged input.

Evidence scope: the PBL oracle's fixed arm contains the six changes listed
in tools/temf_wrf461_oracle/apply_fixes.py. Agreement with that fixed arm
is distinct from agreement with untouched WRF or forecast skill.
