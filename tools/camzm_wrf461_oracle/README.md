# CAM Zhang-McFarlane oracle (WRF v4.6.1, cu_physics = 7)

Column oracle for the port of WRF's CAMZMSCHEME: `phys/module_cu_camzm_driver.F`
(camzm_driver, get_tpert, zm_conv_init) and `phys/module_cu_camzm.F`
(zm_convi, zm_convr with buoyan_dilute, parcel_dilute, entropy, ientropy,
cldprp, closure and q1q2_pjr; zm_conv_evap; momtran; convtran), with the CAM
support modules they USE (wv_saturation's aqsat and table, cldwat_fice,
physconst, constituents).  The port is `woof/core/camzm.py` and
`woof/core/kernels/camzm.cu`; `tests/test_camzm_launcher_wrf461_parity.py`
grades it on these fixtures word for word.

This recovered legacy harness uses generated soundings. Its corrected-source
variants are attribution instruments only. They are not native WRF parity gates.
The current column kernel uses stock WRF's XLAND test and momtran grouping.
Use `tools/cam5_wrf471_oracle/build.py` for the pinned 4.7.1 stock-module check.
A real forecast capture remains required before admission.

## Files

| file | what it is |
|---|---|
| `SOURCES.sha256` | sha256 of every WRF v4.6.1 file compiled or cited (git d66e442) |
| `make_cases.py OUTDIR` | the column cases: 97 synthetic soundings in four tiles |
| `stub_wrf.F90` | the framework names the CAM files USE that carry no physics |
| `zmio.F90` | fixture writer (`<stem>.bin` + `<stem>.manifest`) |
| `run_camzm.F90` | the full-driver oracle: CAM_INIT, zm_conv_init, then camzm_driver for N consecutive steps, every step's inputs and outputs recorded |
| `fixes.patch` | historical altered-source attribution experiments (below) |
| `compare_variants.py` | per-field count of words the stock and fixed builds differ |
| `build.sh WRF_SRC BUILD [CASES]` | builds the variants, runs every case, gates, collects `BUILD/fixtures` |

## Build variants

* **stock**: byte-unmodified WRF at the flags of the WRF 4.6.1 strict-scalar
  build (`-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize`), WRF's
  preprocessor defines for an EM RWORDSIZE=4 build.  WRF as it ships.
* **fixed**: the same with `fixes.patch`.  It is an altered-source diagnostic only.
* **o0**: fixed at -O0.  Required byte-identical to fixed.
* **snan**: fixed with every local real a signalling NaN and every local
  integer -999999.  Required byte-identical: no uninitialised local reaches
  an output on the cases run.
* **stock, XLAND exchanged**: stock WRF handed XLAND with 1 and 2 swapped.
  Required byte-identical to fixed (see the landfrac defect below).

## Historical altered-source experiments

1. **Land and ocean are inverted in the precipitation efficiency.**
   `module_cu_camzm_driver.F:459-463` sets `landfrac = 1` ("land") when
   `XLAND == 2`, but WRF's XLAND is 1 for land and 2 for water.  cldprp then
   uses the ocean autoconversion rate `c0_ocn = 0.045` over land and the land
   rate `c0_lnd = 0.0059` over the ocean, an 8x swap in how fast updraft
   condensate rains out.  The historical corrected variant uses the land rate over land.  This moves
   every convecting column (stock-vs-fixed.txt counts the words).  The
   exchanged-XLAND run proves the fix is exactly the inversion of that one
   test: stock WRF fed swapped XLAND equals the fixed build word for word on
   every fixture, and camzm_driver reads XLAND nowhere else.
2. **A misplaced parenthesis in momtran's first downdraft level.**
   `module_cu_camzm.F:1633` divides only the pressure-gradient term by the
   mass flux: `(-ed*u*dp)-pgd*dp/md`, which is dimensionally inconsistent
   with the general downdraft formula three lines below
   (`(md*u - ed*u*dp - pgd*dp)/md`).  The historical corrected variant uses `(-ed*u*dp - pgd*dp)/md`.
   The branch needs a downdraft at interface 2 (`md(2) < -1e-15`), and
   cldprp cannot produce one: its downdraft starts at `jd >= j0 >= jt+2`
   with `jt >= limcnv+1 = 3`, so `jd >= 5`.  The fix therefore moves no word
   on any input; the exchanged-XLAND identity above (which includes this
   fix on the fixed side and not on the stock side) is the measurement, and
   tests/test_camzm_wrf461_parity.py also checks `md3d` at that interface
   is zero in every fixture.

## Measured Fortran semantics used by the port

* The CAM code is `real(r8)`; the driver's WRF-facing arithmetic is default
  REAL.  Mixed expressions follow Fortran's promotion: e.g.
  `rqvcuten = zmdq*(1._r8+qv)**2` widens the float32 `zmdq` and rounds once
  to float32 at the store, while `rqvcuten + evapqzm*(1.+qv)**2` is float32
  throughout.  The port follows each statement's own kinds.
* `ke = 1.0E-6` is a default-REAL literal stored into `real(r8)`: its value
  is the float32 rounding of 1e-6, widened.
* Real exponents are libm `pow`/`powf` calls (`(tl/t)**plexp`,
  `(1000/p)**(rd/cp)`, get_tpert's `**.33` and `**rcp`); integer exponents
  2, 3 and 4 are multiplication chains (`x*x`, `x*x*x`, `(x*x)*(x*x)`).
  libm-report.txt lists every libm symbol the objects of record call.
* `8.e-6_r8**3` and `25.e-6_r8**3` in the driver's detrained-number
  denominators are folded by the compiler; their words are recorded in the
  fixture (`const/qcn_den`, `const/qin_den`), not recomputed.
