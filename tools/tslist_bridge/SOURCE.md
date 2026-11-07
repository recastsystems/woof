# WRF station time series source record

Native pin: WRF v4.7.1, `f52c197ed39d`.
Only the public WRF source was used. No competitor source was inspected.

| File | SHA-256 | Used for |
| --- | --- | --- |
| `share/wrf_tsin.F` | `9710a33ce72a053df0b97ee5790c4df535608b0563ce6cffa65d8e1c6d124fea` | Fixed-column station input |
| `share/wrf_timeseries.F` | `f8ae2d927e7248efcc9d82dd4de7b85059019edce18f7f9db612da9befebdeb6` | Sampling arithmetic and text layouts |
| `dyn_em/solve_em.F` | `3322d34e0ad070cf4d891aa6dbc93be70ef8240b84eb5f366a149487550aef89` | Physical boundary-copy and final-forcing order |
| `share/mediation_nest_move.F` | `a373a0c64ad258e781208277521d1d154224ce3d0dfa91017cc542d9e4914c43` | Native once-only station placement across relocation |
| `LICENSE.txt` | `b30611d35ad1922bc741e86d21e1a897234c1a9c9e2af28076e021f565dfbec6` | Public-domain statement |

Source URLs have the form
`https://raw.githubusercontent.com/wrf-model/WRF/f52c197ed39d/FILE`.
The license was checked before transcription. These modules use no separately
licensed tables. The bridge's original code is under the repository license;
the transcribed WRF expressions and layouts retain this notice:

WRF was developed at the National Center for Atmospheric Research (NCAR) which is operated by the University Corporation for Atmospheric Research (UCAR). NCAR and UCAR make no proprietary claims, either statutory or otherwise, to this version and release of WRF and consider WRF to be in the public domain for use by any person or entity for any purpose without any fee or charge. UCAR requests that any WRF user include this notice on any partial or full copies of WRF. WRF is provided on an "AS IS" basis and any warranties, either express or implied, including but not limited to implied warranties of non-infringement, originality, merchantability and fitness for a particular purpose, are disclaimed. In no event shall UCAR be liable for any damages, whatsoever, whether direct, indirect, consequential or special, that arise out of or in connection with the access, use or performance of WRF, including infringement actions.

WRF® is a registered trademark of the University Corporation for Atmospheric Research (UCAR).

`calc_ts` rotates each neighboring vector before averaging unstaggered profile
winds. It always averages W to mass levels. Column water includes QC, QR, QI,
QS and QG, with the native species and level accumulation order. Hail and
water vapor are excluded. `calc_p8w` extrapolates the top pressure in log
space. Replacing these with sounding-column interpolation changes the result.

Strict replay reuses the repository's existing `glibc_flt32.cuh` log/exp
cores and tables. Its retained source notice identifies the MIT grant from
Arm optimized-routines. No new mathematical table was copied. The existing
`LICENSE-third-party.txt`, `NOTICE` and `licenses/` notices remain applicable.
Default sampling uses CUDA's fast log/exp and permits contraction.

The native profile format at `share/wrf_timeseries.F:981-983` is constructed
by assigning the level count to substring `(12:14)`. That assignment replaces
the opening parenthesis of the repeated group. The strict native gfortran
build accepts the resulting format and writes adjacent F13.5 values. The
Rust writer reproduces those observed bytes. The Fortran format test retains
the original string and substring assignment, including this compiler
extension, rather than testing a repaired format.

The Rust writer implements the non-solar, non-WRF-Chem eight-file layout.
The production owner is `woof/core/tslist.py`, with committed-step hooks in
`woof/core/model.py`, live-carrier sampling in CUDA, and buffered Rust I/O.
See `../tslist_wrf471_oracle/README.md` for stock-namelist admission,
proof commands and runtime lifecycle.

Native `solve_em.F` copies physical tangential wind ghosts in
`set_phys_bc_dry_2` before the final specified boundary forcing. That forcing
can change the last mass column while leaving its copied ghost unchanged.
The GPU captures only the needed tangential columns at that same phase, then
combines those packets with the final normal faces. Interior neighbors come
from their canonical committed columns, including across streamed tiles.
Rotation uses the copied edge coefficients. No fresh post-forcing copy or
redundant tile halo replaces native data. Real native phase witnesses and
strict production-kernel replay prove both physical edges.

`share/module_bc.F`, `dyn_em/start_em.F`, `dyn_em/solve_em.F` and
`share/mediation_nest_move.F` at the same native pin were checked under the
public-domain statement above before use. They introduce no separately
licensed tables. The first two supply boundary and rotation-copy semantics;
the latter two are recorded in the source table above. Their complete
source hashes accompany the qualification evidence.

Native placement is guarded by the domain's `have_calculated_tslocs` flag.
A nest move calls `start_domain` without clearing that flag. Station indices
and headers therefore remain those selected initially; replacement carriers
must be rebound with the saved Rust cursor. Reprojecting the geographic
station after each move would change this native behavior.
