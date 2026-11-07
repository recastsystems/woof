# Noah LSM oracle, WRF v4.6.1

Four fixture files, produced by `tools/noah_wrf461_oracle/build.sh` from the
byte-unmodified WRF v4.6.1 tree pinned at
`d66e442fccc04111067e29274c9f9eaccc3cef28`.  `build.sh` refuses to run against
any other commit, and refuses to run if `git diff` reports a change to any of
the six sources it compiles, so no number here can come from an edited tree.

| file | driver switches |
| --- | --- |
| `noah-lsm.csv` | `opt_thcnd=1`, `frpcpn=F`, `usemonalb=F`, `rdlai2d=F` |
| `noah-lsm-thcnd2.csv` | `opt_thcnd=2` (McCumber-Pielke thermal conductivity) |
| `noah-lsm-frpcpn.csv` | `frpcpn=T` (frozen fraction read from `SR`) |
| `noah-lsm-monalb.csv` | `usemonalb=T`, `rdlai2d=T` |

Each is 80 columns x 115 fields: every input `phys/module_sf_noahdrv.F`'s
`lsm` reads, every output it writes, and the T2, TH2 and Q2 that
`phys/module_sf_sfcdiags.F`'s SFCDIAGS then writes from them, called as
`module_surface_driver.F:2994` calls it.  Nothing in them is a hand-computed
expectation -- each value is either a word `run_lsm.F90` placed in an input
array before the call or a word `lsm` placed in an output array during it.
The two derived echoes, `sfcprs` and `zlvl`, are labelled as such in
`run_lsm.F90` and are inputs to gpuwm where WRF forms them from `p8w3d` and
`dz8w`.

The parameter tables are read by WRF's own `SOIL_VEG_GEN_PARM`
(`module_sf_noahdrv.F:1999`) from `run/VEGPARM.TBL`, `run/SOILPARM.TBL` and
`run/GENPARM.TBL` of the same pinned tree, so the fixture also exercises
gpuwm's transcription of that list-directed READ sequence rather than assuming
it.

`libmvec-report.txt` records which libm symbols the reference object actually
carries at `-O0`, at WRF's own `-O2 -ftree-vectorize`, and at `-Ofast`, plus
the positive control proving the `_ZGV*` grep can find a vector symbol on this
toolchain at all.  `oracle-sha256sums.txt` hashes the six WRF sources, the
harness, the three tables and the four fixtures together.

Toolchain: gfortran 13.3.0, glibc 2.39 (Ubuntu 24.04 under WSL2).

Rebuilt 2026-10-05 for the 2.8.6 land-ice port with thirteen land-ice
columns (43-55) added to `run_lsm.F90`, on gfortran 13.3.0 / glibc
2.39-0ubuntu8.9 (Ubuntu 24.04) against a sparse checkout of the same pinned
commit, every source hash as above.  Before adding them the unchanged harness
rebuilt all four committed CSVs byte-identical on that toolchain, and with
them present rows 1-42 of every CSV are still byte-identical.

Rebuilt 2026-10-07 on the test host (gfortran 13.3.0, glibc 2.39, Ubuntu 24.04)
for development branch, against the full WRF v4.6.1 checkout of the same
pinned commit.  Twenty regime columns (56-75) were added first: convective
downpour, outflow, high terrain in summer and winter, dewfall, hoar frost,
desert, rainforest, two ocean columns, wetland, freezing rain and melt on
frozen ground, a frozen top layer, tundra, irrigated crop, -40 C over deep
snow, a cloudburst, and VEGFRA at both table bounds; rows 1-55 rebuilt
byte-identical with them present.  Then CHS, CHS2 and CQS2 after `lsm` and
SFCDIAGS's T2, TH2 and Q2 became six more fields at the end of every row,
and five columns (76-80) were added for that hand-over: CHS2 different from
CQS2 on entry, CQS2 just below and exactly at SFCDIAGS's 1.E-5 threshold, a
70 kPa plateau at night, and open water with CHS2 different from CQS2.  Every
earlier field of rows 1-75 is byte-identical to the 75-column build.
