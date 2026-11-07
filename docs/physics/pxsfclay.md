# Pleim-Xiu surface layer, WRF selector 7

The CUDA column implementation covers PXSFCLAY1D, the PXSFCLAY driver,
PXSFCLAY_SEAICE_WRAPPER and get_local_ice_tsk. PXSFCLAYINIT is empty.
There are no external scheme tables.

Source pins and licence

- WRF v4.6.1 commit d66e442fccc04111067e29274c9f9eaccc3cef28,
  phys/module_sf_pxsfclay.F, SHA-256
  cd1b2095f8e093d4bc423ebbc4a3bfbce58a7d17b34f1f41d770c6061b68d0a4.
- WRF phys/module_surface_driver.F:6687-6972 and :7117-7208.
  Both routines and the complete PX module are byte-identical in v4.7.1.
- WRF share/module_model_constants.F supplies the physical constants.
- The source files are distributed under WRF's public-domain statement:
  https://github.com/wrf-model/WRF/blob/v4.7.1/LICENSE.txt.
  No separate restrictive PX notice or scheme table is present.
- Scalar math reuses the repository's existing MIT Arm and CORE-MATH
  implementations and permissive SunPro arithmetic. Their notices remain
  in the kernel headers, NOTICE and licenses/. No new external library ships.
- SOURCE-MANIFEST.json records each checked source and licence URL, file
  size and SHA-256, including Registry metadata and physical constants.

Runtime and pairing

Set sf_sfclay_physics=7 in a stock WRF namelist or RunConfig. WRF's
isfc=7 pairing law refuses YSU, MYNN, Shin-Hong and MYJ. The current engine
admits PBL-off, UW and SASE. Every published field belongs to the existing
surface state, including restart/checkpoint, history, ensembles and nested
state transfer. RUC uses the existing fractional sea-ice carriers.

The default compiles CUDA math with contraction allowed. Strict proof uses
GPUWM_WRF_EXACT=1, with GPUWM_WRF_GLIBC naming the native referee's library
(default 2.39; 2.41 and newer use the existing CORE-MATH atan implementation).
Pinned constants are validated once before launch and cached on the host.
The forecast path has no new synchronization or device-to-host transfer.

Proof scope

The tools/pxsfclay_wrf461_oracle constructed-column fixtures exercise five
configurations, regimes, carried state, land-model flux replacement and both
fractional sea-ice thresholds. A separate native capture fixture records
12 calls inside a real 300 by 300 WRF forecast: the first row of four MPI
tiles at each of the first three calls. Strict replay matches all 43,200
output words, using the native referee's glibc 2.43 arithmetic. Reproduce it
with `GPUWM_WRF_EXACT=1 GPUWM_WRF_GLIBC=2.43 python -m
woof.verify.pxsfclay_native_oracle --report native-px-replay.json`.

The stock namelist door is `woof go --wrfinput DIR --outdir DIR`, where
the input directory contains namelist.input, wrfinput_d01 and wrfbdy_d01.
All 24 PX output/state words are exposed in history. Registry fields keep
their native metadata; five module-local transfer diagnostics are explicitly
WOOF history extensions. Both nest feedback modes, one- and four-member
ensembles, restart continuation and both RUC ice thresholds have device
checks.

One real 300 by 300 by 49 forecast completed all 1,440 steps over six
hours through that door in default arithmetic, with a passing final health
check. It used WSM6, Noah, UW and RRTM/Dudhia for 2026-10-16 09-15 UTC.
Native WRF also completed all six hours. Real ASOS observations and native
WRF were compared by eye at two, four and six hours with Rust-rendered
temperature, dewpoint and wind maps. The final sensible heat, latent heat
and friction-velocity maps were also inspected. The large-scale patterns
agree closely, with small wind and flux differences. Both forecasts share
a dry bias against ASOS. Complete receipts belong to the lane handoff;
this is one case, without a claim of general forecast skill or whole-
trajectory bitwise equality.

On RTX 4090, NVRTC 12.9.86, the default column stage took 0.008105 ms for
90,000 real-input columns, versus 0.114212 ms for classic MM5, measured
with 20 warmups and 200 CUDA-event calls. Both PX kernel entries used
zero local-memory bytes in default and strict modes. Other architectures
have not been measured; the frame record is explicitly partial.

The optional sea-ice wrapper took 0.012104 and 0.023910 ms on the existing
mixed-ice oracle fixtures resized to 90,000 columns. Classic MM5 plain
columns on identical forcing took 0.168279 and 0.161736 ms, ratios 0.072
and 0.148; classic wrapper overhead is excluded. Those measurements used
the same device, default arithmetic, 20 warmups and 200 CUDA-event calls.
