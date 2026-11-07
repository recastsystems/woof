# WRF 4.7.1 miscellaneous options

`mp_physics=95` is the old Eta Ferrier scheme (`ETAMP_NEW`), distinct from
Ferrier-Aligo selector 5. It transports total condensate QT and retains its
ice, rain and rime fractions. Native WRF leaves reflectivity unwritten.
Use the native legacy RRTMG radiation pair; RRTMGP optics are not qualified.

`sf_sfclay_physics=89` prescribes heat flux, latent heat flux and skin
temperature from six scalar state controls, including their time tendencies.
They are state variables, not namelist parameters. Cold wrfinput restores
them. Use PBL0 and LSM0. A physically valid forcing stream is required for
a useful forecast; the native zero skin-temperature initialization is retained.
Native-unwritten U10, V10, TH2, T2 and Q2 retain their incoming words.

Strict native module replays use real GFS forecast captures at steps 1, 2,
30 and 100 across eight tiles. Mp95 compares 20,604,340 words and sf89
compares 219,072 words, with zero differences and no exclusions. Native
WRF 4.7.1 is REAL32, GNU Fortran dmpar with `-O2 -fno-fast-math
-ffp-contract=off -fno-tree-vectorize`. Native and GPU forecasts each
complete three hours. This verifies code against WRF; it does not establish
skill against observations or equality of whole forecast trajectories.

Permanent inputs, reference outputs, native executable/configuration,
commands, hashes, sizes and replay scripts are under
`~/weather/data/datasets/paper/wrf-ports/port-misc-schemes/` in
`mp95-pbl0-cu0/MANIFEST.md` and `sf89-valid/MANIFEST.md`. Set
`GPUWM_WRF_EXACT=1`, then run each directory's
`python replay.py captures replay-current.json` with the installed engine.
No native-written word is excluded.

One A-B-B-A batch of complete fast forecasts on the same RTX PRO 6000
Blackwell compares mp95 with mp5 and sf89 with sf1. CUDA events cover the
actual stages; first calls and parent preflight calls are excluded. The
ratio of pooled warm medians is **0.907x mp5** for mp95 (0.713 versus
0.786 ms), and **0.138x sf1** for sf89 (0.099 versus 0.716 ms). There
are 358 warm samples per scheme. Every arm checks free card memory and
uses the same immutable engine build. No timing reruns were selected.
The receipt is [misc-wrf471.json](receipts/misc-wrf471.json).

The Rust temperature and QPF maps were opened. Mp95 shows a broad
temperature gradient, warmer south/east, with rain over Arkansas and
west Tennessee and no missing tiles. Sf89 shows a broad temperature
gradient and patchy rain bands, with two small warm spots west and
southwest of Little Rock and no missing tiles. Full logs, timing streams,
configs, scripts and images are retained by the maintainers
(`evidence/port-misc-schemes/`).

The option-off comparison against the matching upstream engine has zero
differences across 133,938,800 stored field bytes in four forecast frames,
including integers and NaN payloads. This compares stored field bytes,
not NetCDF headers. The receipt names the exact earlier commit pair;
later upstream changes are not attributed to this port.

`mp_physics=27` is WRF 4.7.1 UDM, with six condensate masses, three number
fields and its own initialized table words. It is distinct from WDM7.
The complete native driver replays **0/21,260,544 words** over 32 real-input
calls, without exclusions. The independent native repeat has identical
outputs. Native radar is enabled; native scheme radii are disabled. Use
legacy RRTMG: WRF's optional UDM radius routine passes uninitialized
`den1d` (`module_mp_udm.F:289,411`), so that coupling has no defined native
reference. No radius or observation-skill qualification is claimed.

All four fast public `woof go` forecasts in the interleaved UDM/WDM7 batch
complete 180 steps and four Rust-rendered frames. The MP27 stage median is
16.393215 ms versus MP26 14.257680 ms, **1.149781x**, from 358 warm calls
per scheme on the same shared RTX 5070 Ti and immutable wheel. The F003
temperature and QPF maps were opened: broad temperature gradient, warmer
south, localized warmth near Dallas and west Tennessee, rain over northwest
Arkansas and west Tennessee, no missing tiles or checkerboard.

UDM's native evidence is kept under `mp27-driver/`, its independent repeat
under `mp27-driver-witness/`, initialized tables under `mp27-gfs/`, and
initialized radar state under `mp27-radar-state/` in the permanent paper
root above. The replay command is in the receipt. Logs, card checks, timing
streams, maps and recipes are under `evidence/port-misc-schemes/udm-node4/`
in Downloads. Option off against matching upstream
`94199e02b96002613e581c549721795abb35296e` matches all 135,592,576 stored
field bytes over four complete forecast frames, including integer words
and NaN payloads. NetCDF headers are outside this comparison.

`mp_physics=29` is the WRF 4.7.1 RCON warm-rain revision of aerosol-aware
Thompson. Its complete driver replays 25,246,820 native-written words
across 32 real calls with zero differences and no exclusions, with radar
and effective radii enabled. A three-hour fast forecast and inspected Rust
temperature and QPF maps pass. Captures and native inputs are retained in
`mp29-driver/` and `mp29-refl/` beneath the permanent paper directory above.

RCON remains unqualified because its GPU stage costs **1.948x MP28**
(8.680 versus 4.455 ms), measured in one A-B-B-A batch of whole forecasts
on the same card and immutable build, with 358 warm samples per scheme.
The strict native driver profile costs only **1.054x MP28** (33.070 versus
31.383 ms), so it does not support an exception for that GPU cost.
Timing streams, native per-rank profiles and commands are retained under
`evidence/port-misc-schemes/{rcon-timing,rcon-native-profile,final}/` in
the Downloads directory above. Hardware-counter profiling was refused
with `ERR_NVGPUCTRPERM`; no system permissions were changed.
