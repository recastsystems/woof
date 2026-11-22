# MMM-physics 550b5b4 module replay (YSU top-down cap, revised MM5 KIM form drag)

Referee tools for `mmm_version` (woof/hex/mpasphys/mmm_version.py). CPU Fortran here is a referee only; it is never on
a run path.

Sources (pristine, fetched from GitHub):

| file | commit | sha256 |
|---|---|---|
| NCAR/MMM-physics `bl_ysu.F90` | a4baf7f3243d | b2c9f11f92f967b053948b42a7c5dbaf1b7f72ddb9d35948e96c32a5d91092d2 |
| NCAR/MMM-physics `bl_ysu.F90` | 550b5b494758 | 15d76892fc6ffa5c9f8f7ebecfeb6eaa5c23d71b44e8c53a8832f49ff711ea7e |
| NCAR/MMM-physics `sf_sfclayrev.F90` | a4baf7f3243d | 76412a20ec95275973418888b0cdc378ea92f9cc28e3731b6a1c1a6985fa6142 |
| NCAR/MMM-physics `sf_sfclayrev.F90` | 550b5b494758 | 1bff41ec6d8d2bf73a07a40c5a6dfac2565b46601c9ee3a9b0f6aa6fe85ed698 |
| MPAS-Model v8.4.2 `physics_wrf/module_bl_ysu.F` | v8.4.2 | a1f21428c48598afef912ece6901db1a15890e72a31944fe735a19841a45071c |

Steps:

1. `build_native.sh WORK SRC` builds `libysu.so` (MPAS v8.4.2 wrapper + MMM `bl_ysu.F90`, through
   `tests/mpas_ysu_v842/native_bridge.F90`) and `libmm5.so` (MMM `sf_sfclayrev.F90` through `mm5_bridge.F90`;
   `-DMMM550` passes `varf`, `if_kim_tofd`, `tofd_factor`) for both commits, with the MPAS strict GNU flags.
2. Capture: run a hex forecast with `PYTHONPATH` holding `capture_sitecustomize.py` (renamed `sitecustomize.py`) and
   `L5_CAPTURE_DIR`, `L5_CAPTURE_EVERY`, `L5_CAPTURE_MAX` set. It saves the inputs of `ysu_v842.ysu` (the MPAS wrapper
   level) and `sfclay_mm5_v842.launch_columns` (plus `var2d`) before each kept call; the run is unchanged.
3. `WOOF_HEX_NUMERICS=strict replay_l5.py CAPTURE WORK OUT.json [--native-ftz]` runs every captured call through
   the hex CUDA kernel and the native build at a4baf7f and at 550b5b4 (MM5 also with KIM drag on, 550b5b4 only) and
   compares every output word.
   - The pass bar is the plain IEEE native build (no `--native-ftz`): on the committed kernels (e04347ed), a fresh
     capture of this case (1 h, 5 YSU + 5 MM5 calls) is 0 of 88,805,920 words different at both commits, in strict
     and in fast numerics.
   - `--native-ftz` runs the native YSU columns with x86 flush-to-zero, a diagnostic only. On the committed
     kernels it ADDS differences (182,158 words, all in `rqcblten`/`rqiblten`: native flushes subnormal
     tendencies the CUDA kernel keeps), so the CUDA YSU kernel keeps IEEE subnormals like native. An earlier
     uncommitted state of this replay saw the opposite (163,924 words without FTZ, 0 with it); that does not
     reproduce on this commit and is not a hex YSU defect.
4. `ncwords.py A B OUT.json` compares two forecast directories word for word; `score_asos.py` scores arms against
   ASOS (T2, Td2, ceiling <= 1500 ft, visibility <= 3 mi proxy); `sheets.py` lays the products door's maps side by
   side and draws the ASOS series chart.
