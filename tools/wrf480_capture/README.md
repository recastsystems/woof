# WRF 4.8.0 wrapper-argument captures

Real inputs and outputs of the native WRF 4.8.0 scheme wrappers, for bitwise module
replay of the WOOF-ArWen ports. Same method as `tools/wrf_instrumented`: the referee
binary stays pristine, and a separate instrumented copy only reads state and writes
dumps.

1. Inputs come from our own init, with no WPS and no real.exe: `woof fetch --source hrrr` and then
   `woof prep --source hrrr --domain-spec cases/iowa-convective-domain.json
   --wps-namelist cases/namelist.wps --namelist-input cases/<K>/namelist.input.461`
   write `wrf-native-input/wrfinput_d01` and `wrfbdy_d01`.
   `cases/<K>/namelist.input.480` is the same namelist with the keys 4.8.0 removed
   (`icloud_bl`) and no iofields file.
2. To instrument a copy of the source tree, run
   `python3 instrument.py <WRF-copy> targets-480.json`. It adds
   `frame/module_gpuwm_capture.F`, a step hook in `solve_em` and entry/exit dumps in
   each target, and writes `GPUWM-CAPTURE-RECEIPT.json`. Build it with the pinned strict recipe
   (`VARIANT=capture`).
3. To run it, use `GPUWM_CAPTURE_DIR=<dir> GPUWM_CAPTURE_STEPS=1,2,50,100,240 run_native.sh <rundir> <name>`.
4. To prove the copy changes nothing, run
   `compare_history.py <instrumented wrfout> <referee wrfout>`. It must report 0 differing words.
5. To read the dumps, use `capture_reader.read_capture(path)`. It returns `{arg: Arg(value, lbound)}`, and
   `capture_index(dir)` keys the files by (subroutine, step, call).
