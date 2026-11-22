# WRF 4.8.0 whole-forecast gate (port/wrf-480 assembly)

The referee is native WRF 4.8.0 strict serial (`WRF-4.8.0-serial-pristine`,
wrf.exe f6f47f86) run inside its build container `localhost/wrfref2404:1`
(Ubuntu 24.04, glibc 2.39). WOOF-ArWen runs in that same container with the GPU
passed through (`run_container.sh`), so both sides read one C library.
Strict solar declination takes the host C library's `asinf` (woof.core.solar),
so on a glibc 2.41+ host the run would differ from a glibc 2.39 referee by
one ulp of DECLIN on some radiation calls; run the gate in the container.

    # The scripts read their locations from the environment:
    #   WRF480_ROOT   referee working root (build/WRF-4.8.0-serial-pristine, gate/, captures/)
    #   WOOF_TREE    the engine checkout; GATE_PYTHON its venv python;
    #   GATE_CUDA_PATH  the nvidia cu13 wheel root inside that venv
    # Rust bridges for the container (glibc 2.39), once per tree:
    podman run --rm --userns=keep-id -v $HOME:$HOME -w <tree>/tools/rustwx \
      -e HOME=$HOME -e PATH=$HOME/.cargo/bin:/usr/bin:/bin \
      -e CARGO_TARGET_DIR=<tree>/tools/rustwx/target-c239 localhost/wrfref2404:1 \
      cargo build --release --locked --offline -p static-fields -p netcdf-writer
    # Native referee run dir for a case (tables, wrf.exe f6f47f86, our own init's
    # wrfinput/wrfbdy from gate/<CASE>-in, cases/<CASE>/namelist.input and the
    # iofields file the deck names, so native writes the full history shape), then run it:
    ./native_rundir.sh A071 <gate>/A071-native
    ./run_native.sh <gate>/A071-native A071-native
    # A071 on WRF 4.8.0 (GFL, 4.8.0 YSU, 4.8.0 sun), strict, one hour:
    EXTRA_ENV="-e GPUWM_WRF_EXACT=1" ./run_container.sh A071 A071-strict \
      --scheme cu_gf_version=gfl_873b414 --scheme solar_generation=wrf_480 \
      --scheme ysu_version=wrf_480
    python compare_hour.py <native wrfout (241 frames)> <gate>/A071-strict/wrfout out.json
    # A071 full shape on 1a4028d1f74: 241 frames, 215 fields, 654,871,971 words, 0 different.
    # A195, the HRRR recipe (Thompson, MYNN-EDMF 8f8e8b59, MYNN-SFC b66187f, RUC), strict,
    # one hour. RUC must name the stock-WRF SOILPROP (ruc_soilprop=wrf_461): the engine
    # default is the operational form, and without it native leaves at step 1.
    ./native_rundir.sh A195 <gate>/A195-native
    ./run_native.sh <gate>/A195-native A195-native
    EXTRA_ENV="-e GPUWM_WRF_EXACT=1" ./run_container.sh A195 A195-strict \
      --scheme bl_mynn_version=wrf_480 --scheme mynn_sfclay_variant=mynnsfc_b66187f \
      --scheme solar_generation=wrf_480 --scheme ruc_soilprop=wrf_461
    python compare_hour.py <native wrfout (241 frames)> <gate>/A195-strict/wrfout out.json
    # A195 on e8e6f5588d6: 241 frames, 221 fields, 605,907,981 words, 0 different
    # (native: pristine f6f47f86, no Thompson patch).

`score_obs.py` scores arms at +1 h against ASOS (T2, Td2, 10 m wind) and
MRMS (composite reflectivity, 1 h multi-sensor QPE) fetched with `woof obs`.
