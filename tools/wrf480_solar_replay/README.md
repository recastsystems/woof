# WRF 4.8.0 solar replay

WOOF-ArWen's radiation-call sun (`radconst` + `calc_coszen`, and the calendar
that feeds them) against the native strict referee build's own object
code, word for word. See docs/dev/solar-generations.md for the result.

On a development machine (`~/wrf480-port` holds the referee builds):

```
podman run --rm -v $HOME/wrf480-port:/w -v $PWD:/r -w /r localhost/wrfref2404:1 \
  bash /r/tools/wrf480_solar_replay/build_native.sh /w/build/WRF-4.8.0-serial-pristine /r/native-480
python -m tools.wrf480_solar_replay.replay inputs RUN
for g in iowa3km world2deg; do ./native-480 RUN/$g.in RUN/$g.host480; done      # host libm
python -m tools.wrf480_solar_replay.replay gpu RUN --generation wrf_480 --path solar
python -m tools.wrf480_solar_replay.replay gpu RUN --generation wrf_480 --path geometry_at
python -m tools.wrf480_solar_replay.replay compare RUN host480 gpu-wrf_480-solar
```

The native binary is dynamically linked. Its ASIN result depends on the
runtime glibc, so say which libm it ran on. `timing.py` times one radiation
call's calendar plus kernel on the A071 grid.
