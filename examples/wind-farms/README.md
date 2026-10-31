# Turbine forcing demo

The accompanying `forecast.toml` describes the 24-hour, 3 km demo. Set its
forcing and geography paths to your local input locations. Run preparation
without a GPU reservation, then run the prepared forecast on a GPU:

```sh
woof go forecast.toml --wind-farms uswtdb --prepare-only --outdir prepared
woof go forecast.toml --wind-farms uswtdb --prepared-root PREP_ROOT --wps-namelist WPS_PATH --outdir farm-forecast
```

The preparation receipt prints `PREP_ROOT` and `WPS_PATH`. To compare with
no turbines, prepare and run the same configuration without `--wind-farms`.
Use `--wind-farms /data/turbines/windturbines.txt` for a custom WRF table.
Read `docs/wind-farms.md` for input units, output fields, measured results
and limitations.
